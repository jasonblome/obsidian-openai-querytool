from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Literal
from urllib.parse import quote

import typer
from openai import OpenAI
from rich import box
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

app = typer.Typer(help="Index and query Obsidian markdown notes in an OpenAI Vector Store.")
console = Console()


# --------------------------- Data & Types ---------------------------

@dataclass(frozen=True)
class RemoteFile:
    """A simplified view of a Vector Store file row."""
    id: str
    file_id: str
    attributes: Dict[str, str]
    created_at: int


@dataclass(frozen=True)
class RemoteIndexRow:
    """Row stored in the remote index mapping."""
    id: str
    uploaded_ts: int
    sha256: Optional[str]
    mtime_attr: Optional[int]


RemoteIndex = Dict[str, RemoteIndexRow]
FileWithAttrs = Tuple[Path, Dict[str, str]]


# --------------------------- Client Helpers ---------------------------

def get_client() -> OpenAI:
    """
    Create an OpenAI client. Requires OPENAI_API_KEY in the environment.
    """
    return OpenAI()


def ensure_vector_store(
    client: OpenAI,
    vector_store_id: Optional[str],
    vector_store_name: Optional[str],
) -> str:
    """
    Return an existing Vector Store id or create a new one by name.
    Priority: explicit id > create (or reuse) by name.
    """
    if vector_store_id:
        return vector_store_id

    if not vector_store_name:
        raise typer.BadParameter("Provide either --vector-store-id or --vector-store-name.")

    vs = client.vector_stores.create(name=vector_store_name)
    return vs.id


# --------------------------- Vector Store I/O ---------------------------

def iter_vs_files(
    client: OpenAI, vector_store_id: str, page_size: int = 200
) -> Generator[RemoteFile, None, None]:
    """
    Page through Vector Store files and yield RemoteFile objects.
    """
    after: Optional[str] = None
    while True:
        page = client.vector_stores.files.list(
            vector_store_id=vector_store_id,
            limit=page_size,
            after=after,
        )
        for f in page.data:
            yield RemoteFile(
                id=f.id,
                file_id=getattr(f, "file_id", f.id),
                attributes=getattr(f, "attributes", {}) or {},
                created_at=int(getattr(f, "created_at", 0) or 0),
            )

        if not getattr(page, "has_more", False):
            break
        after = page.data[-1].id


def build_remote_index(client: OpenAI, vector_store_id: str) -> RemoteIndex:
    """
    Returns {obsidian_path: RemoteIndexRow}. `uploaded_ts` is compared to local mtime.
    """
    idx: RemoteIndex = {}
    for f in iter_vs_files(client, vector_store_id):
        attrs = f.attributes or {}
        mtime_attr = _safe_int(attrs.get("mtime"))
        uploaded_ts = mtime_attr or f.created_at
        path_key = attrs.get("obsidian_path") or f.file_id

        idx[path_key] = RemoteIndexRow(
            id=f.id,
            uploaded_ts=uploaded_ts,
            sha256=attrs.get("sha256"),
            mtime_attr=mtime_attr,
        )
    return idx


def build_fileid_to_path(client: OpenAI, vector_store_id: str) -> Dict[str, str]:
    """Map vector-store file_id -> obsidian_path (if available)."""
    out: Dict[str, str] = {}
    for f in iter_vs_files(client, vector_store_id):
        attrs = f.attributes or {}
        path = attrs.get("obsidian_path")
        if path:
            out[f.file_id] = path
    return out


# --------------------------- Local File Scanning ---------------------------

def file_attrs(p: Path) -> Dict[str, str]:
    """
    Compute attributes for a local file to store alongside the upload.
    """
    data = p.read_bytes()
    return {
        "obsidian_path": str(p),
        "mtime": str(int(p.stat().st_mtime)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "kind": "obsidian-md",
    }


def choose_changed_files(vault_root: Path, remote_index: RemoteIndex) -> List[FileWithAttrs]:
    """
    Decide which files need upload: new or strictly newer (local mtime > uploaded_ts).
    """
    to_upload: List[FileWithAttrs] = []
    for p in _iter_markdown_files(vault_root):
        local_mtime = int(p.stat().st_mtime)
        attrs = file_attrs(p)
        remote = remote_index.get(str(p))
        if (remote is None) or (local_mtime > int(remote.uploaded_ts or 0)):
            to_upload.append((p, attrs))
    return to_upload


# --------------------------- Upload / Tag ---------------------------

def upload_changed_files(
    client: OpenAI,
    vector_store_id: str,
    files_with_attrs: Iterable[FileWithAttrs],
) -> Dict[str, object]:
    """
    Upload changed files and patch their attributes.
    Returns {"uploaded": int, "file_ids": [str, ...]}.
    """
    files_with_attrs = list(files_with_attrs)
    if not files_with_attrs:
        return {"uploaded": 0, "file_ids": []}

    streams = [_open_rb(p) for p, _ in files_with_attrs]
    try:
        _ = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=streams,
        )
    finally:
        for s in streams:
            try:
                s.close()
            except Exception:
                pass

    # Rebuild index to discover new ids
    remote_after = build_remote_index(client, vector_store_id)

    updated_ids: List[str] = []
    for p, attrs in files_with_attrs:
        rid = remote_after.get(str(p), RemoteIndexRow("", 0, None, None)).id
        if rid:
            client.vector_stores.files.update(
                vector_store_id=vector_store_id,
                file_id=rid,
                attributes=attrs,
            )
            updated_ids.append(rid)

    return {"uploaded": len(updated_ids), "file_ids": updated_ids}


# --------------------------- Q&A / Citations ---------------------------

@dataclass(frozen=True)
class Reference:
    file_id: str
    path: Optional[str]
    quote: Optional[str]


@dataclass(frozen=True)
class Answer:
    text: str
    references: List[Reference]


def ask_question(client: OpenAI, vector_store_id: str, question: str) -> Answer:
    """
    Ask a question using Responses API + file_search against the Vector Store.
    Attempts to collect file citations and map them back to local file paths.
    """
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": question}],
        tools=[{"type": "file_search"}],
        tool_choice="auto",
        metadata={"app": "obsidian-openai-repl"},
        search_options={"vector_store_ids": [vector_store_id]},
    )

    text = _extract_text(resp)
    raw = _to_dict(resp)
    file_ids = _collect_file_ids_from_citations(raw)

    id_to_path = build_fileid_to_path(client, vector_store_id)

    refs: List[Reference] = []
    for fc in _collect_file_citations(raw):
        f_id = fc.get("file_id")
        refs.append(
            Reference(
                file_id=f_id or "",
                path=id_to_path.get(f_id) if f_id else None,
                quote=fc.get("quote"),
            )
        )

    # Add any missing ids without quotes
    for fid in file_ids:
        if all(r.file_id != fid for r in refs):
            refs.append(Reference(file_id=fid, path=id_to_path.get(fid), quote=None))

    return Answer(text=text, references=refs)


# --------------------------- High-level Workflows ---------------------------

def index_vault(client: OpenAI, vector_store_id: str, vault_root: Path) -> Dict[str, object]:
    """
    End-to-end index: build index, diff, upload, tag. Returns upload result.
    """
    _validate_vault_root(vault_root)
    remote_idx = build_remote_index(client, vector_store_id)
    changed = choose_changed_files(vault_root, remote_idx)
    result = upload_changed_files(client, vector_store_id, changed)
    return result


# --------------------------- CLI Commands ---------------------------

@app.command()
def index(
    vault_root: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Path to your Obsidian vault root.",
    ),
    vector_store_id: Optional[str] = typer.Option(None, help="Existing Vector Store id."),
    vector_store_name: Optional[str] = typer.Option(None, help="Name (create or reuse) for the Vector Store."),
) -> None:
    """
    One-shot indexing of the vault into the Vector Store.
    """
    client = get_client()
    vs_id = ensure_vector_store(client, vector_store_id, vector_store_name)

    console.rule("[bold]Indexing Vault")
    result = index_vault(client, vs_id, vault_root)

    table = Table(title="Upload Summary", box=box.SIMPLE)
    table.add_column("Uploaded", justify="right")
    table.add_column("File IDs", overflow="fold")
    table.add_row(str(result["uploaded"]), json.dumps(result["file_ids"]))
    console.print(table)


@app.command()
def repl(
    vault_root: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Path to your Obsidian vault root.",
    ),
    vector_store_id: Optional[str] = typer.Option(None, help="Existing Vector Store id."),
    vector_store_name: Optional[str] = typer.Option(None, help="Name (create or reuse) for the Vector Store."),
    link_scheme: Literal["obsidian", "file"] = typer.Option(
        "obsidian",
        help="How to render reference links: 'obsidian' (obsidian://open?path=...) or 'file' (file://...).",
    ),
) -> None:
    """
    Start a small REPL to (re)index and ask questions against your Vector Store.
    Commands:
      - index                : index changed files
      - ask <question text>  : query your notes
      - exit / quit          : leave
      - help                 : show commands
    """
    client = get_client()
    vs_id = ensure_vector_store(client, vector_store_id, vector_store_name)
    console.print(f"[bold green]Vector Store:[/bold green] {vs_id}")

    while True:
        try:
            raw = Prompt.ask("[cyan]obsidian>[/cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not raw:
            continue
        if raw in {"exit", "quit"}:
            console.print("Bye!")
            break
        if raw == "help":
            console.print("Commands: index | ask <question> | exit | quit | help")
            continue
        if raw == "index":
            result = index_vault(client, vs_id, vault_root)
            console.print(f"Uploaded {result['uploaded']} files.")
            continue
        if raw.startswith("ask "):
            question = raw[4:].strip()
            if not question:
                console.print("[red]Provide a question after 'ask'.[/red]")
                continue
            ans = ask_question(client, vs_id, question)
            console.rule("[bold]Answer")
            console.print(ans.text)
            _print_references(ans.references, link_scheme=link_scheme)
            console.rule()
            continue

        console.print("[yellow]Unknown command. Type 'help'.[/yellow]")


# --------------------------- Utilities ---------------------------

def _validate_vault_root(vault_root: Path) -> None:
    if not vault_root.exists() or not vault_root.is_dir():
        raise typer.BadParameter(f"Vault root '{vault_root}' does not exist or is not a directory.")


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    return (p for p in root.rglob("*.md") if p.is_file())


def _open_rb(path: Path):
    return open(str(path), "rb")


def _safe_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _extract_text(resp) -> str:
    """
    Best-effort text extraction from the Responses API return type.
    Keeps this tiny and resilient to SDK changes.
    """
    if hasattr(resp, "output_text"):
        return resp.output_text  # newer SDKs
    # Fallbacks
    d = _to_dict(resp)
    # Try to walk output[...].content[...].text
    output = d.get("output") or d.get("response", {}).get("output")
    if isinstance(output, list):
        # look for text blocks
        texts: List[str] = []
        for block in output:
            content = block.get("content")
            if isinstance(content, list):
                for c in content:
                    t = c.get("text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t)
        if texts:
            return "\n".join(texts)
    # Last resort
    return str(resp)


def _to_dict(obj) -> Dict:
    """Best-effort conversion of SDK objects to a plain dict."""
    # Newer SDKs expose model_dump
    for attr in ("model_dump", "to_dict"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    # Try JSON then parse
    for attr in ("model_dump_json",):
        if hasattr(obj, attr):
            try:
                return json.loads(getattr(obj, attr)())
            except Exception:
                pass
    # Fallback: json round-trip via str() may fail; return empty
    return {}


def _collect_file_citations(d: Dict) -> List[Dict[str, str]]:
    """
    Traverse a responses dict to collect citation dicts like {file_id, quote}.
    Supports shapes seen in file_search annotations.
    """
    found: List[Dict[str, str]] = []
    output = d.get("output") or []
    if not isinstance(output, list):
        return found

    for block in output:
        content = block.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            # Common shapes: {type: "output_text", text: "...", annotations: [{type:"file_citation", file_id, quote, ...}]}
            ann = c.get("annotations") or []
            if isinstance(ann, list):
                for a in ann:
                    if a.get("type") in {"file_citation", "citation", "file"}:
                        entry = {"file_id": a.get("file_id"), "quote": a.get("quote")}
                        if entry["file_id"]:
                            found.append(entry)
    return found


def _collect_file_ids_from_citations(d: Dict) -> List[str]:
    return [c.get("file_id") for c in _collect_file_citations(d) if c.get("file_id")]


def _path_to_uri(p: str, scheme: Literal["obsidian", "file"]) -> str:
    """
    Convert a local path to a clickable URI for Rich.

    - obsidian: obsidian://open?path=<absolute-path-URL-encoded>
    - file: file://<absolute-path>
    """
    path = Path(p).absolute()
    if scheme == "obsidian":
        return f"obsidian://open?path={quote(str(path))}"
    # file:// fallback
    if os.name == "nt":
        return f"file:///{str(path).replace(os.sep, '/')}"
    return f"file://{path}"


def _print_references(refs: List[Reference], *, link_scheme: Literal["obsidian", "file"]) -> None:
    if not refs:
        return
    subtitle = "obsidian:// links" if link_scheme == "obsidian" else "file:// links"
    table = Table(title="References", box=box.SIMPLE, subtitle=subtitle)
    table.add_column("File")
    table.add_column("Quote", overflow="fold")

    seen = set()
    for r in refs:
        key = (r.file_id, r.path, r.quote)
        if key in seen:
            continue
        seen.add(key)
        path_display = r.path or f"(file_id: {r.file_id})"
        if r.path:
            uri = _path_to_uri(r.path, scheme=link_scheme)
            path_display = f"[link={uri}]{r.path}[/link]"
        table.add_row(path_display, r.quote or "")

    console.print(table)


# --------------------------- Entrypoint ---------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()