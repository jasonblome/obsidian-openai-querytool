from typing import Optional, Dict, Generator

import typer
from openai import OpenAI
from openai.types import VectorStore

from obsidian_openai_querytool.models import RemoteIndex, RemoteIndexRow, RemoteFile


def get_client() -> OpenAI:
    """
    Create an OpenAI client. Requires OPENAI_API_KEY in the environment.
    """
    return OpenAI()


def get_or_create_vector_store(client: OpenAI, name: str) -> VectorStore:
    """
    Get or create a Vector Store by name.
    """
    stores = client.vector_stores.list()
    for vs in stores.data:
        if getattr(vs, "name", "") == name:
            return vs
    return client.vector_stores.create(name=name)


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
