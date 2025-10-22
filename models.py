from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict


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
