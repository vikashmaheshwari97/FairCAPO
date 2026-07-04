from __future__ import annotations

import hashlib


def file_digest(path: str) -> str:
    state = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1048576)
            if not block:
                break
            state.update(block)
    return state.hexdigest()
