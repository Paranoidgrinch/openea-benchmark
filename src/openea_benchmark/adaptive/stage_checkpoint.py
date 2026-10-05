\
"""Trusted local checkpointing for validation workflow stages.

This uses Python pickle only for OpenEA-generated files inside an explicit run
folder.  It must never load arbitrary/untrusted pickle files.  A workflow
signature is checked before any stage object is reused.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any


class StageCheckpointStore:
    def __init__(self, root: Path, *, signature: str):
        if not signature.strip():
            raise ValueError("stage checkpoint signature must be non-empty")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self._manifest_path = self.root / "manifest.json"
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        digest = hashlib.sha256(self.signature.encode("utf-8")).hexdigest()
        if self._manifest_path.exists():
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if payload.get("signature_sha256") != digest:
                raise ValueError(
                    "stage checkpoint signature does not match this workflow configuration"
                )
            return
        payload = {
            "format": "OPENEA_VALIDATION_STAGE_PICKLE_V1",
            "signature_sha256": digest,
            "trusted_local_only": True,
        }
        self._atomic_text(self._manifest_path, json.dumps(payload, indent=2, sort_keys=True)+"\n")

    @staticmethod
    def _safe_key(key: str) -> str:
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in key):
            raise ValueError("stage checkpoint key contains unsupported characters")
        return key

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def path_for(self, key: str) -> Path:
        return self.root / f"{self._safe_key(key)}.pkl"

    def has(self, key: str) -> bool:
        return self.path_for(key).exists()

    def save(self, key: str, value: Any) -> Path:
        path = self.path_for(key)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return path

    def load(self, key: str) -> Any:
        path = self.path_for(key)
        if not path.exists():
            raise KeyError(key)
        # TRUST BOUNDARY: only files written by this OpenEA run directory.
        with path.open("rb") as handle:
            return pickle.load(handle)

    def remove(self, key: str) -> bool:
        """Remove one reusable stage checkpoint if it exists."""
        path = self.path_for(key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def load_if_valid(
        self,
        key: str,
        *,
        validator,
        invalidate_invalid: bool = True,
    ) -> Any | None:
        """Load a checkpoint only when it still satisfies a reuse contract.

        Failed or obsolete scientific results must not become sticky merely
        because they were serialized.  If validation fails, the checkpoint is
        removed by default so the current solver/policy gets a fresh attempt.
        """
        if not self.has(key):
            return None
        value = self.load(key)
        if bool(validator(value)):
            return value
        if invalidate_invalid:
            self.remove(key)
        return None
