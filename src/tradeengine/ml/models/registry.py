from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - fallback when optional dependency is absent
    joblib = None  # type: ignore[assignment]


class ModelRegistryError(ValueError):
    """Raised when persisted model artifacts are invalid."""


class ModelRegistry:
    """Persist model artifacts and metadata alongside one another."""

    @staticmethod
    def _dump_artifact(bundle: dict[str, Any], target: Path) -> None:
        if joblib is not None:
            joblib.dump(bundle, target)
            return
        target.write_bytes(pickle.dumps(bundle))

    @staticmethod
    def _load_artifact(model_path: str) -> Any:
        if joblib is not None:
            return joblib.load(model_path)
        return pickle.loads(Path(model_path).read_bytes())

    @staticmethod
    def metadata_path_for(model_path: str) -> Path:
        path = Path(model_path)
        return path.with_name(f"{path.stem}_metadata.json")

    def save(
        self,
        bundle: dict[str, Any],
        model_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if "model" not in bundle:
            raise ModelRegistryError("Model bundle must include a 'model' key")

        target = Path(model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._dump_artifact(bundle, target)

        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("training_date", datetime.now(UTC).date().isoformat())
        metadata_payload.setdefault("features", bundle.get("feature_columns", []))
        metadata_payload.setdefault("class_labels", bundle.get("class_labels", []))

        meta_path = self.metadata_path_for(model_path)
        meta_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        return metadata_payload

    @staticmethod
    def load_bundle(model_path: str) -> dict[str, Any]:
        loaded = ModelRegistry._load_artifact(model_path)
        if not isinstance(loaded, dict):
            raise ModelRegistryError("Model artifact must be a dictionary bundle")
        if "model" not in loaded:
            raise ModelRegistryError("Model artifact is missing the 'model' key")
        return loaded
