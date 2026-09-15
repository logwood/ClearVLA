"""Checkpoint identity and explicit compatibility for the clean mainline."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, cast

from .config import ExperimentConfig
from .manifest import (
    architecture_manifest_for_bspine_implementation,
    manifest_from_mapping,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_source_text(path: Path) -> str:
    """Hash source text with checkout-independent newline semantics."""

    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _is_hex_digest(value: str, *, length: int) -> bool:
    if len(value) != int(length):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _local_module_path(root: Path, module: str) -> Path | None:
    if module != "clearvla" and not module.startswith("clearvla."):
        return None
    base = root.joinpath(*module.split("."))
    source = base.with_suffix(".py")
    package = base / "__init__.py"
    if source.is_file():
        return source
    if package.is_file():
        return package
    return None


def _local_imports(root: Path, source: Path) -> set[Path]:
    """Resolve static ClearVLA imports from one active Python source."""

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    current_module = _module_name(root, source)
    package = current_module if source.name == "__init__.py" else current_module.rpartition(".")[0]
    imports: set[Path] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                keep = len(parts) - (int(node.level) - 1)
                if keep < 0:
                    continue
                base = ".".join(parts[:keep])
                module = ".".join(value for value in (base, node.module or "") if value)
            else:
                module = node.module or ""
            if module:
                candidates.append(module)
                candidates.extend(f"{module}.{alias.name}" for alias in node.names)
        for module in candidates:
            resolved = _local_module_path(root, module)
            if resolved is not None:
                imports.add(resolved)
    return imports


def _active_python_closure(root: Path, seeds: list[Path]) -> set[Path]:
    """Return local Python files recursively reachable from explicit seeds.

    The independent mainline intentionally keeps a mechanically extracted
    V120 oracle beside its adapters.  Only a small, audited subset of that
    oracle is executable.  Seeding the closure with every oracle file made an
    edit to an unimported historical CVAE or trunk reject an otherwise exact
    resume, despite having no route into the current graph.
    """

    pending = list(seeds)
    closure: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in closure or "__pycache__" in source.parts:
            continue
        closure.add(source)
        for dependency in _local_imports(root, source):
            if dependency not in closure:
                pending.append(dependency)
            # Importing a submodule executes each package initializer.  Those
            # files are therefore executable identity even when empty today.
            parent = dependency.parent
            while parent != root and root in parent.parents:
                initializer = parent / "__init__.py"
                if initializer.is_file() and initializer not in closure:
                    pending.append(initializer)
                parent = parent.parent
    return closure


@dataclass(frozen=True)
class ArtifactIdentity:
    """Identity of one required external artifact such as the T5 condition."""

    logical_name: str
    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_file(cls, logical_name: str, path: str | Path) -> "ArtifactIdentity":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"required artifact does not exist: {source}")
        return cls(
            logical_name=str(logical_name),
            path=str(source),
            size_bytes=int(source.stat().st_size),
            sha256=_sha256_file(source),
        )

    def validate(self) -> None:
        if (
            not self.logical_name
            or self.size_bytes < 0
            or not _is_hex_digest(self.sha256, length=64)
        ):
            raise ValueError("external artifact identity is invalid")


@dataclass(frozen=True)
class DatasetIdentity:
    """Dataset/cache identity without hashing image or tensor caches into Git."""

    raw_root: str
    hdf5_glob: str
    inventory_sha256: str
    state_normalizer_sha256: str
    action_normalizer_sha256: str
    decoded_cache_identity: str
    dino_cache_identity: str

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"dataset identity {name} must be non-empty")
        for name in (
            "inventory_sha256",
            "state_normalizer_sha256",
            "action_normalizer_sha256",
            "decoded_cache_identity",
            "dino_cache_identity",
        ):
            if not _is_hex_digest(str(values[name]), length=64):
                raise ValueError(f"dataset identity {name} must be SHA-256")


@dataclass(frozen=True)
class SourceSnapshot:
    """Canonical-text hash of only the active mainline source closure."""

    files: tuple[tuple[str, str], ...]
    digest: str

    def validate(self) -> None:
        if not self.files or not _is_hex_digest(self.digest, length=64):
            raise ValueError("active source snapshot is invalid")
        paths = [path for path, _ in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("active source files must be sorted and unique")
        if any(not _is_hex_digest(digest, length=64) for _, digest in self.files):
            raise ValueError("active source file hash is invalid")
        if hashlib.sha256(_canonical(self.files)).hexdigest() != self.digest:
            raise ValueError("active source snapshot digest is inconsistent")
        forbidden = (
            "policy_runtime_v39.py",
            "train_v40_policy.py",
            "current_v",
            "scripts/",
        )
        if any(any(token in path.replace("\\", "/") for token in forbidden) for path in paths):
            raise ValueError("legacy/version launcher leaked into active source identity")


def active_source_snapshot(repo_root: str | Path) -> SourceSnapshot:
    """Hash the complete imported source closure and active JSON spec.

    The final bottom/observation implementations are being extracted into the
    same package, so no historical V39 source file is admitted here.  Adding a
    new active module under ``clearvla/mainline`` automatically changes the
    closure; editing an archived experiment script does not.
    """

    root = Path(repo_root).resolve()
    package = root / "clearvla" / "mainline"
    if not package.is_dir():
        raise FileNotFoundError(f"mainline package does not exist under {root}")
    # Source identity follows the executable entry point.  This is stricter
    # than hashing a hand-maintained file list and more accurate than hashing
    # every prototype that happens to remain under ``mainline``: only modules
    # reachable from the formal trainer (including imported V120 extraction
    # modules and package initializers) can alter a run's graph identity.
    # Archived/inactive alternatives must not make an exact resume fail.
    seeds = [package / "train.py"]
    if not seeds[0].is_file():
        raise FileNotFoundError("mainline training entry point is missing")
    sources = list(_active_python_closure(root, seeds))
    preset = root / "configs" / "mainline" / "object_intent_dynamics_323.json"
    if preset.is_file():
        sources.append(preset)
    rows = tuple(
        (path.relative_to(root).as_posix(), _sha256_source_text(path))
        for path in sorted(set(sources))
    )
    digest = hashlib.sha256(_canonical(rows)).hexdigest()
    snapshot = SourceSnapshot(files=rows, digest=digest)
    snapshot.validate()
    return snapshot


def git_commit(repo_root: str | Path) -> str:
    """Return the immutable Git commit without making dirty state architecture."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise ValueError("Git did not return one full commit hash")
    return value


@dataclass(frozen=True)
class CheckpointIdentity:
    """Serializable identity of one trainable graph and its factual inputs."""

    manifest: dict[str, object]
    manifest_digest: str
    config_digest: str
    source: SourceSnapshot
    git_commit: str
    dataset: DatasetIdentity
    language: ArtifactIdentity

    def validate(self, *, require_current_manifest: bool = True) -> None:
        manifest = manifest_from_mapping(
            self.manifest,
            require_current_schema=require_current_manifest,
        )
        if manifest.digest() != self.manifest_digest:
            raise ValueError("checkpoint manifest digest is inconsistent")
        if not _is_hex_digest(self.config_digest, length=64):
            raise ValueError("checkpoint config digest is invalid")
        if not _is_hex_digest(self.git_commit, length=40):
            raise ValueError("checkpoint Git commit is invalid")
        self.source.validate()
        self.dataset.validate()
        self.language.validate()

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest,
            "manifest_digest": self.manifest_digest,
            "config_digest": self.config_digest,
            "source": {
                "files": [list(row) for row in self.source.files],
                "digest": self.source.digest,
            },
            "git_commit": self.git_commit,
            "dataset": asdict(self.dataset),
            "language": asdict(self.language),
        }


def build_checkpoint_identity(
    config: ExperimentConfig,
    *,
    repo_root: str | Path,
    dataset: DatasetIdentity,
    language: ArtifactIdentity,
    commit: str | None = None,
) -> CheckpointIdentity:
    config.validate()
    manifest = architecture_manifest_for_bspine_implementation(
        config.bottom.bspine_implementation
    )
    manifest.validate()
    identity = CheckpointIdentity(
        manifest=manifest.as_dict(),
        manifest_digest=manifest.digest(),
        config_digest=config.digest(include_paths=False),
        source=active_source_snapshot(repo_root),
        git_commit=git_commit(repo_root) if commit is None else str(commit),
        dataset=dataset,
        language=language,
    )
    identity.validate()
    return identity


def checkpoint_identity_from_mapping(
    value: Mapping[str, object],
    *,
    require_current_manifest: bool = True,
) -> CheckpointIdentity:
    source_value = value.get("source")
    dataset_value = value.get("dataset")
    language_value = value.get("language")
    manifest_value = value.get("manifest")
    if not isinstance(source_value, Mapping):
        raise ValueError("checkpoint source identity must be a mapping")
    if not isinstance(dataset_value, Mapping) or not isinstance(language_value, Mapping):
        raise ValueError("checkpoint external identities must be mappings")
    if not isinstance(manifest_value, Mapping):
        raise ValueError("checkpoint manifest must be a mapping")
    file_rows = source_value.get("files")
    if not isinstance(file_rows, (tuple, list)):
        raise ValueError("checkpoint source files must be a sequence")
    rows: list[tuple[str, str]] = []
    for row in file_rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("checkpoint source row must be a path/hash pair")
        rows.append((str(row[0]), str(row[1])))
    identity = CheckpointIdentity(
        manifest=dict(cast(Mapping[str, object], manifest_value)),
        manifest_digest=str(value.get("manifest_digest", "")),
        config_digest=str(value.get("config_digest", "")),
        source=SourceSnapshot(
            files=tuple(rows),
            digest=str(source_value.get("digest", "")),
        ),
        git_commit=str(value.get("git_commit", "")),
        dataset=DatasetIdentity(**dict(dataset_value)),
        language=ArtifactIdentity(**dict(language_value)),
    )
    identity.validate(require_current_manifest=require_current_manifest)
    return identity


@dataclass(frozen=True)
class CompatibilityReport:
    exact_resume: bool
    reusable_components: tuple[str, ...]
    rejected_components: tuple[str, ...]
    reasons: tuple[str, ...]


def compare_checkpoint_identity(
    saved: CheckpointIdentity,
    current: CheckpointIdentity,
) -> CompatibilityReport:
    """Classify exact resume separately from explicit bottom-only migration."""

    saved.validate(require_current_manifest=False)
    current.validate()
    reasons: list[str] = []
    saved_dataset_semantic = {
        key: value for key, value in asdict(saved.dataset).items() if key != "raw_root"
    }
    current_dataset_semantic = {
        key: value for key, value in asdict(current.dataset).items() if key != "raw_root"
    }
    saved_language_semantic = {
        "logical_name": saved.language.logical_name,
        "size_bytes": saved.language.size_bytes,
        "sha256": saved.language.sha256,
    }
    current_language_semantic = {
        "logical_name": current.language.logical_name,
        "size_bytes": current.language.size_bytes,
        "sha256": current.language.sha256,
    }
    for name, left, right in (
        ("manifest", saved.manifest_digest, current.manifest_digest),
        ("config", saved.config_digest, current.config_digest),
        ("source", saved.source.digest, current.source.digest),
        ("dataset", saved_dataset_semantic, current_dataset_semantic),
        ("language", saved_language_semantic, current_language_semantic),
    ):
        if left != right:
            reasons.append(f"{name} identity differs")
    exact = not reasons
    if exact:
        return CompatibilityReport(
            exact_resume=True,
            reusable_components=("observation", "top", "bottom", "training", "runtime"),
            rejected_components=(),
            reasons=(),
        )
    saved_manifest = manifest_from_mapping(
        saved.manifest,
        require_current_schema=False,
    )
    current_manifest = manifest_from_mapping(current.manifest)
    bottom_same = saved_manifest.components.bottom == current_manifest.components.bottom
    return CompatibilityReport(
        exact_resume=False,
        reusable_components=("bottom",) if bottom_same else (),
        rejected_components=("observation", "top", "training", "runtime")
        + (() if bottom_same else ("bottom",)),
        reasons=tuple(reasons),
    )


__all__ = [
    "ArtifactIdentity",
    "CheckpointIdentity",
    "CompatibilityReport",
    "DatasetIdentity",
    "SourceSnapshot",
    "active_source_snapshot",
    "build_checkpoint_identity",
    "checkpoint_identity_from_mapping",
    "compare_checkpoint_identity",
]
