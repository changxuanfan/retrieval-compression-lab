"""Remove local repository paths from shareable snapshots, preserving raw runs.

Backs up modified public files under ignored private/publish-backup. This only
changes exported metadata/ZIP metadata and their checksums, never numeric scores.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from retrieval_lab.common import read_json, sha256, write_json


def clean(value, repository):
    if isinstance(value, str) and value.startswith(str(repository) + "/"):
        return str(Path(value).relative_to(repository))
    if isinstance(value, list):
        return [clean(item, repository) for item in value]
    if isinstance(value, dict):
        return {key: clean(item, repository) for key, item in value.items()}
    return value


def main():
    repository = Path(__file__).resolve().parents[1]
    backup = repository / "private/publish-backup"

    def preserve(path):
        target = backup / path.relative_to(repository)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    changes = []
    for path in sorted((repository / "docs").rglob("*.json")):
        before = read_json(path)
        after = clean(before, repository)
        if after != before:
            preserve(path)
            if path.name == "study.json":
                after["public_export"] = {
                    "local_paths_removed": True,
                    "original_metadata_sha256": sha256(path),
                    "numeric_results_changed": False,
                }
            write_json(path, after)
            changes.append(str(path.relative_to(repository)))
    for path in sorted((repository / "docs").rglob("original-source.zip")):
        replacements = {}
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.filename.endswith(".json"):
                    before = json.loads(archive.read(info.filename))
                    after = clean(before, repository)
                    if after != before:
                        replacements[info.filename] = (json.dumps(after, indent=2) + "\n").encode()
        if replacements:
            preserve(path)
            temporary = path.with_suffix(".zip.part")
            with zipfile.ZipFile(path) as source, zipfile.ZipFile(temporary, "w") as target:
                for info in source.infolist():
                    target.writestr(
                        info, replacements.get(info.filename, source.read(info.filename))
                    )
            temporary.replace(path)
            metadata = path.parent / "supplementary-provenance.json"
            preserve(metadata)
            value = read_json(metadata)
            value["source_archive_sha256"] = sha256(path)
            value["source_archive_metadata_local_paths_removed"] = True
            write_json(metadata, value)
            changes.append(str(path.relative_to(repository)))
    for path in sorted((repository / "docs/results").glob("*/snapshot.json")):
        value = read_json(path)
        updated = {name: sha256(path.parent / name) for name in value["files"]}
        if updated != value["files"]:
            preserve(path)
            value["files"] = updated
            value["public_export"] = {"local_paths_removed": True, "numeric_results_changed": False}
            write_json(path, value)
    print(f"Sanitized {len(changes)} exported files; raw runs preserved; backups under private/.")


if __name__ == "__main__":
    main()
