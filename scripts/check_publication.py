"""Check Git candidate files, including ZIP members, without printing matched data."""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path

RULES = {
    "local_user_path": re.compile(r"/Users/[A-Za-z0-9_.-]+/|/var/folders/[A-Za-z0-9_.-]+/"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_credential": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
}


def main():
    repository = Path(__file__).resolve().parents[1]
    rules = dict(RULES)
    identifiers_file = repository / "private/publication-identifiers.json"
    if identifiers_file.exists():
        identifiers = json.loads(identifiers_file.read_text())["identifiers"]
        if identifiers:
            rules["student_identifier"] = re.compile(
                r"(?<!\d)(?:"
                + "|".join(re.escape(str(value)) for value in identifiers)
                + r")(?!\d)"
            )
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=repository
        )
        .decode()
        .split("\0")
    )
    failures = []

    def inspect(label, data):
        content = data.decode("utf-8", errors="ignore")
        for rule, pattern in rules.items():
            if pattern.search(content):
                failures.append((label, rule))

    for name in sorted(set(filter(None, names))):
        path = repository / name
        if not path.is_file():
            continue
        if name.startswith(("data/", "cache/", "runs/", "private/", ".venv/")):
            failures.append((name, "private_or_generated_directory"))
        if path.suffix.lower() == ".pdf":
            failures.append((name, "unreviewed_pdf"))
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append((name, "large_file_over_10MiB"))
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    if not item.is_dir():
                        inspect(f"{name}:{item.filename}", archive.read(item))
        elif path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            inspect(name, path.read_bytes())
    if failures:
        for name, reason in failures:
            print(f"FAIL {reason}: {name}")
        raise SystemExit(1)
    print(f"PASS: {len(set(filter(None, names)))} candidate files; no matching publication issues.")
    print("This is a scoped content check, not a guarantee about every possible secret or right.")


if __name__ == "__main__":
    main()
