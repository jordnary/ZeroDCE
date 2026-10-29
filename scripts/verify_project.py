"""Audit public repository files; optionally verify private experiment outputs."""

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


LOCAL_ROOTS = {
    "assignment", "reports", "data", "results", "private", "scratch", "dist", "build",
    ".cache", ".venv", "venv", "env", ".conda", ".idea", ".vscode",
}
RUNTIME_PARTS = {
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov",
}
DISALLOWED_NAMES = (
    ".env", ".env.*", "*.key", "*.pem", "*.p12", "*.pfx", "*.crt", "*.cer",
    "*.log", "*.db", "*.sqlite*", "*.pyc", "*.pyo", "*.zip", "*.7z",
    "*.tar", "*.gz", "*.bundle", "*.pt", "*.ckpt", ".coverage",
)
PUBLIC_WEIGHTS = {Path("weights/Epoch99.pth"), Path("weights/manifest.json")}


class AuditError(ValueError):
    """An audit error whose message is safe to display."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def git_output(root, *arguments, input_text=None):
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, input=input_text, capture_output=True,
            encoding="utf-8", check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        raise AuditError("Unable to inspect Git state; run inside a Git checkout") from None
    return result.stdout


def selected_files(root):
    output = git_output(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted({Path(name) for name in output.split("\0") if name})


def audit_ignore_rules(root):
    probes = [f"{directory}/privacy_probe.txt" for directory in sorted(LOCAL_ROOTS)]
    probes += [".env", "nested/.env.local", "nested/credentials.pem",
               "nested/cache.sqlite3", "nested/runtime.log", "weights/custom.pth"]
    ignored = git_output(root, "check-ignore", "--no-index", "-z", "--stdin",
                         input_text="\0".join(probes) + "\0").split("\0")
    require(set(probes) <= set(ignored), "Required local-only ignore rules are missing")


def audit_files(root, files):
    root = Path(root).resolve()
    selected = {root / path for path in files}
    path_pattern = re.compile(r"(?i)(?:(?<![a-z0-9])[a-z]:[\\/]|/(?:Users|home)/)")
    secret_pattern = re.compile(
        r"(?:ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----)"
    )
    private_markers = [
        value for key in ("USERNAME", "COMPUTERNAME", "CONDA_DEFAULT_ENV")
        if (value := os.environ.get(key, "")) and len(value) > 3 and value != "base"
    ]
    text_suffixes = {".md", ".py", ".json", ".csv", ".txt", ".toml", ".yml", ".yaml"}
    for relative in files:
        require(
            not relative.is_absolute() and ".." not in relative.parts
            and all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in relative.parts),
            "Published paths must use English names and stay inside the repository",
        )
        require(
            relative.parts[0].lower() not in LOCAL_ROOTS
            and not RUNTIME_PARTS.intersection(part.lower() for part in relative.parts),
            "Local-only material selected for publication",
        )
        require(
            not any(fnmatch.fnmatch(relative.name.lower(), pattern) for pattern in DISALLOWED_NAMES),
            "Credential, local database, archive, or runtime file selected",
        )
        require(
            (relative.parts[0].lower() != "weights" or relative in PUBLIC_WEIGHTS)
            and (relative.suffix.lower() != ".pth" or relative == Path("weights/Epoch99.pth")),
            "Only the official checkpoint and manifest may be published",
        )
        path = root / relative
        require(not path.is_symlink() and path.resolve().is_relative_to(root),
                "Published file points outside the repository")
        require(path.is_file(), "Selected file is missing; stage intended removals")
        require(path.stat().st_size < 10 * 1024 * 1024, "A published file exceeds the size limit")
        if relative.suffix.lower() == ".png":
            from PIL import Image
            with Image.open(path) as image:
                require(image.mode == "RGB" and not image.info and not image.getexif(),
                        "Published image has metadata or an unsupported mode")
        if relative.suffix.lower() in text_suffixes or relative.name in {".gitignore", ".gitattributes"}:
            content = path.read_text(encoding="utf-8")
            require(not path_pattern.search(content), "Absolute local path detected")
            require(not secret_pattern.search(content), "Credential-like content detected")
            require(
                not any(re.search(r"(?i)(?<!\w)" + re.escape(value) + r"(?!\w)", content)
                        for value in private_markers),
                "Local identifier detected",
            )
            if relative.suffix.lower() == ".md":
                for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
                    if not link.startswith(("https://", "http://", "#")):
                        target = (path.parent / link.split("#")[0]).resolve()
                        require(target in selected and target.is_file(),
                                "Document link targets a missing or local-only file")
    return len(files)


def audit_upstream(root):
    manifest = json.loads((root / "weights/manifest.json").read_text(encoding="utf-8"))
    checkpoint = (root / "weights/Epoch99.pth").read_bytes()
    require(hashlib.sha256(checkpoint).hexdigest() == manifest["sha256"],
            "Checkpoint checksum mismatch")
    code = (root / "third_party/zero_dce/model.py").read_bytes()
    git_blob = hashlib.sha1(f"blob {len(code)}\0".encode() + code).hexdigest()
    require(git_blob == "3b710a5a33b38ee742c200c69b060f05246c7d1c", "Upstream source differs")
    return manifest


def audit_local_results(root, input_directory, results_directory, manifest):
    sys.path.insert(0, str(root))
    import zerodce  # Configure threading before importing torch.
    from zerodce.io import read_image
    from zerodce.metrics import image_statistics

    metrics = json.loads((results_directory / "metrics.json").read_text(encoding="utf-8"))
    require(metrics["parameter_count"] == 79416, "Incorrect parameter count")
    require(metrics["weight_sha256"] == manifest["sha256"], "Measurement checkpoint differs")
    require(bool(metrics["records"]), "No local experiment records found")
    for record in metrics["records"]:
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", record["image"])),
                "Local image names must use English letters, digits, underscores, or hyphens")
        versions = [
            ("input", input_directory, "input_sha256"),
            ("enhanced", results_directory / "enhanced", "output_sha256"),
        ]
        if "final" in record:
            versions.append(("final", results_directory / "final", "final_sha256"))
        for version, directory, checksum in versions:
            path = directory / f"{record['image']}.png"
            require(hashlib.sha256(path.read_bytes()).hexdigest() == record[checksum],
                    "Image checksum mismatch")
            measured = image_statistics(read_image(path))
            require(all(abs(measured[key] - record[version][key]) < 1e-7 for key in measured),
                    "Measurements do not match saved pixels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-results", action="store_true",
                        help="Also verify ignored local image outputs and saved metrics")
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit_ignore_rules(root)
    count = audit_files(root, selected_files(root))
    manifest = audit_upstream(root)
    print("PASS: official source and checkpoint checksums")
    print(f"PASS: {count} public files checked for privacy, English names, size, and links")
    print("PASS: coursework, personal data, and generated outputs are excluded")
    if args.local_results:
        audit_local_results(root, root / args.input, root / args.results, manifest)
        print("PASS: local image checksums and saved-pixel statistics")


if __name__ == "__main__":
    try:
        main()
    except AuditError as error:
        print(f"Audit failed: {error}")
        raise SystemExit(1)
    except (OSError, ValueError, KeyError, ImportError):
        print("Audit failed: unable to read required files or dependencies")
        raise SystemExit(1)
