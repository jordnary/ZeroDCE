"""Download the official checkpoint from a pinned upstream revision."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd"
URL = (f"https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/{REVISION}/"
       "Zero-DCE_code/snapshots/Epoch99.pth")


def main():
    directory = Path("weights")
    directory.mkdir(exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
        if digest != expected:
            raise ValueError("Checkpoint checksum mismatch; no file was replaced.")
    (directory / "Epoch99.pth").write_bytes(content)
    manifest_path.write_text(json.dumps({
        "filename": "Epoch99.pth", "upstream_revision": REVISION,
        "source": URL, "sha256": digest, "bytes": len(content),
        "license": "CC BY-NC 4.0; academic research only (upstream terms)",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Verified checkpoint: {len(content)} bytes; SHA256 {digest}")


if __name__ == "__main__":
    main()
