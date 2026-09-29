"""Explicitly download the pinned, checksummed actionlint development tool."""

import argparse
import hashlib
import io
import platform
import tarfile
import urllib.request
from pathlib import Path

VERSION = "1.7.7"
DIGESTS = {
    ("Darwin", "x86_64"): "28e5de5a05fc558474f638323d736d822fff183d2d492f0aecb2b73cc44584f5",
    ("Darwin", "arm64"): "2693315b9093aeacb4ebd91a993fea54fc215057bf0da2659056b4bc033873db",
    ("Linux", "x86_64"): "023070a287cd8cccd71515fedc843f1985bf96c436b7effaecce67290e7e0757",
    ("Linux", "aarch64"): "401942f9c24ed71e4fe71b76c7d638f66d8633575c4016efd2977ce7c28317d0",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    host = (platform.system(), platform.machine())
    if host not in DIGESTS:
        parser.error("unsupported host; no unverified tool will be downloaded")
    os_name = host[0].lower()
    arch = "amd64" if host[1] == "x86_64" else "arm64"
    archive = f"actionlint_{VERSION}_{os_name}_{arch}.tar.gz"
    url = f"https://github.com/rhysd/actionlint/releases/download/v{VERSION}/{archive}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(10 * 1024 * 1024 + 1)
    if hashlib.sha256(data).hexdigest() != DIGESTS[host]:
        raise SystemExit("actionlint archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        member = tar.getmember("actionlint")
        if not member.isfile():
            raise SystemExit("invalid actionlint archive")
        source = tar.extractfile(member)
        if source is None:
            raise SystemExit("missing actionlint binary")
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_bytes(source.read())
    args.dest.chmod(0o755)
    print(f"Installed actionlint {VERSION} with verified SHA-256")


if __name__ == "__main__":
    main()
