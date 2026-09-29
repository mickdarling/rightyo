"""Small, offline checks for public repository hygiene, not privacy certification."""

import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

PRIVATE_PARTS = {
    "recordings",
    "transcripts",
    "datasets",
    "artifacts",
    "runs",
    "logs",
    "models",
    "checkpoints",
    "features",
    "feature_cache",
    "consent",
    "secrets",
    "private",
    "local",
    "wandb",
    "tensorboard",
}
PRIVATE_SUFFIXES = {
    ".wav",
    ".aiff",
    ".aif",
    ".aifc",
    ".caf",
    ".pcm",
    ".raw",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".mp4",
    ".webm",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".mlmodel",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".pkl",
    ".pickle",
    ".pem",
    ".key",
    ".p8",
    ".p12",
    ".mobileprovision",
    ".log",
}
MAX_FILE_BYTES = 1024 * 1024
IGNORE_CASES = {
    "src/rightyo.egg-info/PKG-INFO": True,
    "recordings/sample.unfamiliar": True,
    "nested/transcripts/example.json": True,
    "datasets/sample.unfamiliar": True,
    "features/sample.unfamiliar": True,
    "feature_cache/sample.unfamiliar": True,
    "consent/sample.json": True,
    "checkpoints/sample.unfamiliar": True,
    "models/sample.unfamiliar": True,
    "artifacts/sample.unfamiliar": True,
    "runs/sample.unfamiliar": True,
    "wandb/sample.unfamiliar": True,
    "nested/example.npz": True,
    "nested/example.opus": True,
    ".env": True,
    ".env.local": True,
    "secrets/example.key": True,
    ".env.example": False,
    "src/rightyo/model.py": False,
    "tests/test_controller.py": False,
    "examples/synthetic-turns.json": False,
    "docs/model-card.md": False,
    "LICENSE": False,
}


def git_files(root):
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    return [Path(p.decode()) for p in result.stdout.split(b"\0") if p]


def artifact_reason(path, data):
    """Return a category only; never expose matched content or participant paths."""
    name = PurePosixPath(path)
    parts = {part.lower() for part in name.parts}
    if parts & PRIVATE_PARTS or any(p.endswith((".mlmodelc", ".mlpackage")) for p in parts):
        return "private/artifact directory"
    if name.suffix.lower() in PRIVATE_SUFFIXES:
        return "private/artifact extension"
    if name.name == ".env" or (name.name.startswith(".env.") and name.name != ".env.example"):
        return "local environment"
    if len(data) > MAX_FILE_BYTES:
        return "oversized file"
    if b"\0" in data:
        return "binary file"
    if name.suffix.lower() == ".ipynb":
        try:
            notebook = json.loads(data)
            if any(c.get("outputs") or c.get("attachments") for c in notebook.get("cells", [])):
                return "saved notebook output"
        except (ValueError, AttributeError, TypeError):
            return "invalid notebook"
    return None


def artifact_errors(root, paths):
    errors = []
    for index, path in enumerate(paths, 1):
        full = root / path
        if full.is_symlink():
            errors.append(f"tracked entry {index}: symlinks require explicit review")
            continue
        try:
            with full.open("rb") as file:
                data = file.read(MAX_FILE_BYTES + 1)
        except OSError:
            errors.append(f"tracked entry {index}: missing/unreadable file")
            continue
        reason = artifact_reason(path.as_posix(), data)
        if reason:
            errors.append(f"tracked entry {index}: {reason}")
    return errors


def ignore_errors(root):
    paths = list(IGNORE_CASES)
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=root,
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        return ["git check-ignore failed"]
    ignored = set(result.stdout.splitlines())
    return [
        f"ignore fixture {i}: expected ignored={expected}"
        for i, (path, expected) in enumerate(IGNORE_CASES.items(), 1)
        if (path in ignored) != expected
    ]


def without_code(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.sub(r"(?ms)^(`{3,}|~{3,}).*?^\1[^\n]*$", "", text)


def markdown_errors(root, path):
    text = (root / path).read_text()
    errors = []
    if not text.endswith("\n") or text.endswith("\n\n"):
        errors.append(f"{path}: require one final newline")
    for number, line in enumerate(text.splitlines(), 1):
        if line.rstrip() != line:
            errors.append(f"{path}:{number}: trailing whitespace")
        if "\t" in line:
            errors.append(f"{path}:{number}: tabs in Markdown")
    fences = re.findall(r"(?m)^\s*(`{3,}|~{3,})", text)
    if len(fences) % 2:
        errors.append(f"{path}: unclosed code fence")
    # Inline links plus reference definitions. External URLs are checked separately.
    content = without_code(text)
    targets = re.findall(r"\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)", content)
    targets += re.findall(r"(?m)^\s*\[[^\]]+\]:\s*(\S+)", content)
    for target in targets:
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        local = (root / path.parent / unquote(parsed.path)).resolve()
        if not local.is_relative_to(root.resolve()) or not local.exists():
            errors.append(f"{path}: missing/escaping local link")
    return errors


def issue_references(event, repository):
    """Parse only explicit linkage lines from event data; never evaluate its text."""
    pr = event.get("pull_request", {})
    body = without_code(pr.get("body") or "")
    own_url = rf"https://github\.com/{re.escape(repository)}/issues/([1-9]\d*)"
    issue_pattern = re.compile(rf"(?:#([1-9]\d*)\b|{own_url}\b)")
    numbers = set()
    for line in body.splitlines():
        if re.match(r"(?i)^\s*(?:refs?|fix(?:es)?|close[sd]?|resolve[sd]?)\s+", line):
            for match in issue_pattern.finditer(line):
                numbers.add(int(match.group(1) or match.group(2)))
    return sorted(numbers)


def workflow_errors(path, document):
    """A deliberately narrow secretless baseline policy; actionlint validates syntax."""
    errors = []
    if not isinstance(document, dict):
        return [f"{path}: workflow must be a mapping"]
    if "secrets." in str(document):
        errors.append(f"{path}: baseline workflow must not reference secrets")
    event = document.get("on", document.get(True, {}))  # YAML 1.1 parsers treat on as True.
    if isinstance(event, str):
        event = [event]
    if "pull_request_target" in event or "workflow_run" in event:
        errors.append(f"{path}: privileged events require a separate reviewed policy")
    if document.get("permissions") != {"contents": "read"}:
        errors.append(f"{path}: baseline permissions must be contents: read")
    for job in document.get("jobs", {}).values():
        if job.get("runs-on") != "ubuntu-24.04":
            errors.append(f"{path}: baseline uses ephemeral ubuntu-24.04 only")
        timeout = job.get("timeout-minutes")
        if not isinstance(timeout, int) or not 1 <= timeout <= 15:
            errors.append(f"{path}: job timeout must be 1-15 minutes")
        if "permissions" in job and job["permissions"] not in ({}, {"contents": "read"}):
            errors.append(f"{path}: baseline job permissions exceed policy")
        for step in job.get("steps", []):
            action = step.get("uses", "")
            if action and not re.fullmatch(r"[\w.-]+/[\w./-]+@[a-f0-9]{40}", action):
                errors.append(f"{path}: external actions require full commit SHAs")
            if "${{" in step.get("run", ""):
                errors.append(f"{path}: pass expressions as data via env, not shell interpolation")
    return errors
