"""Small, offline checks for public repository hygiene, not privacy certification."""

import copy
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import yaml

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

# Voice-data guard (#109, #137): real voice audio, speaker embeddings and voiceprints
# must never be tracked. Files with these extensions, and the path segments below, are
# refused unless the path sits under SYNTHETIC_VOICE_FIXTURES. The guard runs before
# the generic artifact rules so a failure names the privacy requirement.
VOICE_DATA_SUFFIXES = {
    ".wav",
    ".pcm",
    ".raw",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".caf",
    ".aif",
    ".aiff",
    ".aifc",
    ".webm",
    ".wave",
    ".oga",
    ".amr",
    ".awb",
    ".3gp",
    ".3g2",
    ".wma",
    ".m4b",
    ".m4p",
    ".spx",
    ".mka",
    ".ac3",
    ".au",
    ".snd",
    ".npy",
    ".npz",
    ".emb",
    ".pt",
    ".pth",
    ".onnx",
    ".mlmodel",
    ".mlmodelc",
    ".mlpackage",
    ".safetensors",
}
VOICE_DATA_PARTS = {"enrollment", "enrollments", "voiceprint", "voiceprints"}
# Path prefix -> why everything under it is synthetic. Inventory of tracked files on
# 2026-10-09: none has a voice-data extension or segment, so the list is empty. An
# entry needs a reviewed PR recording the synthetic source (an authored signal or a
# TTS voice, never a recording of a real person). An entry lifts only the voice-data
# and extension rules; the binary and size rules still apply.
SYNTHETIC_VOICE_FIXTURES = {}
VOICE_DATA_REASON = (
    "voice audio/embedding/voiceprint outside the synthetic fixture allow-list "
    "(privacy requirement #109; see CONTRIBUTING.md)"
)
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
    "nested/example.emb": True,
    "enrollment/sample.unfamiliar": True,
    "nested/voiceprints/sample.json": True,
    "examples/enrolled-override.jsonl": False,
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


def synthetic_voice_fixture(path, fixtures=None):
    """True only for a file strictly under an allow-listed synthetic fixture prefix."""
    fixtures = SYNTHETIC_VOICE_FIXTURES if fixtures is None else fixtures
    parts = PurePosixPath(path).parts
    return any(
        len(parts) > len(prefix) and parts[: len(prefix)] == prefix
        for prefix in (PurePosixPath(entry).parts for entry in fixtures)
    )


def voice_data_reason(path, fixtures=None):
    """Refuse voice audio, embeddings and enrollment paths outside synthetic fixtures."""
    parts = [part.lower() for part in PurePosixPath(path).parts]
    voice = any(PurePosixPath(part).suffix in VOICE_DATA_SUFFIXES for part in parts) or (
        not VOICE_DATA_PARTS.isdisjoint(parts)
    )
    if voice and not synthetic_voice_fixture(path, fixtures):
        return VOICE_DATA_REASON
    return None


def artifact_reason(path, data, fixtures=None):
    """Return a category only; never expose matched content or participant paths."""
    voice = voice_data_reason(path, fixtures)
    if voice:
        return voice
    name = PurePosixPath(path)
    parts = {part.lower() for part in name.parts}
    synthetic = synthetic_voice_fixture(path, fixtures)
    if parts & PRIVATE_PARTS or (
        not synthetic and any(p.endswith((".mlmodelc", ".mlpackage")) for p in parts)
    ):
        return "private/artifact directory"
    suffix = name.suffix.lower()
    if suffix in PRIVATE_SUFFIXES and not (synthetic and suffix in VOICE_DATA_SUFFIXES):
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
    issue_pattern = re.compile(rf"(?:(?<![\w/.-])#([1-9]\d*)\b|{own_url}\b)")
    numbers = set()
    for line in body.splitlines():
        if re.match(r"(?i)^\s*(?:refs?|fix(?:es)?|close[sd]?|resolve[sd]?)\s+", line):
            for match in issue_pattern.finditer(line):
                numbers.add(int(match.group(1) or match.group(2)))
    return sorted(numbers)


def mentions_secrets(value):
    """Inspect strings without escaping away context-token word boundaries."""
    if isinstance(value, str):
        return re.search(r"\bsecrets\b", value, re.I) is not None
    if isinstance(value, dict):
        return any(mentions_secrets(k) or mentions_secrets(v) for k, v in value.items())
    if isinstance(value, list):
        return any(mentions_secrets(item) for item in value)
    return False


AI_REVIEW_WORKFLOW = ".github/workflows/ai-review.yml"
REVIEW_RELAY_WORKFLOW = ".github/workflows/review-activity.yml"
CHECKOUT_ACTION = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"


REVIEW_INVALIDATION_COMMAND = r"""python3 - <<'PY'
import json
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path

repository = "mickdarling/rightyo"
if os.environ.get("GITHUB_REPOSITORY") != repository:
    raise SystemExit("Review invalidation repository mismatch")
event = os.environ.get("GITHUB_EVENT_NAME")
if event not in {"workflow_run", "issue_comment"}:
    raise SystemExit(0)
payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())

def command(body):
    return isinstance(body, str) and re.match(
        r"\A\s*@codex (?:security )?review(?:\s|\Z)", body, re.I) is not None

def stamp(value):
    if not isinstance(value, str) or not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z", value):
        raise RuntimeError("Invalid request timestamp")
    return datetime.fromisoformat(value).replace(microsecond=0)

if event == "issue_comment":
    comment = payload.get("comment") or {}
    action = payload.get("action")
    old_body = payload.get("changes", {}).get("body", {}).get("from")
    current = command(comment.get("body"))
    removed = (command(old_body) and not current) or (action == "deleted" and current)
    issue = payload.get("issue") or {}
    if (action not in {"created", "edited", "deleted"} or not (current or removed)
            or not issue.get("pull_request")):
        raise SystemExit(0)
    number, comment_id = issue.get("number"), comment.get("id")
    if type(number) is not int or number < 1 or type(comment_id) is not int or comment_id < 1:
        raise SystemExit("Review request event identity is invalid")
    try:
        cutoff = stamp(comment.get("updated_at"))
    except Exception:
        raise SystemExit("Review request timestamp is invalid") from None
else:
    run = payload["workflow_run"]
    if run.get("name") != "Native review activity relay":
        raise SystemExit(0)
    paths = [value for value in (run.get("path"), (payload.get("workflow") or {}).get("path"))
             if value is not None]
    if (run.get("repository", {}).get("full_name") != repository
            or not paths or any(path != ".github/workflows/review-activity.yml" for path in paths)
            or run.get("event") not in {"pull_request_review", "pull_request_review_comment"}
            or run.get("status") != "completed"):
        raise SystemExit("Review invalidation source hints are invalid")
    prs = run.get("pull_requests")
    if not isinstance(prs, list) or len(prs) > 20:
        raise SystemExit("Review invalidation association hints are invalid")
    heads = [run.get("head_sha")]
    for pr in prs:
        if not isinstance(pr, dict) or not isinstance(pr.get("head"), dict):
            raise SystemExit("Review invalidation association hints are invalid")
        heads.append(pr["head"].get("sha"))
    if any(not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head) for head in heads):
        raise SystemExit("Review invalidation revision hints are invalid")

run_id = os.environ.get("GITHUB_RUN_ID", "")
token = os.environ.get("GITHUB_TOKEN", "")
if not re.fullmatch(r"[1-9][0-9]*", run_id) or not token:
    raise SystemExit("Review invalidation publisher identity is unavailable")

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Review invalidation redirect refused")

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

def api(path, method="GET", data=None):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}{path}",
        data=None if data is None else json.dumps(data).encode(), method=method,
        headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28"},
    )
    with opener.open(request, timeout=5) as response:
        if response.status != (201 if method == "POST" else 200):
            raise RuntimeError("Review bootstrap API failed")
        if method == "POST":
            return None
        raw = response.read(3_000_001)
    if len(raw) > 3_000_000:
        raise RuntimeError("Review bootstrap response exceeds bound")
    return json.loads(raw)

def pending(head, description):
    api(f"/statuses/{head}", "POST", {
        "state": "pending", "context": "rightyo/review-gate", "description": description,
        "target_url": f"https://github.com/{repository}/actions/runs/{run_id}",
    })

def persist(head, cutoff):
    description = f"v1 comment:{comment_id} requested:{cutoff.isoformat().replace('+00:00', 'Z')}"
    try:
        pending(head, description)
    finally:
        pending(head, description)

def authorized(user):
    if not isinstance(user, dict) or user.get("type") != "User":
        return False
    login = user.get("login")
    if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", login):
        return False
    result = api(f"/collaborators/{login}/permission")
    if (not isinstance(result, dict) or result.get("permission") not in {
            "none", "read", "triage", "write", "maintain", "admin"}):
        raise RuntimeError("Review request permission metadata is invalid")
    return result.get("permission") in {"admin", "maintain", "write"}

def previous(head):
    latest = None
    for page in range(1, 11):
        batch = api(f"/commits/{head}/statuses?per_page=100&page={page}")
        if not isinstance(batch, list) or len(batch) > 100:
            raise RuntimeError("Review request history is invalid")
        for status in batch:
            creator = status.get("creator") or {}
            description = status.get("description") or ""
            if (creator.get("id") != 41898282 or creator.get("login") != "github-actions[bot]"
                    or status.get("context") not in {
                        "rightyo/review-gate", "rightyo/review-request"}
                    or not description.startswith("v1 comment:")):
                continue
            match = re.fullmatch(r"v1 comment:([1-9][0-9]*) requested:(.+)", description)
            if not match:
                raise RuntimeError("Review request history marker is invalid")
            value = stamp(match[2])
            if int(match[1]) == comment_id:
                latest = max(latest, value) if latest is not None else value
        if len(batch) < 100:
            return latest
    raise RuntimeError("Review request history exceeds bound")

try:
    if event == "issue_comment":
        pr = api(f"/pulls/{number}")
        if pr.get("state") != "open":
            raise SystemExit(0)
        head = pr.get("head", {}).get("sha")
        if (pr.get("base", {}).get("repo", {}).get("full_name") != repository
                or not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head)):
            raise RuntimeError("Review request PR revision is invalid")
        try:
            pending(head, "Review request received; authenticating durable request barrier")
        except Exception:
            persist(head, cutoff)
            raise
        try:
            if not authorized(comment.get("user")) or not authorized(payload.get("sender")):
                raise SystemExit(0)
            if removed:
                existing = previous(head)
                if existing is not None:
                    raise SystemExit(0)
        except Exception:
            persist(head, cutoff)
            raise
        persist(head, cutoff)
    else:
        failed = False
        for head in sorted(set(heads)):
            try:
                pending(head, "Native review activity; awaiting source resolution")
            except Exception:
                failed = True
        if not prs:
            # A review relay may report a merge SHA with no embedded PRs.
            # Discover denial targets before checkout; this never grants approval.
            revision = run["head_sha"]
            associated = api(f"/commits/{revision}/pulls?per_page=100&page=1")
            if not isinstance(associated, list) or len(associated) >= 100:
                raise RuntimeError("Review invalidation associations exceed bound")
            inventory = not associated
            if inventory:
                associated = api("/pulls?state=open&per_page=100&page=1")
                if not isinstance(associated, list) or len(associated) >= 100:
                    raise RuntimeError("Review invalidation PR inventory exceeds bound")
            candidates = set()
            for candidate in associated:
                if not isinstance(candidate, dict):
                    raise RuntimeError("Review invalidation PR metadata is invalid")
                if (candidate.get("state") != "open"
                        or candidate.get("base", {}).get("repo", {}).get("full_name")
                        != repository):
                    continue
                if inventory and revision not in {
                        candidate.get("merge_commit_sha"), candidate.get("head", {}).get("sha")}:
                    continue
                number = candidate.get("number")
                if type(number) is not int or number < 1:
                    raise RuntimeError("Review invalidation PR identity is invalid")
                candidates.add(number)
            if len(candidates) != 1:
                raise RuntimeError("Review invalidation PR association is not unique")
            number = candidates.pop()
            current_pr = api(f"/pulls/{number}")
            current_head = current_pr.get("head", {}).get("sha")
            if (current_pr.get("number") != number or current_pr.get("state") != "open"
                    or current_pr.get("base", {}).get("repo", {}).get("full_name") != repository
                    or not isinstance(current_head, str)
                    or not re.fullmatch(r"[0-9a-f]{40}", current_head)):
                raise RuntimeError("Review invalidation current PR revision is invalid")
            try:
                pending(current_head, "Native review activity; authoritative PR head invalidated")
            except Exception:
                failed = True
        if failed:
            raise RuntimeError("Review invalidation pending publication failed")
except Exception:
    raise SystemExit("Review bootstrap pending or durable capture failed") from None
print("Review denial captured before checkout and source resolution")
PY
"""


def ai_review_workflow_errors(path, document):
    """Freeze the trusted metadata-only publisher; it never invokes model providers."""
    expected_events = {
        "pull_request_target": {
            "types": ["opened", "synchronize", "reopened", "ready_for_review", "edited"]
        },
        "issue_comment": {"types": ["created", "edited", "deleted"]},
        "workflow_run": {
            "workflows": ["RightyO CI", "Native review activity relay"],
            "types": ["completed"],
        },
        "workflow_dispatch": {
            "inputs": {
                field: {"required": True, "type": "string"}
                for field in ("pr_number", "expected_head", "expected_base")
            }
        },
    }
    expected = {
        "on": expected_events,
        "permissions": {},
        "concurrency": {
            "group": "rightyo-subscription-status",
            "queue": "max",
            "cancel-in-progress": False,
        },
        "jobs": {
            "invalidate": {
                "runs-on": "ubuntu-24.04",
                "timeout-minutes": 2,
                "permissions": {"pull-requests": "read", "statuses": "write"},
                "steps": [
                    {
                        "env": {"GITHUB_TOKEN": "${{ github.token }}"},
                        "run": REVIEW_INVALIDATION_COMMAND,
                    }
                ],
            },
            "resolve": {
                "needs": "invalidate",
                "if": (
                    "${{ github.event_name != 'issue_comment' || github.event.issue.pull_request }}"
                ),
                "runs-on": "ubuntu-24.04",
                "timeout-minutes": 5,
                "permissions": {"actions": "read", "contents": "read", "pull-requests": "read"},
                "outputs": {"pr_number": "${{ steps.route.outputs.pr_number }}"},
                "steps": [
                    {
                        "uses": CHECKOUT_ACTION,
                        "with": {
                            "ref": "${{ github.workflow_sha }}",
                            "persist-credentials": False,
                        },
                    },
                    {
                        "id": "route",
                        "run": "python3 scripts/subscription_review.py resolve",
                        "env": {"GITHUB_TOKEN": "${{ github.token }}"},
                    },
                ],
            },
            "record": {
                "needs": "resolve",
                "if": "${{ needs.resolve.outputs.pr_number != '' }}",
                "runs-on": "ubuntu-24.04",
                "timeout-minutes": 5,
                "permissions": {
                    "actions": "read",
                    "contents": "read",
                    "pull-requests": "read",
                    "statuses": "write",
                },
                "steps": [
                    {
                        "uses": CHECKOUT_ACTION,
                        "with": {
                            "ref": "${{ github.workflow_sha }}",
                            "persist-credentials": False,
                        },
                    },
                    {
                        "run": "python3 scripts/subscription_review.py record",
                        "env": {
                            "GITHUB_TOKEN": "${{ github.token }}",
                            "GATE_PR_NUMBER": "${{ needs.resolve.outputs.pr_number }}",
                        },
                    },
                ],
            },
            "gate": {
                "needs": ["resolve", "record"],
                "if": "${{ needs.resolve.outputs.pr_number != '' }}",
                "runs-on": "ubuntu-24.04",
                "timeout-minutes": 5,
                "permissions": {
                    "actions": "read",
                    "contents": "read",
                    "pull-requests": "read",
                    "statuses": "write",
                },
                "steps": [
                    {
                        "uses": CHECKOUT_ACTION,
                        "with": {
                            "ref": "${{ github.workflow_sha }}",
                            "persist-credentials": False,
                        },
                    },
                    {
                        "run": "python3 scripts/subscription_review.py",
                        "env": {
                            "GITHUB_TOKEN": "${{ github.token }}",
                            "GATE_PR_NUMBER": "${{ needs.resolve.outputs.pr_number }}",
                        },
                    },
                ],
            },
        },
    }

    actual = copy.deepcopy(document)
    actual.pop("name", None)
    if True in actual:
        actual["on"] = actual.pop(True)
    jobs = actual.get("jobs")
    if isinstance(jobs, dict):
        for job in jobs.values():
            if isinstance(job, dict):
                job.pop("name", None)
                steps = job.get("steps")
                if isinstance(steps, list):
                    for step in steps:
                        if isinstance(step, dict):
                            step.pop("name", None)
    event = actual.get("on")
    if isinstance(event, dict):
        dispatch = event.get("workflow_dispatch")
        inputs = dispatch.get("inputs") if isinstance(dispatch, dict) else None
        if isinstance(inputs, dict):
            for spec in inputs.values():
                if isinstance(spec, dict):
                    spec.pop("description", None)

    def equal(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, list):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    if not equal(actual, expected) or mentions_secrets(document):
        return [f"{path}: subscription review workflow exceeds reviewed metadata-only surface"]

    return []


def review_activity_workflow_errors(path, document):
    """The PR-merge-tree relay has no credentials, checkout or event-controlled programs."""
    actual = copy.deepcopy(document)
    actual.pop("name", None)
    if True in actual:
        actual["on"] = actual.pop(True)
    jobs = actual.get("jobs")
    if isinstance(jobs, dict):
        for job in jobs.values():
            if isinstance(job, dict):
                job.pop("name", None)
                if isinstance(job.get("steps"), list):
                    for step in job["steps"]:
                        if isinstance(step, dict):
                            step.pop("name", None)
    expected = {
        "on": {
            "pull_request_review": {"types": ["submitted", "edited", "dismissed"]},
            "pull_request_review_comment": {"types": ["created", "edited", "deleted"]},
        },
        "permissions": {},
        "jobs": {
            "notify": {
                "runs-on": "ubuntu-24.04",
                "timeout-minutes": 1,
                "permissions": {},
                "steps": [
                    {
                        "run": "echo 'Review activity notification; "
                        "the trusted publisher reads GitHub metadata independently'"
                    }
                ],
            }
        },
    }

    def equal(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, list):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    if not equal(actual, expected) or mentions_secrets(document):
        return [f"{path}: review relay exceeds fixed unprivileged notification surface"]
    return []


def actionlint_source(path, source):
    """Only the frozen queue field is omitted for the pinned pre-queue actionlint parser."""
    if str(path) != AI_REVIEW_WORKFLOW:
        return source
    try:
        document = yaml.safe_load(source)
    except yaml.YAMLError:
        raise ValueError("Review workflow compatibility input is invalid") from None
    if workflow_errors(path, document) or source.count("  queue: max\n") != 1:
        raise ValueError("Review workflow compatibility requires exact frozen queue policy")
    return source.replace("  queue: max\n", "", 1)


def workflow_errors(path, document):
    """Narrow baseline policy; reject the secrets token even in harmless text labels."""
    errors = []
    if not isinstance(document, dict):
        return [f"{path}: workflow must be a mapping"]

    if str(path) == AI_REVIEW_WORKFLOW:
        return ai_review_workflow_errors(path, document)
    if str(path) == REVIEW_RELAY_WORKFLOW:
        return review_activity_workflow_errors(path, document)

    # Walk actual strings: repr/JSON escaping can hide word boundaries around newlines.
    if mentions_secrets(document):
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
