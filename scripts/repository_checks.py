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
AI_REVIEW_JOBS = {"prepare", "codex", "claude", "publish"}
AI_REVIEW_EVENTS = {"pull_request_target", "workflow_dispatch", "workflow_run"}
AI_REVIEW_TYPES = {"opened", "synchronize", "reopened", "ready_for_review", "edited"}
AI_REVIEW_PERMISSIONS = {
    "prepare": {"contents": "read", "pull-requests": "read", "checks": "write"},
    "codex": {"contents": "read"},
    "claude": {"contents": "read"},
    "publish": {"contents": "read", "checks": "write", "pull-requests": "read"},
}
CHECKOUT_ACTION = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
CODEX_REVIEW_ACTION = "openai/codex-action@86365089eb2b84e0a8fb0717b304f8bdcb13b20e"
UPLOAD_REVIEW_ACTION = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
DOWNLOAD_REVIEW_ACTION = "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
AI_REVIEW_ACTIONS = {
    CHECKOUT_ACTION,
    CODEX_REVIEW_ACTION,
    UPLOAD_REVIEW_ACTION,
    DOWNLOAD_REVIEW_ACTION,
}
CLAUDE_REVIEW_COMMAND = "python3 scripts/ai_review.py claude --directory review-data"
PROVIDER_ALLOWED_CONDITION = "${{ needs.prepare.outputs.allowed == 'true' }}"

AI_REVIEW_ROUTE = (
    "(github.event_name != 'workflow_run' || "
    "github.event.workflow_run.event == 'pull_request') && "
    "(github.event_name != 'pull_request_target' || github.actor != 'dependabot[bot]')"
)
AI_REVIEW_PREPARE_CONDITION = "${{ " + AI_REVIEW_ROUTE + " }}"
AI_REVIEW_PUBLISH_CONDITION = "${{ always() && (" + AI_REVIEW_ROUTE + ") }}"
AI_REVIEW_CONCURRENCY = {
    "group": (
        "rightyo-ai-${{ github.event.pull_request.number || inputs.pr_number || "
        "github.event.workflow_run.pull_requests[0].number }}-${{ "
        "github.event.pull_request.head.sha || inputs.expected_head || "
        "github.event.workflow_run.head_sha }}-${{ "
        "github.event_name == 'workflow_dispatch' && 'manual' || "
        "((github.event_name == 'workflow_run' && "
        "github.event.workflow_run.event != 'pull_request') || "
        "(github.event_name == 'pull_request_target' && "
        "(github.actor == 'dependabot[bot]' || "
        "github.event.pull_request.head.repo.full_name != github.repository))) "
        "&& 'ignored' || 'automatic' }}"
    ),
    "cancel-in-progress": "${{ github.event_name == 'workflow_dispatch' }}",
}


CODEX_REVIEW_PERMISSIONS = {
    "review-data-only": {
        "filesystem": {":minimal": "read", "/tmp/rightyo-review-empty": "read"},
        "network": {"enabled": False},
    },
}

CODEX_REVIEW_CONFIG = {
    "approval_policy": "never",
    "project_doc_max_bytes": 0,
    "web_search": "disabled",
    "shell_environment_policy": {"inherit": "none"},
    "features": {
        feature: False
        for feature in (
            "shell_tool",
            "unified_exec",
            "shell_snapshot",
            "view_image",
            "apps",
            "plugins",
            "collab",
            "multi_agent",
            "js_repl",
            "code_mode",
            "hooks",
            "codex_hooks",
            "memory_tool",
            "memories",
            "search_tool",
            "skill_search",
            "workspace_dependencies",
            "image_generation",
        )
    },
    "permissions": CODEX_REVIEW_PERMISSIONS,
}


def codex_profile_errors(document):
    """Freeze the entire trusted CLI config; preflight proves runner enforcement.

    Explicit root denial can mask allowed runtime mounts on Linux. The reviewed
    profile grants only minimal runtime files and the literal empty cwd; all other
    paths remain denied by default. The complete config also freezes disabled tools,
    customization discovery, environment inheritance and network access.
    """

    def matches(actual, expected):
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, dict):
            return set(actual) == set(expected) and all(
                matches(actual[key], value) for key, value in expected.items()
            )
        return actual == expected

    if not matches(document, CODEX_REVIEW_CONFIG):
        return ["Codex review config must match the complete reviewed capability policy"]
    return []


def approved_ai_review_steps():
    """The credentialed lane may execute only this reviewed orchestration surface.

    Human-readable step names may change. Commands, ordering, inputs, env, conditions,
    output IDs and artifact locations must be reviewed here alongside the workflow.
    """
    checkout = {
        "uses": CHECKOUT_ACTION,
        "with": {
            "ref": "${{ github.workflow_sha }}",
            "persist-credentials": False,
        },
    }
    download = {
        "uses": DOWNLOAD_REVIEW_ACTION,
        "with": {
            "name": "rightyo-review-snapshot",
            "path": "review-data",
        },
    }
    install = {"run": "npm ci --ignore-scripts --no-audit --no-fund --prefix .github/reviews"}
    snapshot_env = {"SNAPSHOT_SHA": "${{ needs.prepare.outputs.snapshot_sha }}"}

    def upload(provider):
        name = "snapshot" if provider == "snapshot" else provider
        return {
            "uses": UPLOAD_REVIEW_ACTION,
            "with": {
                "name": "rightyo-review-" + name,
                "path": "review-data/" + name + ".json",
                "if-no-files-found": "error",
                "retention-days": 1,
            },
        }

    return {
        "prepare": [
            checkout,
            {
                "id": "snapshot",
                "run": "python3 scripts/ai_review.py prepare --directory review-data",
                "env": {
                    "GITHUB_TOKEN": "${{ github.token }}",
                    "GITHUB_WORKFLOW_SHA": "${{ github.workflow_sha }}",
                },
            },
            upload("snapshot"),
        ],
        "codex": [
            checkout,
            download,
            install,
            {
                "run": "python3 scripts/ai_review.py sandbox-check --directory review-data",
                "env": snapshot_env,
            },
            {
                "uses": CODEX_REVIEW_ACTION,
                "with": {
                    "openai-api-key": "${{ secrets.OPENAI_API_KEY }}",
                    "codex-version": "0.159.2",
                    "prompt-file": "review-data/prompt.txt",
                    "output-schema-file": "review-data/schema.json",
                    "output-file": "review-data/codex-output.json",
                    "working-directory": "/tmp/rightyo-review-empty",
                    "codex-home": "/tmp/rightyo-codex-home",
                    "permission-profile": "review-data-only",
                    "safety-strategy": "drop-sudo",
                    "allow-bot-users": "dependabot[bot]",
                    "codex-args": '["--ephemeral", "--skip-git-repo-check"]',
                },
            },
            {
                "run": "python3 scripts/ai_review.py codex --directory review-data",
                "env": snapshot_env,
            },
            upload("codex"),
        ],
        "claude": [
            checkout,
            download,
            install,
            {
                "run": CLAUDE_REVIEW_COMMAND,
                "env": {
                    **snapshot_env,
                    "ANTHROPIC_API_KEY": "${{ secrets.ANTHROPIC_API_KEY }}",
                    "CLAUDE_CODE_OAUTH_TOKEN": "${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}",
                },
            },
            upload("claude"),
        ],
        "publish": [
            checkout,
            {
                "uses": DOWNLOAD_REVIEW_ACTION,
                "with": {
                    "pattern": "rightyo-review-*",
                    "path": "review-data",
                    "merge-multiple": True,
                },
            },
            {
                "run": "python3 scripts/ai_review.py publish --directory review-data",
                "if": "${{ always() }}",
                "env": {
                    "GITHUB_TOKEN": "${{ github.token }}",
                    "REVIEW_METADATA": "${{ needs.prepare.outputs.metadata }}",
                    "SNAPSHOT_SHA": "${{ needs.prepare.outputs.snapshot_sha }}",
                    "JOB_RESULTS": "${{ toJSON(needs) }}",
                },
            },
        ],
    }


def ai_review_workflow_errors(path, document):
    """Allow the separately reviewed trusted lane without weakening ordinary CI.

    This is a structural guard, not proof of the trusted snapshot/validator programs.
    Changes to those programs and this policy still require independent exact-head review.
    """
    errors = []

    def reject(reason):
        errors.append(f"{path}: AI review {reason}")

    allowed_root = {"name", "on", True, "permissions", "concurrency", "jobs"}
    if set(document) - allowed_root:
        reject("workflow contains unsupported root settings")
    if document.get("permissions") != {}:
        reject("root permissions must be empty")
    if document.get("concurrency") != AI_REVIEW_CONCURRENCY:
        reject("concurrency must isolate ignored callbacks and preserve authorized fork runs")
    event = document.get("on", document.get(True, {}))
    if not isinstance(event, dict) or set(event) != AI_REVIEW_EVENTS:
        reject("requires only pull_request_target, workflow_dispatch and trusted workflow_run")
    else:
        target = event["pull_request_target"]
        if (
            not isinstance(target, dict)
            or set(target) != {"types"}
            or not isinstance(target["types"], list)
            or not all(isinstance(item, str) for item in target["types"])
            or set(target["types"]) != AI_REVIEW_TYPES
        ):
            reject("target events must cover all prescribed PR changes without filters")
        if event["workflow_run"] != {"workflows": ["RightyO CI"], "types": ["completed"]}:
            reject("workflow_run must consume only completed RightyO CI events")
        dispatch = event["workflow_dispatch"]
        inputs = dispatch.get("inputs", {}) if isinstance(dispatch, dict) else {}
        if (
            not isinstance(dispatch, dict)
            or set(dispatch) != {"inputs"}
            or not isinstance(inputs, dict)
            or set(inputs) != {"pr_number", "expected_head", "expected_base"}
            or any(
                not isinstance(spec, dict)
                or spec.get("required") is not True
                or spec.get("type") != "string"
                or set(spec) - {"description", "required", "type"}
                for spec in inputs.values()
            )
        ):
            reject("manual dispatch requires PR number, expected head and expected base strings")
    jobs = document.get("jobs", {})
    if not isinstance(jobs, dict) or set(jobs) != AI_REVIEW_JOBS:
        reject("requires prepare, codex, claude and publish jobs")
        return errors
    allowed_job = {
        "name",
        "runs-on",
        "timeout-minutes",
        "permissions",
        "needs",
        "outputs",
        "if",
        "steps",
    }
    allowed_step = {"id", "name", "uses", "with", "run", "env", "if"}
    approved_steps = approved_ai_review_steps()
    for name, job in jobs.items():
        if not isinstance(job, dict):
            reject(f"{name} job must be a mapping")
            continue
        if set(job) - allowed_job:
            reject(f"{name} job contains unsupported runner settings")
        if job.get("runs-on") != "ubuntu-24.04":
            reject(f"{name} requires an ephemeral ubuntu-24.04 runner")
        timeout = job.get("timeout-minutes")
        if type(timeout) is not int or not 1 <= timeout <= 15:
            reject(f"{name} timeout must be 1-15 minutes")
        permission = job.get("permissions")
        expected = AI_REVIEW_PERMISSIONS[name]
        if permission != expected and not (name in {"codex", "claude"} and permission == {}):
            reject(f"{name} permissions exceed its role")
        needed = job.get("needs", [])
        needed = [needed] if isinstance(needed, str) else needed
        expected_needs = [] if name == "prepare" else ["prepare"]
        if name == "publish":
            expected_needs = ["prepare", "codex", "claude"]
        if (
            not isinstance(needed, list)
            or not all(isinstance(item, str) for item in needed)
            or set(needed) != set(expected_needs)
        ):
            reject(f"{name} dependencies do not enforce the review boundary")
        if name == "publish":
            if job.get("if") != AI_REVIEW_PUBLISH_CONDITION:
                reject("publisher must always handle authorized routes and skip ignored callbacks")
        elif name in {"codex", "claude"}:
            if job.get("if") != PROVIDER_ALLOWED_CONDITION:
                reject(f"{name} job must respect the trusted snapshot authorization output")
        elif job.get("if") != AI_REVIEW_PREPARE_CONDITION:
            reject("preparation must reject restricted bot targets and non-PR workflow callbacks")
        expected_outputs = (
            {
                "allowed": "${{ steps.snapshot.outputs.allowed }}",
                "metadata": "${{ steps.snapshot.outputs.metadata }}",
                "snapshot_sha": "${{ steps.snapshot.outputs.snapshot_sha }}",
            }
            if name == "prepare"
            else None
        )
        if job.get("outputs") != expected_outputs:
            reject(f"{name} outputs must come only from the trusted preparation step")
        steps = job.get("steps", [])
        if not isinstance(steps, list) or not steps:
            reject(f"{name} requires explicit trusted steps")
            continue
        stripped_steps = [
            {key: value for key, value in step.items() if key != "name"}
            if isinstance(step, dict)
            else step
            for step in steps
        ]
        if stripped_steps != approved_steps[name]:
            reject(f"{name} step sequence, commands, inputs or environment exceed reviewed surface")
        for step in steps:
            if not isinstance(step, dict):
                reject(f"{name} step must be a mapping")
                continue
            if set(step) - allowed_step:
                reject(f"{name} step contains unsupported settings")
            action = step.get("uses", "")
            if action:
                if (
                    not isinstance(action, str)
                    or not re.fullmatch(r"[\w.-]+/[\w./-]+@[a-f0-9]{40}", action)
                    or action not in AI_REVIEW_ACTIONS
                ):
                    reject("actions require an approved official repository and full commit SHA")
                elif action.startswith("openai/codex-action@"):
                    if name != "codex" or action != CODEX_REVIEW_ACTION:
                        reject("Codex action must use the reviewed pin in its own job")
                elif action.startswith("actions/checkout@"):
                    options = step.get("with", {})
                    if (
                        action != CHECKOUT_ACTION
                        or not isinstance(options, dict)
                        or options.get("ref") != "${{ github.workflow_sha }}"
                        or options.get("persist-credentials") is not False
                        or set(options) - {"ref", "persist-credentials", "fetch-depth", "path"}
                    ):
                        reject(
                            "checkout must use trusted immutable workflow SHA without credentials"
                        )
            command = step.get("run", "")
            if not isinstance(command, str):
                reject("shell commands must be strings")
            elif "${{" in command:
                reject("pass expressions through env instead of shell interpolation")
            for field in ("with", "env"):
                if field in step and not isinstance(step[field], dict):
                    reject(f"step {field} must be a mapping")
            # Remove only explicitly allowed credential slots, then inspect the whole step.
            # Labels, bracket access, fallback expressions and any other scope still fail.
            credential_free = dict(step)
            if name == "codex" and action == CODEX_REVIEW_ACTION:
                options = step.get("with", {})
                options = dict(options) if isinstance(options, dict) else {}
                if options.get("openai-api-key") == "${{ secrets.OPENAI_API_KEY }}":
                    del options["openai-api-key"]
                credential_free["with"] = options
            if name == "claude" and step.get("run") == CLAUDE_REVIEW_COMMAND:
                environment = step.get("env", {})
                environment = dict(environment) if isinstance(environment, dict) else {}
                for variable in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
                    if environment.get(variable) == "${{ secrets." + variable + " }}":
                        del environment[variable]
                credential_free["env"] = environment
            if mentions_secrets(credential_free):
                reject("credentials are permitted only in the scoped provider invocation")
        nonsteps = {key: value for key, value in job.items() if key != "steps"}
        if mentions_secrets(nonsteps):
            reject("credentials are forbidden outside provider steps")
    root_without_jobs = {key: value for key, value in document.items() if key != "jobs"}
    if mentions_secrets(root_without_jobs):
        reject("credentials are forbidden at workflow scope")
    return errors


def workflow_errors(path, document):
    """Narrow baseline policy; reject the secrets token even in harmless text labels."""
    errors = []
    if not isinstance(document, dict):
        return [f"{path}: workflow must be a mapping"]

    if str(path) == AI_REVIEW_WORKFLOW:
        return ai_review_workflow_errors(path, document)

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
