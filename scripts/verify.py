"""Run the same offline repository, lint, unit and optional package checks as CI."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from repository_checks import (
    artifact_errors,
    git_files,
    ignore_errors,
    markdown_errors,
    workflow_errors,
)

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd=ROOT, env=None):
    print("Running: " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True, timeout=180)


def repository_errors(root):
    paths = git_files(root)
    errors = artifact_errors(root, paths)
    if errors:
        # Stop before format/parser diagnostics can reveal identifying paths or content.
        return errors
    errors = ignore_errors(root)
    for required in ("README.md", "LICENSE", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md"):
        if not (root / required).is_file():
            errors.append(f"required repository document missing: {required}")
    if (root / "LICENSE").exists() and "GNU AFFERO GENERAL PUBLIC LICENSE" not in (
        root / "LICENSE"
    ).read_text():
        errors.append("LICENSE must contain the declared AGPL license")
    for path in paths:
        if path.suffix == ".md":
            errors.extend(markdown_errors(root, path))
        if path.suffix in (".yaml", ".yml"):
            try:
                document = yaml.safe_load((root / path).read_text())
                if path.parts[:2] == (".github", "workflows"):
                    errors.extend(workflow_errors(path, document))
            except yaml.YAMLError:
                errors.append(f"{path}: invalid YAML")
        if path.suffix == ".json":
            try:
                json.loads((root / path).read_text())
            except ValueError:
                errors.append(f"{path}: invalid JSON")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actionlint", default=shutil.which("actionlint"))
    args = parser.parse_args()
    if not args.actionlint:
        parser.error("install pinned actionlint first; see docs/ci.md")
    errors = repository_errors(ROOT)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print("Repository metadata, tracked artifacts, ignore rules, Markdown and YAML/JSON passed")
    version = subprocess.run(
        [args.actionlint, "-version"], check=True, capture_output=True, text=True, timeout=10
    ).stdout.splitlines()[0]
    if version != "1.7.7":
        raise SystemExit("actionlint must be the pinned version 1.7.7")
    # Disable optional host shellcheck/pyflakes discovery to keep local/CI parity.
    run([args.actionlint, "-shellcheck=", "-pyflakes="])
    python_paths = [name for name in ("scripts", "tests", "src") if (ROOT / name).is_dir()]
    run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E,F,I",
            "--target-version",
            "py311",
            "--line-length",
            "100",
            *python_paths,
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "--line-length",
            "100",
            "--target-version",
            "py311",
            *python_paths,
        ]
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], env=env)
    browser_script = ROOT / "src" / "rightyo" / "web" / "app.js"
    if browser_script.is_file():
        node = shutil.which("node")
        if not node:
            raise SystemExit("Install Node.js 20+ for the prototype JavaScript syntax check")
        run([node, "--check", browser_script])
    if (ROOT / "pyproject.toml").is_file() and (ROOT / "src").is_dir():
        with tempfile.TemporaryDirectory(prefix="rightyo-build-") as directory:
            temp = Path(directory)
            run([sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", temp])
            wheels = list(temp.glob("*.whl"))
            if len(wheels) != 1:
                raise SystemExit("package build must produce exactly one wheel")
            installed = temp / "installed"
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-index",
                    "--target",
                    installed,
                    wheels[0],
                ]
            )
            fixture = ROOT / "examples" / "synthetic-turns.json"
            if not fixture.exists():
                raise SystemExit("package smoke requires the explicit synthetic fixture")
            run(
                [
                    sys.executable,
                    "-m",
                    "rightyo",
                    "evaluate",
                    "--input",
                    fixture,
                    "--provider",
                    "mock",
                ],
                cwd=temp,
                env=dict(os.environ, PYTHONPATH=str(installed)),
            )
            run(
                [
                    sys.executable,
                    "-c",
                    "import json, subprocess, sys; "
                    "result = subprocess.run([sys.executable, '-m', 'rightyo', 'tool-replay', "
                    "'--input', sys.argv[1]], check=True, capture_output=True, text=True); "
                    "events = [json.loads(line) for line in result.stdout.splitlines()]; "
                    "assert events[0]['type'] == 'session' and "
                    "events[0]['phase'] == 'started'; "
                    "assert events[-1]['phase'] == 'stopped'; "
                    "requests = [e for e in events if e['type'] == 'request']; "
                    "assert len(requests) == 1 and requests[0]['turn']['finalized']; "
                    "assert requests[0]['decision']['provider'] == 'mock'; "
                    "assert all(e['schema_version'] == 1 for e in events); "
                    "print('Installed headless tool contract smoke passed (authored mock)')",
                    fixture,
                ],
                cwd=temp,
                env=dict(os.environ, PYTHONPATH=str(installed)),
            )
            if browser_script.is_file():
                run(
                    [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import rightyo.prototype as lab; "
                        "assets = Path(lab.__file__).parent / 'web'; "
                        "assert all((assets / name).is_file() for name in "
                        "('index.html', 'style.css', 'app.js'))",
                    ],
                    cwd=temp,
                    env=dict(os.environ, PYTHONPATH=str(installed)),
                )
    else:
        print("Package checks not applicable: no Python package exists in this checkout")
    print("All applicable local/CI checks passed")


if __name__ == "__main__":
    main()
