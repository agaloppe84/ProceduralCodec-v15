#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, subprocess, pathlib, datetime, sys
from fnmatch import fnmatch
from typing import List, Dict, Any, Tuple

DEFAULT_EXCLUDES = [
    ".git/**","**/__pycache__/**","**/.pytest_cache/**","**/.mypy_cache/**",
    "**/.idea/**","**/.vscode/**","**/node_modules/**","**/.venv/**","**/venv/**",
    "storage/**","datasets/**",
    "**/*.png","**/*.jpg","**/*.jpeg","**/*.gif","**/*.pdf",
    "**/*.zip","**/*.tar","**/*.7z","**/*.mp4","**/*.pc15",
]

DEFAULT_INCLUDE_EXTS = {
    ".py",".md",".txt",".toml",".json",".yaml",".yml",".ini",".cfg",
    ".sh",".bat",".ps1",".ipynb",".c",".h",".cpp",".hpp",".cu",".cuh",".java",
    ".kt",".rs",".go",".php",".rb",".ts",".tsx",".js",".jsx",".css",".scss",
    ".html",".sql",".m",".mm"
}
DEFAULT_INCLUDE_BASENAMES = {"Dockerfile", "Makefile", "LICENSE"}

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)

def git_ls_files() -> List[str]:
    out = run(["git", "ls-files", "-z"]).stdout
    return [p for p in out.split("\x00") if p]

def match_glob(path: str, patterns: List[str]) -> bool:
    return any(fnmatch(path, pat) for pat in patterns)

def lang_from_path(path: str) -> str:
    name = pathlib.Path(path).name
    ext = pathlib.Path(path).suffix.lower()
    if name == "Dockerfile":
        return "dockerfile"
    return {
        ".py":"python",".md":"md",".json":"json",".toml":"toml",".yaml":"yaml",".yml":"yaml",
        ".ini":"ini",".cfg":"ini",".txt":"text",".sh":"bash",".bat":"bat",".ps1":"powershell",
        ".c":"c",".h":"c",".cpp":"cpp",".hpp":"cpp",".cu":"cuda",".cuh":"cuda",
        ".java":"java",".kt":"kotlin",".rs":"rust",".go":"go",".php":"php",".rb":"ruby",
        ".ts":"ts",".tsx":"tsx",".js":"javascript",".jsx":"jsx",".css":"css",".scss":"scss",
        ".html":"html",".sql":"sql",".ipynb":"json",".m":"objectivec",".mm":"objectivec"
    }.get(ext, "")

def is_text_and_read(path: str, max_bytes: int) -> Tuple[bool, Tuple[str, bytes] | None]:
    p = pathlib.Path(path)
    try:
        data = p.read_bytes()
    except Exception:
        return False, None
    if b"\x00" in data[:1024]:
        return False, None
    for enc in ("utf-8", "latin-1"):
        try:
            _ = data[:max_bytes].decode(enc)
            return True, (enc, data)
        except UnicodeDecodeError:
            continue
    return False, None

def build_tree(paths: List[str]) -> str:
    root: Dict[str, Any] = {}
    for p in paths:
        parts = p.split("/")
        node = root
        for d in parts[:-1]:
            node = node.setdefault(d, {})
        node.setdefault("__files__", []).append(parts[-1])
    lines: List[str] = []
    def rec(node: Dict[str, Any], prefix: str = ""):
        dirs = sorted([k for k in node if k != "__files__"])
        files = sorted(node.get("__files__", []))
        entries = [(d, True) for d in dirs] + [(f, False) for f in files]
        for i, (name, is_dir) in enumerate(entries):
            last = (i == len(entries) - 1)
            connector = "└── " if last else "├── "
            lines.append(f"{prefix}{connector}{name}{'/' if is_dir else ''}")
            if is_dir:
                rec(node[name], prefix + ("    " if last else "│   "))
    rec(root)
    return "\n".join(lines)

def resolve_out_name(cli_output: str | None) -> str:
    if cli_output:
        return cli_output
    env = os.environ.get("CODE_STATE_FILE")
    if env:
        return env
    run_number = os.environ.get("GITHUB_RUN_NUMBER")
    if run_number:
        return f"code_state_v{run_number}.md"
    return "code_state.md"

def export_github_output(path: str):
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"file={path}\n")

def main():
    ap = argparse.ArgumentParser(description="Generate a versioned code_state markdown snapshot.")
    ap.add_argument("--output", help="Override output filename (otherwise v${GITHUB_RUN_NUMBER}).")
    ap.add_argument("--max-bytes", type=int, default=200_000)
    ap.add_argument("--exclude-glob", action="append", default=[], help="Extra glob to exclude (repeatable).")
    ap.add_argument("--include-ext", default=",".join(sorted(DEFAULT_INCLUDE_EXTS)))
    args = ap.parse_args()

    include_exts = {e.strip() for e in args.include_ext.split(",") if e.strip()}
    include_basenames = set(DEFAULT_INCLUDE_BASENAMES)
    excludes = list(DEFAULT_EXCLUDES) + args.exclude_glob

    out_name = resolve_out_name(args.output)
    sha = os.environ.get("GITHUB_SHA", "")[:12]
    ref = os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_REF", "")
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    files = [p for p in git_ls_files() if p != out_name and not match_glob(p, excludes)]
    tree_txt = build_tree(files)

    # ------ Header Markdown COMPLET (⚠️ keep the backticks) ------
    header = f"""# Code State Snapshot

_Generated by CI. Do not edit._

- **Commit**: `{sha}`
- **Ref**: `{ref}`
- **UTC time**: `{now}`

## Project tree
```
{tree_txt}
```

## Files
"""

    # Write file + dump contents
    out = pathlib.Path(out_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header, encoding="utf-8", newline="\n")

    with out.open("a", encoding="utf-8", newline="\n") as f:
        for path in files:
            name = pathlib.Path(path).name
            ext = pathlib.Path(path).suffix.lower()
            show_content = (ext in include_exts) or (name in include_basenames)

            is_text, payload = is_text_and_read(path, args.max_bytes)
            reason = None
            if not is_text:
                show_content = False
                reason = "binary or undecodable"
            elif not show_content:
                reason = "extension/basename not in include list"

            f.write(f"\n### `{path}`\n\n")
            if show_content and payload:
                enc, data = payload
                truncated = ""
                if len(data) > args.max_bytes:
                    data = data[:args.max_bytes]
                    truncated = f"\n\n> _Truncated to first {args.max_bytes} bytes._"
                try:
                    txt = data.decode(enc, errors="replace")
                except Exception:
                    txt = data.decode("utf-8", errors="replace")
                lang = lang_from_path(path)
                f.write(f"```{lang}\n{txt}\n```\n{truncated}\n")
            else:
                f.write(f"> _Content omitted ({reason})._\n")

    export_github_output(str(out))
    print(f"Wrote {out} ({out.stat().st_size} bytes)")

if __name__ == "__main__":
    sys.exit(main())
