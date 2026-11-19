#!/usr/bin/env python3
"""Complexity ratchet for the llama.cpp sources.

Counts decision points (`if`/`else if`/`for`/`while`/`do`/`switch`/`case`/`catch`,
plus `&&`, `||` and `?:`) per function and per file, and compares the result with
the snapshot in scripts/complexity-budget.json.

  ./scripts/check-complexity.py            # check, exit 1 on any regression
  ./scripts/check-complexity.py --update   # accept the current tree as the budget
  ./scripts/check-complexity.py --top 25   # show the worst offenders

The budget only ratchets downwards: a function or file that grows, or a new one
that lands above the floor, fails until it is either simplified or added with
--update in the same commit. That is the point - the snapshot is small enough to
review in a diff.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = os.path.join(ROOT, "scripts", "complexity-budget.json")

# trees that hold hand-written code; vendor/, pocs/ and generated files are out
TREES = ["src", "ggml/src", "common", "tools", "examples", "include"]
EXTS = (".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm")
SKIP_DIRS = ("/vendor/", "/pocs/", "/openvino/openvino/")

# entries below these are not tracked: noise, and the snapshot should stay readable
FUNC_FLOOR = 40
FILE_FLOOR = 300

TOKEN_RE = re.compile(
    r'("(?:\\.|[^"\\])*")'          # string literal
    r"|('(?:\\.|[^'\\])*')"         # char literal
    r"|(//[^\n]*)"                  # line comment
    r"|(/\*.*?\*/)",                # block comment
    re.S | re.X,
)
DECISION_RE = re.compile(r"\b(if|for|while|do|case|catch)\b|\bswitch\b|&&|\|\||\?")
DEF_RE = re.compile(r"^[A-Za-z_][\w \t\*&:<>,]*\b(\w+)\s*\(")
NOT_A_DEF = ("return", "typedef", "using", "else", "case", "if", "while", "for", "switch")


def strip_code(src: str) -> str:
    return TOKEN_RE.sub(lambda m: " " if (m.group(1) or m.group(2)) else "", src)


def scan_file(path: str):
    """Return (file_decisions, file_sloc, [(name, line, decisions, sloc), ...])."""
    code = strip_code(open(path, "r", errors="ignore").read())
    lines = code.splitlines()
    sloc = sum(1 for ln in lines if ln.strip() and not ln.lstrip().startswith("#"))
    total = len(DECISION_RE.findall(code))

    funcs = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        m = DEF_RE.match(ln)
        if m and not ln.strip().startswith(NOT_A_DEF) and not ln.strip().startswith(("#", "//")):
            j = i
            while j < min(i + 4, n) and "{" not in lines[j]:
                j += 1
            if j < n and "{" in lines[j]:
                depth, body = 0, []
                for k in range(j, n):
                    # a definition at column 0 while still inside braces means the
                    # body's braces are unbalanced (#if alternatives do that in C):
                    # treat it as the end of this function instead of swallowing the file
                    if k > j + 1 and depth > 0 and DEF_RE.match(lines[k]) and \
                            not lines[k][:1].isspace():
                        break
                    body.append(lines[k])
                    depth += lines[k].count("{") - lines[k].count("}")
                    if depth <= 0:
                        break
                text = "\n".join(body)
                dec = len(DECISION_RE.findall(text))
                if dec >= FUNC_FLOOR:
                    funcs.append((m.group(1), i + 1, dec, sum(1 for b in body if b.strip())))
                i = k + 1
                continue
        i += 1
    return total, sloc, funcs


def measure():
    files = {}
    functions = {}
    for tree in TREES:
        base = os.path.join(ROOT, tree)
        for dirpath, _, names in os.walk(base):
            if any(s in dirpath + "/" for s in SKIP_DIRS):
                continue
            for name in names:
                if not name.endswith(EXTS):
                    continue
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, ROOT)
                total, sloc, funcs = scan_file(path)
                if total >= FILE_FLOOR:
                    files[rel] = {"decisions": total, "sloc": sloc}
                for fname, line, dec, fsloc in funcs:
                    functions[f"{rel}::{fname}"] = {"decisions": dec, "sloc": fsloc, "line": line}
    return {"files": files, "functions": functions}


def compare(budget, current):
    problems = []
    for kind in ("files", "functions"):
        for key, now in current[kind].items():
            was = budget[kind].get(key)
            if was is None:
                problems.append(f"new {kind[:-1]} above floor: {key} ({now['decisions']} decisions)")
            elif now["decisions"] > was["decisions"]:
                problems.append(
                    f"{kind[:-1]} grew: {key} {was['decisions']} -> {now['decisions']} decisions")
        # going down is good: mention it so the budget can be tightened in the same PR
    return problems


def improvements(budget, current):
    out = []
    for kind in ("files", "functions"):
        for key, was in budget[kind].items():
            now = current[kind].get(key)
            if now is None or now["decisions"] < was["decisions"]:
                out.append(f"{kind[:-1]} improved or gone: {key} {was['decisions']} -> "
                           f"{now['decisions'] if now else 0}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="rewrite the budget from the current tree")
    ap.add_argument("--top", type=int, default=0, help="print the N worst functions and exit")
    args = ap.parse_args()

    current = measure()

    if args.top:
        rows = sorted(current["functions"].items(), key=lambda kv: -kv[1]["decisions"])
        print(f"worst {args.top} functions by decision points:")
        for key, val in rows[:args.top]:
            print(f"{val['decisions']:6} dec {val['sloc']:6} sloc  {key}:{val['line']}")
        return 0

    if args.update:
        payload = {
            "_comment": "generated by scripts/check-complexity.py --update; only ratchets down",
            "floors": {"function_decisions": FUNC_FLOOR, "file_decisions": FILE_FLOOR},
            "totals": {
                "files": len(current["files"]),
                "file_decisions": sum(v["decisions"] for v in current["files"].values()),
                "function_decisions": sum(v["decisions"] for v in current["functions"].values()),
            },
            **current,
        }
        with open(BUDGET, "w") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {os.path.relpath(BUDGET, ROOT)}: "
              f"{len(current['files'])} files, {len(current['functions'])} functions")
        return 0

    if not os.path.exists(BUDGET):
        print(f"error: {os.path.relpath(BUDGET, ROOT)} is missing; run with --update",
              file=sys.stderr)
        return 2

    budget = json.load(open(BUDGET))
    problems = compare(budget, current)
    improved = improvements(budget, current)

    for line in improved:
        print(f"  ok  {line}")

    if problems:
        print(f"\ncomplexity ratchet: {len(problems)} regression(s)", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        print("\nsimplify the code, or accept it deliberately with "
              "scripts/check-complexity.py --update", file=sys.stderr)
        return 1

    totals = current["files"], current["functions"]
    print(f"complexity ratchet: ok ({len(totals[0])} files, {len(totals[1])} functions tracked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
