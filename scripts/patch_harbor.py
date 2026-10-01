#!/usr/bin/env python3
"""Raise Terminus-2's tool-install timeouts so tmux/asciinema can be apt-installed
inside qemu-emulated amd64 containers. Upstream defaults (120s per command,
240s total) assume native speed; under emulation apt routinely overruns them and
the trial proceeds without tmux, which Terminus needs to drive the terminal.

Idempotent. Re-run after any `uv tool upgrade harbor`.
"""
import glob, pathlib, re, sys

SUBS = [
    (r"_TOOL_INSTALL_TIMEOUT_SEC = \d+", "_TOOL_INSTALL_TIMEOUT_SEC = 600"),
    (r"_TOOL_INSTALL_BUDGET_SEC = \d+",  "_TOOL_INSTALL_BUDGET_SEC = 1200"),
]

matches = glob.glob(str(pathlib.Path.home() /
    ".local/share/uv/tools/harbor/lib/python*/site-packages/harbor"
    "/agents/terminus_2/tmux_session.py"))
if not matches:
    sys.exit("could not find harbor's tmux_session.py")

p = pathlib.Path(matches[0])
t = orig = p.read_text()
for pat, rep in SUBS:
    t, n = re.subn(pat, rep, t)
    if n != 1:
        sys.exit(f"expected 1 match for {pat!r}, got {n}")
if t != orig:
    p.write_text(t)
    print(f"patched {p}")
else:
    print("already patched")
for pat, rep in SUBS:
    print("  ", rep)
