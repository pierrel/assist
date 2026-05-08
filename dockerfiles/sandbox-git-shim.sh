#!/bin/bash
# Sandbox-wide refusal of git push, regardless of how the call was
# constructed.  Sits at /usr/bin/git in the assist-sandbox image,
# replacing the upstream binary (moved aside to /usr/bin/git-real).
#
# Why a binary-level wrapper exists in addition to the static
# GitPushBlockerMiddleware:
#
# The middleware reads the literal `command` string from the
# `execute` tool call and rejects forms it can lex.  It catches
# direct invocations and one-level shell-outs (bash -c "git push",
# eval "..."), but cannot follow into interpreter-mediated dynamic
# construction — Python `subprocess.run(['git','push'])`, char-code
# assembly, base64-decoded payloads, file-mediated `bash <script>`,
# etc.  No string-matching scheme can, since computation is
# universal.
#
# This shim closes that gap: every bypass route ultimately performs
# `execve("git", ...)` (or one of git's subcommand symlinks like
# /usr/lib/git-core/git-push).  All those resolve to /usr/bin/git
# via PATH lookup or via the symlinks pointing at it, so every
# realised push attempt enters this script.
#
# Two checks:
# 1. argv[0] ends in -push, -send-pack, or -receive-pack — catches
#    direct invocation of the subcommand binary or its symlinks
#    (/usr/lib/git-core/git-push origin main, etc.).
# 2. argv[1] is push or send-pack — the standard `git push` form.
#
# Any other invocation exec's the real binary unchanged.  The wrapper
# is intentionally simple — fewer lines mean fewer edge cases for the
# next reviewer to audit.
#
# Design doc: docs/2026-05-07-per-thread-web-git-isolation.org
# (see "Dynamic-construction attack surface").

case "${0##*/}" in
  git-push|git-send-pack|git-receive-pack)
    echo "Error: $0 is disabled in this sandbox.  Pushes are user-initiated from the web UI." >&2
    exit 1
    ;;
esac

case "$1" in
  push|send-pack)
    echo "Error: 'git $1' is disabled in this sandbox.  Pushes are user-initiated from the web UI." >&2
    exit 1
    ;;
esac

# `exec -a git` so the real binary sees argv[0] = "git", not
# "/usr/bin/git-real".  Without this, git inspects argv[0], strips
# the "git-" prefix, and tries to dispatch the rest ("real") as a
# subcommand — failing with "fatal: cannot handle real as a builtin"
# on every passthrough call.
exec -a git /usr/bin/git-real "$@"
