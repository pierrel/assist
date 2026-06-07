---
name: elisp
description: Guidance for writing Emacs Lisp. TRIGGER WORDS — elisp, emacs lisp, `.el`, defun, defvar, defcustom, defmacro, byte-compile, checkdoc, ert, `emacs --batch`. MUST load before any tool call that writes, edits, lints, tests, or runs Emacs Lisp (`.el`) code.
---

# Emacs Lisp guide

Rules for writing, linting, and testing Emacs Lisp that runs reliably —
especially headless (`emacs --batch`), which is how an agent invokes it.

## Structure

- Lay the file out with the conventional section markers `checkdoc` expects:
  - header: `;;; name.el --- one-line summary  -*- lexical-binding: t; -*-`
  - `;;; Commentary:` (what and why) then `;;; Code:` (before the code)
  - footer: `(provide 'name)` then `;;; name.el ends here`
- Prefer pure functions that take arguments and return values over commands
  that act on the current buffer — pure logic is testable without a UI.
- Namespace everything: public names `prefix-thing`, internal helpers
  `prefix--thing`. A `defcustom` for user-facing settings, `defvar`/`defconst`
  otherwise.
- Resolve paths with `expand-file-name` against a root you computed (e.g. from
  the file's own `load-file-name`) — never assume the process's working
  directory. Note: relative `org-agenda-files` expand against `org-directory`
  (`~/org`), NOT the cwd, which is a common silent breakage.

## Run it headless — and never let it hang

- Invoke as `emacs --batch -Q -l file.el --eval '(...)'`. `-Q` skips all
  personal init, so the result is reproducible anywhere; declare everything
  the script needs inside the file.
- `--batch` must never block on input. A function that would prompt
  (`y-or-n-p`, `completing-read`, a missing-file `[R]emove/[A]bort`) will hang
  the process forever. Set the variable that suppresses the prompt instead
  (e.g. `org-agenda-skip-unavailable-files t`).
- Output: `princ` writes to stdout, `message` to stderr. Use `princ` for
  anything the caller needs to read back.

## Lint (treat warnings as errors)

- Byte-compile and read the warnings:
  `emacs --batch -Q -f batch-byte-compile file.el`
- Check docstrings and conventions:
  `emacs --batch -Q --eval '(checkdoc-file "file.el")'`

## Test

- Write tests with ERT — `(ert-deftest prefix-test-foo () (should (equal ...)))`
  — in a sibling `name-tests.el`.
- Run them headless, exiting non-zero on failure:
  `emacs --batch -Q -l file.el -l name-tests.el -f ert-run-tests-batch-and-exit`

## Look things up without leaving the shell

- A function's contract is its docstring:
  `emacs --batch -Q --eval '(princ (documentation (quote mapcar)))'`
- Find names with `apropos`; the authoritative reference is the GNU Emacs Lisp
  Reference Manual (the `(elisp)` Info manual, or gnu.org online).

## Always verify

Emacs Lisp is easy to get subtly wrong. After writing or editing a `.el` file,
byte-compile it (warnings clean) and run its ERT tests before reporting done —
do not rely on reading the code alone.
