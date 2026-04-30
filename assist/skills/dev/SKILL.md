---
name: dev
description: Software development work in a code project. TRIGGER WORDS — code, codebase, project, repo, function, class, method, module, test, TDD, refactor, bug, debug, fix, implement, feature, pyproject.toml, package.json, Cargo.toml, go.mod, Makefile, Dockerfile. MUST load before any tool call that explores, explains, writes, edits, or runs code in a software project.
---

# Software Development Workflow

You are doing software development work. Follow this workflow.

## Before anything: identify the task type

Determine which kind of task you have, then jump to the matching section:

- **Code change** (new feature, bug fix, refactor, adding/changing functions, creating tests): full TDD workflow below.
- **Documentation** (READMEs, docstrings, comments, guides): skip TDD; see *Documentation requests* near the bottom.
- **Explanation** (explain code, architecture, how something works): skip TDD; see *Explanation requests* near the bottom.

## TDD workflow (for code-change tasks)

### Step 1: Understand the codebase and verify the environment

Before making ANY changes, complete ALL of the following in order.

**1a. Explore.** Aim for ≤5 direct file reads.
- Call the `context-agent` (via the `task` tool) once with a detailed prompt asking it to discover the project structure, conventions, and files relevant to this task.
- Directly read 1–3 key files central to the task (e.g. the file you'll modify).

**1b. Run at least one existing test — MANDATORY before writing a plan.**
- Use the shell-execution tool (e.g. `execute`) to install dependencies and run the test suite. Example invocations:
  - `execute("pip install -e . 2>&1 | tail -5")`
  - `execute("pytest tests/ -x -q --tb=short 2>&1 | head -80")`
- Include the test output in your response. This confirms the environment works.

**1c. Research (only if needed).**
- If the task involves unfamiliar libraries or patterns, call `research-agent` (via the `task` tool) to research best practices.

**1d. Note findings.**
- Write 3–5 key bullet points to `dev_notes.txt` in the workspace.

**Checkpoint**: before proceeding to Step 2, confirm:
- [ ] Called context-agent at least once.
- [ ] Ran the test suite via the shell-execution tool and showed output in your response.
- [ ] At least one test passes (proves the environment works).

Do NOT proceed to Step 2 until all three boxes are checked.

### Step 2: Write an implementation plan

Before writing any code, write a plan file at `YYYY-MM-DD-[short-summary].md` in the workspace (use today's date from the current date/time).

The plan **must** include these sections:
- **Reason for change** — why this is needed and what problem it solves.
- **Proposed tests** — what tests you'll write, what scenarios they cover (normal, edge, error cases), and which files they go in.
- **Proposed changes** — which files will be modified, what functions/classes will change, and the high-level approach.
- **Expected outcomes** — what should be true after the change (user-visible behavior, test results, performance/correctness properties).
- **Risks / considerations** — edge cases, pitfalls, backward-compat concerns, decisions that need consideration.

Then return to the user: present a summary of the plan, tell them where to find the plan file, and ask them to approve it before you proceed. For example: "I've written the plan at `2026-04-26-feature-name.md`. Please review it and let me know if you'd like to proceed."

**After Step 2, end your response and wait. Do NOT write test files or implementation code yet.** The next steps (writing tests, implementing) only happen AFTER the user responds with approval.

### Step 3: Write tests (TDD — red phase)

Only after the user approves the plan:
1. Write the test file(s) described in the plan, using the file-write tool — always in a separate test file from the implementation.
2. Run the new tests via the shell-execution tool — they **must fail** at this point (this is the "red" phase).
3. If the tests pass before you write the implementation, they aren't testing the right thing — revise them.
4. Show the full failing test output to the user, including exact error messages and assertion failures.
5. Return to the user with the failing output and ask them to approve the tests before you proceed to implementation. Update the plan file if the tests differ meaningfully from what was proposed.

**After Step 3, end your response and wait. Do NOT implement any code yet.** Implementation (Step 4) happens only AFTER the user responds with approval.

### Step 4: Write the implementation (TDD — green phase)

Only after the user approves the failing tests:
1. Implement the minimum code needed to make your new tests pass, using the file-write or file-edit tool.
2. Follow the existing code conventions and patterns you discovered in Step 1.
3. Run ALL tests via the shell-execution tool — both your new tests and the full existing suite.
4. New tests must now pass (green phase); existing tests must still pass.
5. If any test fails, fix the issue and re-run until all tests pass.

### Step 5: Get critique

After your changes are ready:
1. Use the shell-execution tool to get the diff: `execute("git diff main")` (or `git diff master` if main doesn't exist).
2. Call the `critique-agent` (via the `task` tool) with the full diff output for code review.
3. Address any issues the critique raises — fix bugs, improve tests, clean up code.
4. Re-run tests after any changes from critique feedback.

### Step 6: Respond

Summarize what you did:
- What you understood about the codebase (reference specific files).
- What tests you wrote and what they verify.
- What code you changed or created.
- What the critique found and how you addressed it.
- Confirmation that all tests pass.

## Documentation requests

Skip the planning and TDD steps. Just:
1. Use `context-agent` to understand the current documentation structure.
2. Read the existing documentation to understand tone, format, and conventions.
3. Use the file-write or file-edit tool to modify the documentation file directly.
4. ALWAYS modify the actual file — never just describe changes verbally.

## Explanation requests

Skip the planning and TDD steps. Just:
1. Use `context-agent` extensively to explore the project structure.
2. Read all key files: entry points, core modules, configuration, tests.
3. Provide a thorough, structured explanation in your response, referencing specific files and line numbers.
4. Describe: architecture, design patterns, data flow, key abstractions, and how components interact.
5. Optionally write findings to `dev_notes.txt` as a scratchpad.

## Standing rules

- **Always use context-agent** to understand the codebase before making changes.
- For code-change requests: (1) run existing tests, (2) write a plan and ask for approval, (3) ONLY after approval write failing tests and ask for approval, (4) ONLY after that approval implement the code.
- For documentation and explanation requests: skip the planning and TDD steps — just gather context and act directly.
- ALWAYS run tests via the shell-execution tool — never skip running them.
- ALWAYS install dependencies via the shell-execution tool before running code.
- ALWAYS write or edit actual files — never just describe changes verbally.
- ALWAYS get a critique of code changes before finishing.
- Do not write tests or implementation code before the user approves the plan.
- Do not implement code before the user approves the failing tests.
- Write observations and findings to `dev_notes.txt` throughout your work.

## Best practices

- **Test everything**: every code change must have corresponding tests in a separate test file. No exceptions.
- **Small functions**: keep functions focused on a single responsibility.
- **Meaningful names**: variables, functions, classes, and files should have clear, descriptive names.
- **Error handling**: handle edge cases and errors gracefully with clear error messages.
- **Security**: never introduce injection vulnerabilities, hardcoded secrets, or unsafe practices.
- **DRY**: don't repeat yourself — extract shared logic into helpers.
- **Match conventions**: follow the project's existing style for indentation, naming, imports, and file organization.
- **Minimal changes**: only change what is necessary. Don't refactor unrelated code.

## Efficiency

- **Keep exploration lean**: use `context-agent` for broad exploration — avoid reading dozens of files directly.
- Do not retry the same failing command more than twice. Diagnose the root cause and try a different approach.
- Be direct: discover what you need, make the change, verify, done.
- Use `| head -80` or `| tail -30` to limit command output. Use `| grep -n ...` to search instead of reading entire files.
- **NEVER skip the test-run step** in Step 1b — it is mandatory, even for simple tasks.
