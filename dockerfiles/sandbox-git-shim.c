/* Sandbox-wide refusal of git push, regardless of how the call was
 * constructed.  C reimplementation of the bash shim that shipped in
 * PR #97 — required because (a) bash drops setuid privileges on
 * startup so a setuid bash wrapper can't read mode-0700 files, and
 * (b) without the privilege required to read /usr/bin/git-real
 * (locked at mode 0700 root:root by the Dockerfile), even a correct
 * shim can't exec the real binary.  This C wrapper is installed at
 * /usr/bin/git as setuid root (mode 4755).
 *
 * What this layer adds over the bash shim:
 *
 *   The bash shim caught every push form that flowed through `git`
 *   lookup — direct, shell-out, Python subprocess, char-codes,
 *   base64, file-mediated `bash <script>`, plumbing.  But it did
 *   NOT close the case where the agent copied /usr/bin/git-real
 *   itself to a new path and invoked the copy with a spoofed
 *   argv[0] via `exec -a git-receive-pack /tmp/g`.  The new path
 *   never went through the shim, and the receive-pack-name match
 *   only checked the literal symlinks under /usr/lib/git-core.
 *
 *   This layer closes that bypass by making /usr/bin/git-real
 *   unreadable to the agent (mode 0700, container runs as a
 *   non-root uid).  The agent literally cannot copy the binary it
 *   would need to bypass the shim.  The setuid wrapper is the only
 *   process with read permission, and the wrapper does the argv
 *   check before exec.
 *
 * Two checks, identical to the bash version:
 *
 *   1. argv[0]'s basename is git-push, git-send-pack, or
 *      git-receive-pack — catches direct invocation of the
 *      subcommand binary or its symlinks.
 *   2. argv[1] is push or send-pack — catches `git push`.
 *
 * Anything else exec's /usr/bin/git-real with argv[0] forced to
 * "git" so git's own argv[0]-based subcommand dispatch keeps
 * working.
 *
 * Design doc: docs/2026-05-08-restrict-git-real-via-non-root-sandbox.org
 *
 * Why C, not bash:
 *   - Bash drops privileges (and any file capabilities) on startup
 *     for security — a bash shim would execute with no special
 *     access and couldn't exec /usr/bin/git-real (mode 0700).
 *   - A C binary preserves file capabilities through invocation
 *     until it itself execs another binary (which then takes its
 *     own file caps, which git-real doesn't have).
 *
 * Why CAP_DAC_OVERRIDE, not setuid:
 *   The Dockerfile sets file capability cap_dac_override=ep on this
 *   binary.  When invoked, the shim runs with effective uid =
 *   caller (no setuid), but with CAP_DAC_OVERRIDE in its effective
 *   capability set.  That capability lets the kernel's exec
 *   permission check pass for /usr/bin/git-real (mode 0700 root),
 *   even though the caller's uid has no exec bit on the file.
 *
 *   After execv into git-real, the new process loads git-real's
 *   file capabilities (none set), so git-real runs with NO
 *   capabilities and effective uid = caller — meaning files it
 *   creates in /workspace are owned by the caller (uid 1000),
 *   exactly what we need for the host process to manage the
 *   workspace without alpine-rmtree dances.
 *
 *   Why not setuid root + fexecve via pre-opened fd?
 *   Tried.  Linux's execveat (which fexecve uses) re-checks file
 *   permissions against the *current effective uid* at exec time —
 *   not against the open-time uid.  After setuid(getuid()) drops
 *   to uid 1000, fexecve fails with EACCES.  Capabilities don't
 *   have this problem: the kernel's exec check honors
 *   CAP_DAC_OVERRIDE in the current effective set.
 *
 *   Why not setgid + special group?
 *   Hits the same re-check after setgid(getgid()) drop, and
 *   leaves the caller-side files with a confusing gid if we
 *   don't drop.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


static void refuse(const char *what) {
    fprintf(stderr,
            "Error: %s is disabled in this sandbox.  "
            "Pushes are user-initiated from the web UI.\n",
            what);
}


int main(int argc, char *argv[]) {
    /* Hostile-invocation guard: Linux's exec syscalls accept argv ==
     * NULL or argv[0] == NULL (e.g. Python's os.execve("/path", [],
     * {})).  strrchr(NULL, ...) is undefined behaviour; refuse
     * fail-closed before any argv access.
     */
    if (argc < 1 || argv[0] == NULL) {
        refuse("(no argv)");
        return 1;
    }

    /* Find argv[0]'s basename — the part after the last '/'. */
    const char *base = strrchr(argv[0], '/');
    base = base ? base + 1 : argv[0];

    /* Catch direct calls to the subcommand binaries (e.g. someone
     * runs /usr/lib/git-core/git-push origin main directly, where
     * the symlink resolves to this shim).
     */
    if (!strcmp(base, "git-push") ||
        !strcmp(base, "git-send-pack") ||
        !strcmp(base, "git-receive-pack")) {
        refuse(argv[0]);
        return 1;
    }

    /* Catch `git push` and `git send-pack`. */
    if (argc >= 2 &&
        (!strcmp(argv[1], "push") ||
         !strcmp(argv[1], "send-pack"))) {
        char what[64];
        snprintf(what, sizeof(what), "'git %s'", argv[1]);
        refuse(what);
        return 1;
    }

    /* Pass through.  Override argv[0] so git's own dispatch sees
     * "git", not the shim's argv[0] (which could be /usr/bin/git
     * or whatever symlink resolved here).  The kernel's exec
     * permission check on /usr/bin/git-real (mode 0700 root) is
     * satisfied by CAP_DAC_OVERRIDE in our effective set; after
     * exec, the new process inherits git-real's (empty) file caps
     * and runs unprivileged as the caller.
     */
    argv[0] = (char *)"git";
    execv("/usr/bin/git-real", argv);

    /* execv returns only on failure. */
    perror("git-shim: execv /usr/bin/git-real");
    return 127;
}
