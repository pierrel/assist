#define _GNU_SOURCE
#include <errno.h>
#include <linux/sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
static void fail(const char *message) {
    dprintf(STDERR_FILENO, "launch-in-cgroup: %s: %d\n", message, errno);
    _exit(127);
}
int main(int argc, char **argv) {
    if (argc < 4 || argv[2][0] != '-' || argv[2][1] != '-' || argv[2][2] != '\0') {
        dprintf(STDERR_FILENO, "usage: launch-in-cgroup CGROUP_FD -- COMMAND...\n");
        return 2;
    }
    int cgroup_fd = atoi(argv[1]);
    struct clone_args args = {
        .flags = CLONE_INTO_CGROUP,
        .exit_signal = SIGCHLD,
        .cgroup = (unsigned long long)cgroup_fd,
    };
    pid_t child = syscall(SYS_clone3, &args, sizeof(args));
    if (child < 0) fail("clone3");
    if (child == 0) {
        close(cgroup_fd);
        execvp(argv[3], &argv[3]);
        _exit(127);
    }
    int status;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) return 127;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 127;
}
