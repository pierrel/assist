"""Small sandbox-side file operations for research report cleanup.

The caller supplies the Python program on each invocation. The agent cannot edit
the operation being executed. Report names are relative to a verified
``references`` directory handle; transfers use generated paths under ``/tmp``.
"""

SANDBOX_FILES = r'''
import hashlib, json, os, stat, sys

op, directory, expected, payload, transfer = sys.argv[1:]
expected = json.loads(expected)
payload = json.loads(payload)

if op == "ensure":
    os.makedirs(directory, exist_ok=True)

root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    root_stat = os.fstat(root)
    identity = [root_stat.st_dev, root_stat.st_ino]
    if expected and identity != expected:
        raise ValueError("references directory changed during research")

    def stamp(name):
        s = os.stat(name, dir_fd=root, follow_symlinks=False)
        return [s.st_mtime_ns, s.st_size, s.st_ino, s.st_mode]

    def regular(name):
        if not name or name in (".", "..") or "/" in name or "\x00" in name:
            raise ValueError("invalid report filename")
        fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=root)
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode):
            os.close(fd)
            raise ValueError("research report must be a regular file")
        return fd, s

    if op == "ensure":
        print(json.dumps(identity))
    elif op == "snapshot":
        names = os.listdir(root)
        if len(names) > 10000:
            raise ValueError("too many research files")
        files = {}
        for name in names:
            try:
                files[name] = stamp(name)
            except FileNotFoundError:
                pass
        data = json.dumps(files).encode()
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("research file listing is too large")
        with open(transfer, "xb") as output:
            output.write(data)
        print("OK")
    elif op == "exists":
        try:
            fd, _ = regular(payload)
        except (FileNotFoundError, ValueError, OSError):
            print("NO")
        else:
            os.close(fd)
            print("YES")
    elif op == "read":
        fd, s = regular(payload)
        prior = [s.st_mtime_ns, s.st_size, s.st_ino, s.st_mode]
        try:
            if s.st_size > 8 * 1024 * 1024:
                raise ValueError("final research report exceeds 8 MiB source-binding limit")
            source = os.fdopen(fd, "rb")
            fd = -1
            with source, open(transfer, "xb") as output:
                data = source.read(8 * 1024 * 1024 + 1)
                if len(data) > 8 * 1024 * 1024:
                    raise ValueError("final research report exceeds 8 MiB source-binding limit")
                output.write(data)
        finally:
            if fd >= 0:
                os.close(fd)
        if stamp(payload) != prior:
            raise ValueError("research report changed during transfer")
        print(json.dumps(prior))
    elif op == "replace":
        name, prior, digest = payload
        fd, old = regular(name)
        os.close(fd)
        if stamp(name) != prior:
            raise ValueError("research report changed during source binding")
        source_fd = os.open(transfer, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        temporary = None
        try:
            source_stat = os.fstat(source_fd)
            if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size > 8 * 1024 * 1024:
                raise ValueError("invalid resolved report transfer")
            temporary = ".source-" + os.urandom(16).hex()
            target_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                old.st_mode & 0o777, dir_fd=root)
            try:
                with os.fdopen(source_fd, "rb") as source, os.fdopen(target_fd, "wb") as target:
                    source_fd = -1
                    data = source.read(8 * 1024 * 1024 + 1)
                    if len(data) > 8 * 1024 * 1024:
                        raise ValueError("invalid resolved report transfer")
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError("resolved report transfer changed")
                    target.write(data)
                    target.flush()
                    os.fchmod(target.fileno(), old.st_mode & 0o777)
                if stamp(name) != prior:
                    raise ValueError("research report changed during source binding")
                os.replace(temporary, name, src_dir_fd=root, dst_dir_fd=root)
                temporary = None
            finally:
                if temporary is not None:
                    try: os.unlink(temporary, dir_fd=root)
                    except FileNotFoundError: pass
        finally:
            if source_fd >= 0:
                os.close(source_fd)
        print("OK")
    elif op == "prune":
        names_fd = os.open(transfer, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(names_fd, "rb") as names_file:
            names_stat = os.fstat(names_file.fileno())
            if not stat.S_ISREG(names_stat.st_mode) or names_stat.st_size > 4 * 1024 * 1024:
                raise ValueError("invalid research cleanup transfer")
            names_data = names_file.read(4 * 1024 * 1024 + 1)
        if hashlib.sha256(names_data).hexdigest() != payload:
            raise ValueError("research cleanup list changed")
        names = json.loads(names_data)
        if not isinstance(names, list) or len(names) > 10000 or any(
            not isinstance(name, str) or not name or name in (".", "..")
            or "/" in name or "\x00" in name for name in names
        ):
            raise ValueError("invalid research cleanup filename")
        failed = 0
        for name in names:
            try:
                mode = os.stat(name, dir_fd=root, follow_symlinks=False).st_mode
                if not stat.S_ISDIR(mode):
                    os.unlink(name, dir_fd=root)
            except FileNotFoundError:
                pass
            except OSError:
                failed += 1
        print("OK" if failed == 0 else f"skipped {failed} research files")
    else:
        raise ValueError("unknown research file operation")
finally:
    os.close(root)
'''
