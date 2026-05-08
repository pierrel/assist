#!/bin/bash
# Build-time smoke for the sandbox git-push-refusal layers.
# Wired into `make sandbox-smoke`; run after `make sandbox-build`
# and before any deploy that uses the new image.  Catches:
#
#   - Wrong file mode on /usr/bin/git-real (must be 0700 root:root)
#   - Missing CAP_DAC_OVERRIDE on /usr/bin/git
#   - Container running as root instead of uid 1000
#   - Any push variant succeeding (would advance origin/main)
#   - Privilege-drop regression (git creating root-owned files)
#
# Usage: bash dockerfiles/test-sandbox-shim.sh
# Exits 0 on full pass, 1 on any failure.

set -e

HOST_DIR=$(mktemp -d)
trap 'rm -rf "$HOST_DIR"' EXIT

OUTPUT=$(docker run --rm -v "$HOST_DIR":/workspace --user 1000:1000 assist-sandbox bash -c '
set +e
TMPDIR=$(mktemp -d); cd "$TMPDIR"
git init --bare -b main remote.git > /dev/null 2>&1
git init -b main work > /dev/null 2>&1
cd work
git config user.email a@b; git config user.name A
git remote add origin ../remote.git
echo hi > file && git add file && git commit -m initial > /dev/null

# Identity check.
[ "$(id -u)" = "1000" ] || { echo "FAIL: container not running as uid 1000 (got $(id -u))"; exit 2; }

# Capability + mode checks.
[ -u /usr/bin/git ] && { echo "FAIL: /usr/bin/git has setuid bit (should use file caps instead)"; exit 2; }
getcap /usr/bin/git | grep -q "cap_dac_override=ep" || { echo "FAIL: /usr/bin/git missing cap_dac_override=ep"; exit 2; }

stat_mode=$(stat -c "%a %U %G" /usr/bin/git-real)
[ "$stat_mode" = "700 root root" ] || { echo "FAIL: /usr/bin/git-real wrong mode/owner: $stat_mode"; exit 2; }

# Each push variant should fail to advance origin/main.
attempt() { "$@" > /dev/null 2>&1; }

attempt git push origin main
attempt git push -f origin main
attempt bash -c "git push origin main"
attempt sh -c "git push origin main"
attempt eval "git push origin main"
attempt python3 -c "import subprocess; subprocess.run([\"git\",\"push\",\"origin\",\"main\"])"
attempt python3 -c "import os; os.system(\"git push origin main\")"
attempt python3 -c "import subprocess; cmd=chr(103)+chr(105)+chr(116); subprocess.run([cmd,\"push\",\"origin\",\"main\"])"
attempt python3 -c "exec(__import__(\"base64\").b64decode(\"aW1wb3J0IHN1YnByb2Nlc3M7IHN1YnByb2Nlc3MucnVuKFsiZ2l0IiwicHVzaCIsIm9yaWdpbiIsIm1haW4iXSk=\").decode())"
attempt /usr/lib/git-core/git-push origin main
attempt /usr/lib/git-core/git-send-pack ../remote.git main
attempt /usr/bin/git push origin main
attempt bash -c "echo \"git push origin main\" > /tmp/x.sh && chmod +x /tmp/x.sh && bash /tmp/x.sh"
attempt bash -c "g=git; p=push; \$g \$p origin main"
attempt perl -e "system(\"git\",\"push\",\"origin\",\"main\")"

# New variants — the cp-then-exec-a bypass that motivated the cap layer.
cp /usr/bin/git-real /tmp/g 2>/dev/null
chmod +x /tmp/g 2>/dev/null
attempt /tmp/g push origin main

cat > /tmp/recv-wrap <<WRAP 2>/dev/null
#!/bin/bash
exec -a git-receive-pack /tmp/g "\$@"
WRAP
chmod +x /tmp/recv-wrap 2>/dev/null
attempt /tmp/g push --receive-pack=/tmp/recv-wrap origin main

# Ground truth: did any push leak through?
if git -C ../remote.git rev-parse --verify main > /dev/null 2>&1; then
  echo "FAIL: origin/main advanced — a push leaked through"
  exit 1
fi

# Privilege-drop check: files git creates must be owned by uid 1000.
git_file_owner=$(stat -c "%U" .git/HEAD)
[ "$git_file_owner" = "sandbox" ] || { echo "FAIL: .git/HEAD owned by $git_file_owner not sandbox — privilege drop is not working, git-real running as root"; exit 1; }

# git-real must be unreadable to the agent (the bypass premise).
cp /usr/bin/git-real /tmp/g-direct 2>/dev/null && { echo "FAIL: agent could copy git-real — mode 0700 not enforced"; exit 1; }
cat /usr/bin/git-real > /dev/null 2>&1 && { echo "FAIL: agent could read git-real — mode 0700 not enforced"; exit 1; }

# Sanity: non-push git ops still work.
git status > /dev/null 2>&1 || { echo "FAIL: git status broken"; exit 1; }
git diff > /dev/null 2>&1 || { echo "FAIL: git diff broken"; exit 1; }
git log --oneline > /dev/null 2>&1 || { echo "FAIL: git log broken"; exit 1; }
git branch -a > /dev/null 2>&1 || { echo "FAIL: git branch broken"; exit 1; }

echo "PASS"
' 2>&1)

EXIT=$?
echo "$OUTPUT"
exit $EXIT
