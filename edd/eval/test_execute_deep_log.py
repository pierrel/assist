"""Eval: can the dev agent reach a value buried in a large EXECUTE log (by grepping
the offloaded file)?  This is the per-tool gate for adding `execute` to
ToolResultToFileMiddleware — grep-adoption is PROVEN for read_url, UNPROVEN for
execute (a command log is a different interaction than a web page).

A command log's salient line (the error / exit summary) is at the TAIL, so the
execute path uses a head_tail preview.  Here the log ends with "BUILD FAILED"
(shown in the tail preview → orientation), but the UNGUESSABLE error code sits
~5k chars before the end — outside the 300-char tail preview — so the model must
GREP the offloaded file to report it.

Mocked: the sandbox's `execute` is patched to return a canned large log for the
`make build` command only — rate-limit-free, deterministic, no build-time
dependency.  Every OTHER command (crucially the offload write's preflight
`os.makedirs`/write-check, and the model's later grep) delegates to the REAL
sandbox, so the `/tmp` offload actually lands and the grep can find it — otherwise
the mock would swallow the offload write and the eval would test nothing.
"""
import os
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import patch

from assist.agent import AgentHarness, create_agent
from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager
from assist.sandbox import ExecuteResponse
from edd.eval.utils import create_filesystem

os.environ.setdefault("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")

_ERROR_CODE = "ZX-99741"   # unguessable, synthetic — the model can't know it
# The decisive line is ~5k chars before the end (past the tail preview); the very
# end shows the summary the tail preview surfaces for orientation.
_LOG = (
    ("[compile] module auth_handler ... ok\n" * 1400)
    + f"[compile] module payment_gateway ... FAILED\n"
      f"ERROR: compilation aborted — error code {_ERROR_CODE} in module payment_gateway "
      f"(unresolved symbol 'settleBatch').\n"
    + ("[compile] module reporting ... ok\n" * 140)
    + "\n==============================\nBUILD FAILED (1 error)\n"
)


def _make_mock_execute(real_execute):
    """Canned log for `make build`; delegate everything else (offload preflight,
    grep) to the real sandbox so the /tmp offload actually lands."""
    def _mock_execute(command, *a, **k):
        if "make build" in command:
            return ExecuteResponse(output=_LOG, exit_code=1)
        return real_execute(command, *a, **k)
    return _mock_execute


class TestExecuteDeepLog(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="execute_deep_log_")
        create_filesystem(self.workspace, {"README.org": "Dev workspace."})
        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest("Docker sandbox unavailable")
        self.addCleanup(lambda: SandboxManager.cleanup(self.workspace))
        self.addCleanup(lambda: shutil.rmtree(self.workspace, ignore_errors=True))

    def test_agent_reaches_error_code_by_grepping_the_log(self):
        mock_execute = _make_mock_execute(self.sandbox.execute)   # bound real method, pre-patch
        with patch.object(type(self.sandbox), "execute", staticmethod(mock_execute)):
            agent = AgentHarness(create_agent(self.model, self.workspace,
                                              sandbox_backend=self.sandbox))
            out = agent.message(
                "Run `make build` and tell me the EXACT error code it reports "
                "for the failing module.")
        ans = str(out)
        self.assertIn(_ERROR_CODE, ans,
                      f"agent did not report the error code {_ERROR_CODE} — it's ~5k chars "
                      f"deep in the offloaded log, so it needs to grep the file. Answer:\n{ans[:600]}")
