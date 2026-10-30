"""Natural outcome probes with a shortened exploration budget.

The fixture controls the budget, never the user's request. The divergent ledger
has useful facts but never ends; the convergent ledger ends after two records.
No external retrieval or side effects occur. Baseline divergence separately
checks the previous terminal-error behavior without the warning.
"""
import json
import os

import pytest
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from assist.middleware.execution_limit import ExecutionLimitMiddleware
from assist.model_manager import select_assistant_model
from assist.thread import Thread


@pytest.mark.parametrize("case", ["divergent", "convergent", "ordinary", "baseline"])
def test_execution_limit_outcome(case, tmp_path):
    calls = []

    @tool
    def ledger_records(cursor: int = 0) -> str:
        """Read ledger records, starting at cursor. Follow next_cursor until null."""
        calls.append(cursor)
        record = {"invoice": "R-12", "payment_received": 80,
                  "invoice_total": None, "disputed": True}
        if case == "convergent" and cursor >= 1:
            record.update(invoice_total=100, disputed=False)
        return json.dumps({"record": record,
                           "next_cursor": None if case == "convergent" and cursor >= 1
                           else cursor + 1})

    model = select_assistant_model(0.1)
    graph = create_agent(
        model, tools=[ledger_records], checkpointer=InMemorySaver(),
        system_prompt=("Help the user with the ledger. Read records until next_cursor "
                       "is null before reconciling. Never invent missing amounts or "
                       "claim an invoice is closed without evidence."),
        middleware=[] if case == "baseline" else [ExecutionLimitMiddleware()],
    )
    chat = Thread(str(tmp_path), agent=graph, model=model)
    # Capability fixture: retain the production 50-step reserve while shortening
    # exploration so the behavioral probe fits a bounded local-model invocation.
    chat.runconfig["recursion_limit"] = 15 if case == "baseline" else 65
    prompt = ("What is 19 plus 23?" if case == "ordinary" else
              "Reconcile invoice R-12 with the ledger and tell me whether we can close it.")
    if case == "baseline":
        with pytest.raises(GraphRecursionError):
            chat.message(prompt)
        print(json.dumps({"case": case, "model": os.path.basename(model.model_name),
                          "calls": calls, "result": "GraphRecursionError"}))
        return
    answer = chat.message(prompt)
    state = graph.get_state(chat.runconfig)
    print(json.dumps({"case": case, "model": os.path.basename(model.model_name), "calls": calls,
                      "warned": state.values.get("execution_limit_warned"),
                      "answer": answer}))
    assert answer.strip()
    low = answer.lower()
    if case == "ordinary":
        assert "42" in answer
        assert not calls
    elif case == "convergent":
        assert "20" in answer
        assert len(calls) >= 2
    else:
        assert calls
        assert state.values["execution_limit_warned"]
        assert any(word in low for word in ("missing", "unknown", "unable", "cannot", "incomplete", "not available"))
        assert "80" in answer
        assert "closed successfully" not in low
