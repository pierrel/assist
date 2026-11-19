"""Natural call/SMS routing-rule creation coverage using a synthetic store.

This checks kind, number scope and saved-template presence. It does not execute
a call event or validate the later reminder turn.
"""
from pathlib import Path
import tempfile

from assist.agent import AgentHarness, create_agent
from assist.model_manager import select_assistant_model
from assist.events.store import SubscriptionStore
from assist.events.tools import subscription_tools
from .utils import prompt_rewrite_web_main_spec, stub_research_subagent


def test_natural_request_creates_call_rule_separate_from_sms():
    with tempfile.TemporaryDirectory(prefix='call_subscription_eval_') as directory:
        root = Path(directory)
        (root / 'store' / 'synthetic-call-thread').mkdir(parents=True)
        store = SubscriptionStore(str(root / 'store'))
        with stub_research_subagent():
            harness = AgentHarness(create_agent(select_assistant_model(0.1), str(root / 'workspace'),
                spec=prompt_rewrite_web_main_spec(tools=subscription_tools(store))),
                thread_id='synthetic-call-thread')
            harness.message('Whenever +15555550100 calls, note it in this conversation and remind me to ring them back.')
        saved = store.for_thread('synthetic-call-thread')
        assert len(saved) == 1
        assert saved[0].event_kind == 'call'
        assert saved[0].matches('+15555550100')
        assert not saved[0].matches('+15555550101')
        assert saved[0].template.strip()
        assert store.route('+15555550100', 'sms') is None
