from assist.deepagents_agent import deepagents_agent

agent = deepagents_agent()

agent.invoke({'messages': [{'role': 'user', 'content': 'What is langgraph?'}]},
             {'configurable': {'thread_id': 1}})
