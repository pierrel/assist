eval:
	.venv/bin/pytest --junit-xml=edd/history/results-$(date +%Y%m%d-%H%M).xml edd/eval

test:
	.venv/bin/pytest --junit-xml=tests/history/results-$(date +%Y%m%d-%H%M).xml tests
