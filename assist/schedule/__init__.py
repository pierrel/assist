"""Agent-driven thread schedules ("thread-based cron").

A *wakeup* is ``(thread_id, prompt)`` dispatched through the existing message run
path; the cron scheduler is one producer of timed wakeups (future event-triggers
become another). See ``docs/2026-07-01-thread-schedules-design.org``.

This package holds the reusable CORE (model, cadence math, store, scheduler, tools).
The web layer (``manage/web``) is the only v1 client: it instantiates the store on
``MANAGER.root_dir``, starts the ``Scheduler`` in its lifespan, wires the schedule
tools into its ``AgentSpec``, and serves the management view. emacsos can adopt the
same core later by starting its own ``Scheduler``.
"""
