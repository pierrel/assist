"""Tests for the `extra_skill_sources` parameter on `create_agent`
and the underlying `extra_routes` parameter on the backend factories.

Embedders (notably emacsos-server) inject additional virtual-path
routes that hold skill files outside the assist repo.  The contract:

  1. Extra routes register with the composite backend (so reads from
     those paths resolve).
  2. The extra route prefixes are added to `SmallModelSkillsMiddleware`'s
     `sources` list (so the middleware actually lists those skills).
  3. Defaults preserve the pre-2026-05-17 behavior: only `SKILLS_ROUTE`
     contributes skills.
"""

import tempfile
from unittest.mock import patch, MagicMock

from assist.backends import (
    SKILLS_ROUTE,
    STATEFUL_PATHS,
    create_composite_backend,
    create_sandbox_composite_backend,
)
from deepagents.backends import FilesystemBackend


def _route_backend():
    """A FilesystemBackend rooted at a fresh tempdir."""
    return FilesystemBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)


class TestBackendFactoryExtraRoutes:
    def test_create_composite_backend_default_routes_unchanged(self):
        # Default stateful_paths=[]; only SKILLS_ROUTE is registered.
        cb = create_composite_backend()
        assert SKILLS_ROUTE in cb.routes
        # With STATEFUL_PATHS passed, those routes appear too.
        cb_full = create_composite_backend(stateful_paths=STATEFUL_PATHS)
        for p in STATEFUL_PATHS:
            assert p in cb_full.routes
        assert SKILLS_ROUTE in cb_full.routes

    def test_create_composite_backend_extra_routes_merge(self):
        extra = _route_backend()
        cb = create_composite_backend(extra_routes={"/extra/": extra})
        assert "/extra/" in cb.routes
        assert cb.routes["/extra/"] is extra
        # Defaults still present.
        assert SKILLS_ROUTE in cb.routes

    def test_create_composite_backend_extra_routes_can_override_default(self):
        replacement = _route_backend()
        cb = create_composite_backend(
            extra_routes={SKILLS_ROUTE: replacement}
        )
        assert cb.routes[SKILLS_ROUTE] is replacement

    def test_create_sandbox_composite_backend_extra_routes(self):
        sandbox = MagicMock()
        extra = _route_backend()
        cb = create_sandbox_composite_backend(
            sandbox, extra_routes={"/emacsos-skills/": extra}
        )
        assert "/emacsos-skills/" in cb.routes
        assert cb.routes["/emacsos-skills/"] is extra
        assert SKILLS_ROUTE in cb.routes  # default preserved


class TestCreateAgentExtraSkillSources:
    """`create_agent` is heavy (constructs sub-agents, model probes, etc.).
    Patch `create_deep_agent` to a no-op and just verify the wiring.
    """

    def _build(self, **kwargs):
        from assist.agent import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        with patch("assist.agent.create_deep_agent") as fake, \
             patch("assist.agent.create_context_agent") as fake_ctx, \
             patch("assist.agent.create_research_agent") as fake_res:
            fake.return_value = MagicMock()
            fake_ctx.return_value = MagicMock()
            fake_res.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                model = MagicMock()
                # InMemorySaver avoids any sqlite file creation.
                create_agent(
                    model, wd, checkpointer=InMemorySaver(),
                    sandbox_backend=None, **kwargs,
                )
                return fake.call_args.kwargs

    def test_default_skill_sources_only_skills_route(self):
        kwargs = self._build()
        backend = kwargs["backend"]
        # Find the SmallModelSkillsMiddleware in the middleware list.
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware
        skills_mws = [m for m in kwargs["middleware"]
                      if isinstance(m, SmallModelSkillsMiddleware)]
        assert len(skills_mws) == 1
        # The middleware was passed sources=[SKILLS_ROUTE] only.
        # We can't easily introspect the middleware's stored sources without
        # poking at its private state; instead verify the backend has
        # exactly the default routes and no extras.
        assert SKILLS_ROUTE in backend.routes
        # No unexpected extra routes (everything in backend.routes is either
        # SKILLS_ROUTE or a STATEFUL_PATHS entry).
        for path in backend.routes:
            assert path == SKILLS_ROUTE or path in STATEFUL_PATHS, (
                f"unexpected route {path!r} in default-construction backend"
            )

    def test_extra_skill_sources_added_to_backend_routes(self):
        extra = _route_backend()
        kwargs = self._build(extra_skill_sources={"/emacsos-skills/": extra})
        backend = kwargs["backend"]
        assert "/emacsos-skills/" in backend.routes
        assert backend.routes["/emacsos-skills/"] is extra
        # Default skills route still present.
        assert SKILLS_ROUTE in backend.routes

    def test_extra_skill_sources_added_to_middleware_sources(self):
        """The middleware's `sources` list must include the extra paths so
        it actually lists skills from them; the backend routing alone
        isn't enough — `SkillsMiddleware` only looks at paths in `sources`.
        """
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware

        extra = _route_backend()
        kwargs = self._build(extra_skill_sources={"/emacsos-skills/": extra})
        skills_mws = [m for m in kwargs["middleware"]
                      if isinstance(m, SmallModelSkillsMiddleware)]
        assert len(skills_mws) == 1
        mw = skills_mws[0]
        # Upstream `SkillsMiddleware.__init__` assigns `self.sources`
        # as a stable public attribute.  Pin against that directly; if
        # upstream renames it, this fails loudly rather than silently
        # matching some other list attribute (e.g. `source_labels`).
        assert hasattr(mw, "sources"), (
            "SkillsMiddleware should expose `sources` — upstream may have "
            "renamed it; update this test."
        )
        assert mw.sources[0] == SKILLS_ROUTE, (
            "SKILLS_ROUTE must be first so the built-in skills are listed "
            "before any embedder-supplied sources."
        )
        assert "/emacsos-skills/" in mw.sources

    def test_multiple_extra_skill_sources(self):
        extras = {
            "/emacsos-skills/": _route_backend(),
            "/user-skills/": _route_backend(),
        }
        kwargs = self._build(extra_skill_sources=extras)
        backend = kwargs["backend"]
        for path in extras:
            assert path in backend.routes
        assert SKILLS_ROUTE in backend.routes

    def test_extra_skill_sources_overriding_skills_route_does_not_duplicate(self):
        """If an embedder explicitly passes `SKILLS_ROUTE` as a key in
        `extra_skill_sources` (the documented backend-override
        mechanism), the middleware's `sources` list must NOT contain
        `SKILLS_ROUTE` twice — duplicates would make the middleware
        scan the same prefix twice.  The backend route still gets
        overridden (the route map update wins)."""
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware

        replacement = _route_backend()
        kwargs = self._build(extra_skill_sources={SKILLS_ROUTE: replacement})

        # Backend route is the replacement (override).
        backend = kwargs["backend"]
        assert backend.routes[SKILLS_ROUTE] is replacement

        # Middleware sources list has SKILLS_ROUTE exactly once.
        mw = next(m for m in kwargs["middleware"]
                  if isinstance(m, SmallModelSkillsMiddleware))
        assert mw.sources.count(SKILLS_ROUTE) == 1, (
            f"SKILLS_ROUTE duplicated in middleware sources: {mw.sources}"
        )
