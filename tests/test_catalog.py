"""Thread catalog — the read-only list-with-metadata surface both web and voice
consume (assist/catalog.py; docs/2026-07-21-voice-call-tech-design.org §3, P0.5)."""
import json
import os

from assist.catalog import CatalogEntry, ThreadCatalog
from assist.thread_manager import ThreadManager


def _mk(root, tid, *, description=None, stage=None, urgent=False, deleted=False):
    d = os.path.join(root, tid)
    os.makedirs(d, exist_ok=True)
    if description is not None:
        with open(os.path.join(d, "description.txt"), "w") as f:
            f.write(description)
    if stage is not None:
        with open(os.path.join(d, "status.json"), "w") as f:
            json.dump({"stage": stage}, f)
    if urgent:
        open(os.path.join(d, "urgent_response"), "w").close()
    if deleted:
        open(os.path.join(d, ".deleted"), "w").close()
    return d


def _catalog(tmp_path):
    return ThreadCatalog(ThreadManager(str(tmp_path)))


def test_entry_reads_all_sidecars(tmp_path):
    _mk(tmp_path, "t1", description="trip planning", stage="processing", urgent=True)
    cat = _catalog(tmp_path)
    (e,) = cat.entries()
    assert isinstance(e, CatalogEntry)
    assert (e.id, e.description, e.status, e.urgent) == \
        ("t1", "trip planning", "processing", True)
    assert e.updated_at > 0


def test_missing_sidecars_default(tmp_path):
    _mk(tmp_path, "t1")   # no description / status / urgent
    (e,) = _catalog(tmp_path).entries()
    assert e.description == "New thread"   # never generated (needs a model)
    assert e.status == "ready"            # _get_status's default
    assert e.urgent is False


def test_multiline_description_uses_first_nonempty_line(tmp_path):
    # A one-line summary is the contract — an embedded newline must not leak (it would
    # break the numbered list / spoken reply). Take the first non-empty line.
    d = _mk(tmp_path, "t1", stage="ready")
    with open(os.path.join(d, "description.txt"), "w") as f:
        f.write("\n  trip planning to Portugal  \nand a stray second line\n")
    (e,) = _catalog(tmp_path).entries()
    assert e.description == "trip planning to Portugal"


def test_corrupt_status_degrades_not_raises(tmp_path):
    d = _mk(tmp_path, "t1", description="x")
    with open(os.path.join(d, "status.json"), "w") as f:
        f.write("{corrupt")
    (e,) = _catalog(tmp_path).entries()
    assert e.status == "ready"            # corrupt ⇒ default, no exception


def test_nonstring_status_stage_degrades_to_ready(tmp_path):
    # Valid JSON but a non-string stage (null/number) is corrupt too — degrade to
    # the default, don't surface "None"/"123" to a client.
    for bad in (None, 123, ["processing"]):
        d = _mk(tmp_path, "t1", description="x")
        with open(os.path.join(d, "status.json"), "w") as f:
            json.dump({"stage": bad}, f)
        (e,) = _catalog(tmp_path).entries()
        assert e.status == "ready", f"stage={bad!r} should degrade"


def test_soft_deleted_and_pycache_excluded(tmp_path):
    _mk(tmp_path, "live", description="a")
    _mk(tmp_path, "gone", description="b", deleted=True)
    os.makedirs(tmp_path / "__pycache__", exist_ok=True)
    ids = [e.id for e in _catalog(tmp_path).entries()]
    assert ids == ["live"]


def test_entries_sorted_mtime_desc(tmp_path):
    import time
    _mk(tmp_path, "old", description="o")
    time.sleep(0.01)
    _mk(tmp_path, "new", description="n")
    ids = [e.id for e in _catalog(tmp_path).entries()]
    assert ids == ["new", "old"]          # ThreadManager.list order preserved


def test_get_unknown_or_bad_tid_is_none(tmp_path):
    cat = _catalog(tmp_path)
    assert cat.get("does-not-exist") is None
    assert cat.get("../escape") is None   # invalid tid shape ⇒ None, not a raise


def test_no_thread_construction_no_content_access(tmp_path):
    # The catalog must never touch the checkpoint / message store — only the sidecar
    # files. Plant a poison threads.db at the REAL ThreadManager location
    # (<root>/threads.db, thread_manager.py) and confirm entries come solely from the
    # sidecars, with none of the db's content leaking into any field and the db file
    # itself not surfacing as a thread.
    _mk(tmp_path, "t1", description="desc", stage="ready")
    with open(tmp_path / "threads.db", "w") as f:
        f.write("SECRET_TRANSCRIPT should never surface")
    entries = _catalog(tmp_path).entries()
    assert {e.id for e in entries} == {"t1"}         # the db file is not a thread
    (e,) = entries
    assert e.description == "desc"                    # from the sidecar, not the db
    assert all("SECRET" not in (x.description + x.status) for x in entries)
