from agentops.contracts import RunState, RunStatus
from agentops.runs import SQLiteRunRepository

OBJECTIVE = "Compare three approaches for an enterprise AI platform"


def test_run_survives_reopening_the_database(tmp_path):
    path = str(tmp_path / "runs.db")
    state = RunState(objective=OBJECTIVE)
    state.limitations.append("note")
    SQLiteRunRepository(path).save(state)

    loaded = SQLiteRunRepository(path).get(state.run_id)

    assert loaded is not None
    assert loaded.objective == OBJECTIVE
    assert loaded.limitations == ["note"]


def test_save_updates_in_place_and_get_returns_none_for_unknown(tmp_path):
    repo = SQLiteRunRepository(str(tmp_path / "runs.db"))
    state = RunState(objective=OBJECTIVE)
    repo.save(state)
    state.status = RunStatus.COMPLETED
    repo.save(state)

    assert repo.get(state.run_id).status == RunStatus.COMPLETED
    assert len(repo.list()) == 1
    assert repo.get("run_missing") is None


def test_list_is_newest_first_and_respects_limit(tmp_path):
    repo = SQLiteRunRepository(str(tmp_path / "runs.db"))
    states = [RunState(objective=f"{OBJECTIVE} {i}") for i in range(3)]
    for state in states:
        repo.save(state)
    repo.save(states[0])  # re-saving must not move a run to the front

    listed = repo.list(limit=2)

    assert [s.run_id for s in listed] == [states[2].run_id, states[1].run_id]


def test_interrupt_orphans_closes_only_active_runs(tmp_path):
    repo = SQLiteRunRepository(str(tmp_path / "runs.db"))
    running = RunState(objective=OBJECTIVE, status=RunStatus.RUNNING)
    pending = RunState(objective=OBJECTIVE, status=RunStatus.PENDING)
    done = RunState(objective=OBJECTIVE, status=RunStatus.COMPLETED)
    for state in (running, pending, done):
        repo.save(state)

    assert repo.interrupt_orphans() == 2

    assert repo.get(running.run_id).status == RunStatus.INTERRUPTED
    assert repo.get(running.run_id).finished_at is not None
    assert repo.get(running.run_id).limitations
    assert repo.get(pending.run_id).status == RunStatus.INTERRUPTED
    assert repo.get(done.run_id).status == RunStatus.COMPLETED
    assert repo.interrupt_orphans() == 0
