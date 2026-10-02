from agentops.contracts import RunState
from agentops.runs import InMemoryRunRepository

OBJECTIVE = "Compare three approaches for an enterprise AI platform"


def test_save_then_get_returns_equal_state():
    repo = InMemoryRunRepository()
    state = RunState(objective=OBJECTIVE)

    repo.save(state)
    fetched = repo.get(state.run_id)

    assert fetched == state


def test_get_missing_run_returns_none():
    repo = InMemoryRunRepository()
    assert repo.get("does-not-exist") is None


def test_save_stores_a_snapshot_not_a_live_reference():
    repo = InMemoryRunRepository()
    state = RunState(objective=OBJECTIVE)
    repo.save(state)

    state.limitations.append("mutated after save")

    stored = repo.get(state.run_id)
    assert stored.limitations == []  # the repository's copy must not see later mutation


def test_list_returns_newest_first_and_respects_limit():
    repo = InMemoryRunRepository()
    states = [RunState(objective=OBJECTIVE) for _ in range(3)]
    for state in states:
        repo.save(state)

    listed = repo.list(limit=2)

    assert [s.run_id for s in listed] == [states[2].run_id, states[1].run_id]
