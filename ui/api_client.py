import httpx

from agentops.contracts import BudgetLimits, Event, RunState


class AgentOpsClient:
    """Thin, synchronous HTTP client for the AgentOps FastAPI backend.

    No Streamlit imports here -- kept plain and testable (master doc: the UI is a
    thin shell). Reuses agentops.contracts models directly since the API returns
    exactly their JSON shape; redefining them here would just be a second copy to
    keep in sync by hand.
    """

    def __init__(
        self, base_url: str, *, client: httpx.Client | None = None, timeout_s: float = 10.0
    ) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def health(self) -> bool:
        try:
            resp = self._client.get("/health")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    def create_run(self, objective: str, limits: BudgetLimits | None = None) -> RunState:
        body: dict = {"objective": objective}
        if limits is not None:
            body["limits"] = limits.model_dump(mode="json")
        resp = self._client.post("/runs", json=body)
        resp.raise_for_status()
        return RunState.model_validate(resp.json())

    def get_run(self, run_id: str) -> RunState | None:
        resp = self._client.get(f"/runs/{run_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return RunState.model_validate(resp.json())

    def list_runs(self, limit: int = 25) -> list[RunState]:
        resp = self._client.get("/runs", params={"limit": limit})
        resp.raise_for_status()
        return [RunState.model_validate(item) for item in resp.json()]

    def get_events(self, run_id: str, after: int = 0) -> list[Event]:
        resp = self._client.get(f"/runs/{run_id}/events", params={"after": after})
        resp.raise_for_status()
        return [Event.model_validate(item) for item in resp.json()]
