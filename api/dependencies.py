import logging
from functools import lru_cache

from agentops.agents import HybridCritic, LLMAnalyst, LLMFinalizer, LLMPlanner, LLMResearcher
from agentops.config import get_settings
from agentops.errors import ConfigError
from agentops.llm import GeminiProvider, GroqProvider, LLMRouter
from agentops.orchestrator import EventBus, Orchestrator
from agentops.runs import InMemoryRunRepository, RunManager, RunRepository, SQLiteRunRepository
from agentops.search import InMemorySearchCache, SearchRouter, SerperProvider, TavilyProvider


def _build_llm_router() -> LLMRouter:
    settings = get_settings()
    providers = []
    if settings.gemini_api_key and settings.gemini_model:
        key = settings.gemini_api_key.get_secret_value()
        providers.append(GeminiProvider(key, settings.gemini_model))
        extras = [m.strip() for m in settings.gemini_extra_models.split(",") if m.strip()]
        for model in dict.fromkeys(extras):  # de-duplicated, order kept
            if model != settings.gemini_model:
                providers.append(GeminiProvider(key, model, name=f"gemini:{model}"))
    if settings.groq_api_key and settings.groq_model:
        providers.append(
            GroqProvider(settings.groq_api_key.get_secret_value(), settings.groq_model)
        )
    if not providers:
        raise ConfigError(
            "No LLM provider configured: set GEMINI_API_KEY+GEMINI_MODEL and/or "
            "GROQ_API_KEY+GROQ_MODEL in .env"
        )
    # Free tiers rate-limit hard: skip cooling providers and wait (<=30s) when all are.
    return LLMRouter(providers, max_wait_s=30.0)


def _build_search_router() -> SearchRouter:
    settings = get_settings()
    providers = []
    if settings.tavily_api_key:
        providers.append(TavilyProvider(settings.tavily_api_key.get_secret_value()))
    if settings.serper_api_key:
        providers.append(SerperProvider(settings.serper_api_key.get_secret_value()))
    if not providers:
        raise ConfigError(
            "No search provider configured: set TAVILY_API_KEY and/or SERPER_API_KEY in .env"
        )
    return SearchRouter(providers)


def _orchestrator_factory(bus: EventBus) -> Orchestrator:
    """Built fresh per run (RunManager calls this inside start_run()), not once at
    startup -- so a missing key only fails the run that actually needs it, and never
    blocks /health, GET /runs or GET /runs/{id} from working."""
    llm = _build_llm_router()
    search = _build_search_router()
    cache = InMemorySearchCache()
    return Orchestrator(
        planner=LLMPlanner(llm),
        researcher=LLMResearcher(llm, search, cache=cache),
        analyst=LLMAnalyst(llm),
        critic=HybridCritic(llm),
        finalizer=LLMFinalizer(llm),
        bus=bus,
    )


logger = logging.getLogger(__name__)


def _build_repository() -> RunRepository:
    """SQLite when RUNS_DB_PATH is set; in-memory otherwise or if the file is unusable.

    A broken database must degrade the app (runs not kept across restarts), not take it down.
    """
    path = get_settings().runs_db_path
    if not path:
        return InMemoryRunRepository()
    try:
        repo = SQLiteRunRepository(path)
        repo.interrupt_orphans()
    except Exception:
        logger.exception("could not open run database %r; falling back to in-memory runs", path)
        return InMemoryRunRepository()
    return repo


@lru_cache
def get_run_manager() -> RunManager:
    return RunManager(_orchestrator_factory, _build_repository())


# Public name for scripts (e.g. the evaluation runner) that need the same wiring as the API.
build_orchestrator = _orchestrator_factory
