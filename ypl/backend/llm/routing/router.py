import asyncio

from google.auth import default
from google.cloud import run_v2
from langchain_core.language_models.chat_models import BaseChatModel

from ypl.backend.config import settings
from ypl.backend.llm.constants import ChatProvider
from ypl.backend.llm.db_helpers import (
    deduce_original_providers,
)
from ypl.backend.llm.judge import PromptModifierLabeler, YuppMultilabelClassifier, YuppOnlinePromptLabeler
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.ranking import Ranker, get_ranker
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.modules.decision import RoutingDecisionLogger
from ypl.backend.llm.routing.modules.filters import (
    Exclude,
    HighErrorRateFilter,
    Inject,
    OnePerSemanticGroupFilter,
    ProviderFilter,
    RandomJitter,
    StreamableModelFilter,
    TopK,
)
from ypl.backend.llm.routing.modules.misc import ModifierAnnotator
from ypl.backend.llm.routing.modules.proposers import (
    AlwaysGoodModelMetaRouter,
    CostModelProposer,
    EloProposer,
    MaxSpeedProposer,
    ProModelProposer,
    RandomModelProposer,
    StrongModelProposer,
)
from ypl.backend.llm.routing.policy import decayed_random_fraction
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter, OpenAILangChainAdapter
from ypl.backend.utils.monitoring import metric_inc_by

# Begin pro router logic and routine
ROUTING_LLM: BaseChatModel | None = None
ONLINE_LABELER = None
TOPIC_LABELER = None
MODIFIER_LABELER = None
USE_GEMINI_FOR_ROUTING = False


async def get_simple_pro_router(
    prompt: str,
    num_models: int,
    routing_preference: RoutingPreference | None = None,
    reputable_providers: set[str] | None = None,
    user_selected_models: list[str] | None = None,
    show_me_more_models: list[str] | None = None,
    provided_categories: list[str] | None = None,
) -> RouterModule:
    """
    The main routing function.
    """
    from ypl.backend.llm.routing.rule_router import RoutingRuleFilter, RoutingRuleProposer

    preference = routing_preference or RoutingPreference(turns=[], user_id=None)
    reputable_proposer = RandomModelProposer(providers=reputable_providers or set(settings.ROUTING_REPUTABLE_PROVIDERS))
    online_labeler = _get_online_labeler()
    topic_labeler = _get_topic_labeler()
    modifier_labeler = _get_modifier_labeler()
    online_label, topic_labels, modifier_labels = await asyncio.gather(
        online_labeler.alabel(prompt),
        topic_labeler.alabel(prompt),
        modifier_labeler.alabel(prompt),
    )

    online_category = "online" if online_label else "offline"
    categories = [online_category] + topic_labels + (provided_categories or [])
    categories = list(dict.fromkeys(categories))
    applicable_modifiers = modifier_labels

    rule_proposer = RoutingRuleProposer(*categories)
    rule_filter = RoutingRuleFilter(*categories)
    error_filter = HighErrorRateFilter()
    pro_proposer = ProModelProposer()
    num_pro = int(pro_proposer.get_rng().random() * 2 + 1)

    show_me_more_providers = (
        set(deduce_original_providers(tuple(show_me_more_models)).values()) if show_me_more_models else set()
    )

    def get_postprocessing_stage(exclude_models: set[str] | None = None, prefix: str = "first") -> RouterModule:
        """
        Common post-processing stages that's shared in first-turn and non-first-turn routers
        """
        return (
            # -- filter stage --
            Exclude(providers=show_me_more_providers, models=exclude_models)
            # inject user selected model, even if they are already used before.
            # Also they are always treated with priority in the following dedup filters.
            | Inject(user_selected_models or [], score=10000000)
            | OnePerSemanticGroupFilter(priority_models=user_selected_models)  # dedupe by semantic group
            | ProviderFilter(one_per_provider=True, priority_models=user_selected_models)  # dedupe by provider
            # -- ranking stage --
            | TopK(num_models)  # keeps only top k models
            # -- annotation stage --
            | ModifierAnnotator(applicable_modifiers)  # annotate models with modifiers
            # -- logging stage --
            | RoutingDecisionLogger(
                enabled=settings.ROUTING_DO_LOGGING,
                prefix=f"{prefix}-prompt-simple-pro-router",
                metadata={
                    "user_id": preference.user_id,
                    "categories": categories,
                },
            )
        )

    if user_selected_models is not None and len(user_selected_models) > 0:
        metric_inc_by("routing/num_user_selected_models", len(user_selected_models))

    if not preference.turns:
        # --- First Turn Router ---
        # Construct a first-turn router guaranteeing at least one pro model and one reputable model.
        router: RouterModule = (
            # -- candidate prep stage --
            rule_filter  # apply rules for an initial pass clean up on all candidate models, reject based on prompt
            # -- proposal stage --
            | (  # propose a set of models based on several heuristics
                (rule_proposer.with_flags(always_include=True) | RandomJitter(jitter_range=1))
                & (pro_proposer | error_filter | TopK(num_pro)).with_flags(always_include=True, offset=100000)
                & (StrongModelProposer() | error_filter | TopK(1)).with_flags(always_include=True, offset=50000)
                & (
                    reputable_proposer
                    | error_filter
                    | StreamableModelFilter()
                    | MaxSpeedProposer()
                    | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                    | ProviderFilter(one_per_provider=True)
                ).with_flags(always_include=True, offset=5000)
            )
            | error_filter  # removes models with high error rate
            # -- post processing stage --
            | get_postprocessing_stage(exclude_models=None, prefix="first")
        )
    else:
        # --- Non-First Turn Router ---
        all_good_models, all_bad_models = _get_good_and_bad_models(preference)
        rule_filter.exempt_models = all_good_models

        router: RouterModule = (  # type: ignore[no-redef]
            # -- preprocessing stage --
            rule_filter
            # -- proposal stage --
            | (
                (RandomModelProposer(models=all_good_models) | error_filter | TopK(1)).with_flags(
                    always_include=True,
                    offset=10000000,
                )
                & (rule_proposer.with_flags(always_include=True) | error_filter | RandomJitter(jitter_range=1))
                & (
                    (ProModelProposer() | Exclude(models=all_bad_models) | error_filter | TopK(1)).with_flags(
                        always_include=True, offset=10000
                    )
                    ^ (
                        reputable_proposer
                        | StreamableModelFilter()
                        | error_filter
                        | MaxSpeedProposer()
                        | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                        | ProviderFilter(one_per_provider=True)
                    ).with_flags(always_include=True, offset=10000)
                    ^ RandomModelProposer().with_flags(offset=10000, always_include=True)
                ).with_probs(
                    settings.ROUTING_WEIGHTS.get("pro", 0.5),
                    settings.ROUTING_WEIGHTS.get("reputable", 0.25),
                    settings.ROUTING_WEIGHTS.get("random", 0.25),
                )
                & RandomModelProposer().with_flags(offset=-1000, always_include=True)
            )
            | error_filter  # removes models with high error rate
            # -- post processing stage --
            | get_postprocessing_stage(exclude_models=all_bad_models, prefix="nonfirst")
        )

    return router


# TODO(tian) - this is not used anywhere yet?
def get_prompt_conditional_router(
    prompt: str,
    num_models: int,
    routing_preference: RoutingPreference | None = None,
) -> RouterModule:
    """
    Get router based on the prompt
    """
    from ypl.backend.llm.routing.prompt_router import RemotePromptCategorizerProposer

    endpoint = settings.PYTORCH_SERVE_GCP_URL
    key = settings.X_API_KEY
    preference = routing_preference or RoutingPreference(turns=[], user_id=None)

    reputable_proposer = RandomModelProposer(providers=set(settings.ROUTING_REPUTABLE_PROVIDERS))
    categorizer_proposer = RemotePromptCategorizerProposer(
        prompt,
        endpoint,
        key,
        exclude_unknown_models=False,
        skill_deficit_threshold=4,
    )

    if not preference.turns:
        # Construct a first-turn router guaranteeing at least two reputable models, focusing on speed but
        # also with random jitter.
        router: RouterModule = (
            (
                (ProModelProposer() | TopK(1)).with_flags(always_include=True, offset=100000)
                & (
                    reputable_proposer
                    | categorizer_proposer
                    | StreamableModelFilter()
                    | MaxSpeedProposer()
                    | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                    | ProviderFilter(one_per_provider=True)
                ).with_flags(always_include=True, offset=5000)
                & reputable_proposer.with_flags(offset=-1000, always_include=True)
            )
            | ProviderFilter(one_per_provider=True)
            | TopK(num_models)
            | RoutingDecisionLogger(
                enabled=settings.ROUTING_DO_LOGGING,
                prefix="first-prompt-conditional-router",
                metadata={"user_id": preference.user_id},
            )
        )
    else:
        # This is the router for all turns after the first; construct it based on the preference
        # and the branding. If both are bad, we default to one branded and one random. If one branded one is
        # good, we choose that and use a random model for the other.
        all_good_models = set()
        all_bad_models = set()

        for turn in preference.turns:
            if not turn.has_evaluation:
                continue

            for model in turn.models:
                if turn.preferred is None:
                    all_bad_models.add(model)
                else:
                    if model == turn.preferred:
                        all_good_models.add(model)
                    else:
                        all_bad_models.add(model)

        all_good_models = all_good_models - all_bad_models

        router: RouterModule = (  # type: ignore[no-redef]
            (
                (RandomModelProposer(models=all_good_models).with_flags(offset=100000) | TopK(1)).with_flags(
                    always_include=True
                )
                & (ProModelProposer() | Exclude(models=all_bad_models) | TopK(1)).with_flags(
                    always_include=True, offset=10000
                )
                & (
                    categorizer_proposer
                    | Exclude(models=all_bad_models)
                    | ProviderFilter(one_per_provider=True)
                    | MaxSpeedProposer()
                    | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                    | TopK(1)
                ).with_flags(always_include=True)
                & RandomModelProposer().with_flags(offset=-1000, always_include=True)
            )
            | Exclude(models=all_bad_models)
            | ProviderFilter(one_per_provider=True)
            | TopK(num_models)
            | RoutingDecisionLogger(
                enabled=settings.ROUTING_DO_LOGGING,
                prefix="nonfirst-prompt-conditional-router",
                metadata={
                    "user_id": preference.user_id,
                    "turns": [t.model_dump() for t in preference.turns],
                    "all_good_models": list(all_good_models),
                    "all_bad_models": list(all_bad_models),
                },
            )
        )

    return router


def _get_online_labeler() -> YuppOnlinePromptLabeler:
    global ONLINE_LABELER
    if ONLINE_LABELER is None:
        ONLINE_LABELER = YuppOnlinePromptLabeler(_get_routing_llm(), timeout_secs=settings.ROUTING_TIMEOUT_SECS)
    return ONLINE_LABELER


def _get_topic_labeler() -> YuppMultilabelClassifier:
    global TOPIC_LABELER
    if TOPIC_LABELER is None:
        TOPIC_LABELER = YuppMultilabelClassifier(_get_routing_llm(), timeout_secs=settings.ROUTING_TIMEOUT_SECS)
    return TOPIC_LABELER


def _get_modifier_labeler() -> PromptModifierLabeler:
    global MODIFIER_LABELER
    if MODIFIER_LABELER is None:
        MODIFIER_LABELER = PromptModifierLabeler(_get_routing_llm(), timeout_secs=settings.ROUTING_TIMEOUT_SECS)
    return MODIFIER_LABELER


def _get_good_and_bad_models(preference: RoutingPreference) -> tuple[set[str], set[str]]:
    """
    Go through all previous turns and split models into a good set and a bad set based on their user evaluation.
    """
    all_good_models = set()
    all_bad_models = set()
    # Go through all previous turns, add preferred models to all_good_models and others to all_bad_models
    for turn in preference.turns or []:
        if not turn.has_evaluation:
            continue
        if turn.preferred:
            all_good_models.add(turn.preferred)
            all_bad_models.update([m for m in turn.models if m != turn.preferred])
        else:
            all_bad_models.update(turn.models)
    all_good_models = all_good_models - all_bad_models
    return all_good_models, all_bad_models


def _get_routing_llm() -> BaseChatModel:
    global ROUTING_LLM
    if ROUTING_LLM is None:
        if USE_GEMINI_FOR_ROUTING:
            ROUTING_LLM = GeminiLangChainAdapter(
                model_info=ModelInfo(
                    provider=ChatProvider.GOOGLE,
                    model="gemini-2.0-flash-exp",
                    api_key=settings.GOOGLE_API_KEY,
                ),
                model_config_=dict(
                    project_id=settings.GCP_PROJECT_ID,
                    region=settings.GCP_REGION_GEMINI_2,
                    temperature=0.0,
                    max_output_tokens=32,
                    top_k=1,
                ),
            )
        else:
            ROUTING_LLM = OpenAILangChainAdapter(
                model_info=ModelInfo(
                    provider=ChatProvider.OPENAI,
                    model="gpt-4o-mini",
                    api_key=settings.OPENAI_API_KEY,
                ),
                model_config_=dict(
                    temperature=0.0,
                    max_tokens=40,
                ),
            )

    return ROUTING_LLM


"""
TODO(tian) - Following functions seem not used anywhere, review or remove them later.
"""


# TODO - review or remove, not used anywhere
def get_router_ranker(ranker: Ranker | None = None) -> tuple[RouterModule, Ranker]:
    min_weight = settings.ROUTING_WEIGHTS.get("min_simple_cost", 0.1)
    rand_weight = settings.ROUTING_WEIGHTS.get("random", 0.1)
    top_weight = settings.ROUTING_WEIGHTS.get("top", 0.1)
    ranker = ranker or get_ranker()

    router: RouterModule = (CostModelProposer() ^ RandomModelProposer() ^ EloProposer(ranker)).with_probs(
        min_weight,
        rand_weight + decayed_random_fraction(ranker, initial_value=0.6, final_value=0.05, steps=50000),
        top_weight,
    )

    if settings.ROUTING_GOOD_MODELS_ALWAYS:
        router = AlwaysGoodModelMetaRouter(ranker, router, num_good=settings.ROUTING_GOOD_MODELS_RANK_THRESHOLD)

    router = router | TopK(2) | RoutingDecisionLogger(enabled=settings.ROUTING_DO_LOGGING, prefix="default-router")

    return router, ranker


# TODO - review or remove, not used anywhere
def get_router(ranker: Ranker | None = None) -> RouterModule:
    return get_router_ranker(ranker)[0]


# TODO - review or remove, not used anywhere
def get_gcp_cloud_run_uri(service_name: str, region: str) -> str:
    credentials, project_id = default()
    client = run_v2.ServicesClient(credentials=credentials)
    name = f"projects/{project_id}/locations/{region}/services/{service_name}"
    request = run_v2.GetServiceRequest(name=name)
    response = client.get_service(request=request)

    return response.uri
