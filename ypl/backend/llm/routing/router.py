from google.auth import default
from google.cloud import run_v2
from langchain_core.language_models.chat_models import BaseChatModel

from ypl.backend.config import settings
from ypl.backend.llm.constants import IMAGE_CATEGORY, IMAGE_GEN_CATEGORY, ONLINE_CATEGORY, PDF_CATEGORY
from ypl.backend.llm.db_helpers import deduce_original_providers, get_all_active_models, is_user_internal
from ypl.backend.llm.promotions import PromotionModelProposer
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.llm.ranking import Ranker, get_ranker
from ypl.backend.llm.routing.common import SelectIntent
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.modules.decision import RoutingDecisionLogger
from ypl.backend.llm.routing.modules.filters import (
    ContextLengthFilter,
    Exclude,
    FirstK,
    HighErrorRateFilter,
    Inject,
    LiveModelFilter,
    OnePerSemanticGroupFilter,
    ProviderFilter,
    RandomJitter,
    StreamableModelFilter,
    SupportsImageAttachmentModelFilter,
    SupportsPdfAttachmentModelFilter,
    TopK,
)
from ypl.backend.llm.routing.modules.misc import Passthrough
from ypl.backend.llm.routing.modules.proposers import (
    AlwaysGoodModelMetaRouter,
    CostModelProposer,
    EloProposer,
    FastModelProposer,
    ImageGenModelsProposer,
    LiveModelProposer,
    MaxSpeedProposer,
    ModelProposer,
    ProAndStrongModelProposer,
    ProModelProposer,
    RandomModelProposer,
    ReasoningModelProposer,
    StrongModelProposer,
)
from ypl.backend.llm.routing.modules.rankers import (
    PositionMatchReranker,
    ProAndStrongReranker,
    PromotionModelReranker,
    ProviderScatterer,
    ReasoningModelReranker,
    ScoreReranker,
    SemanticGroupScatterer,
    SpeedReranker,
    YappReranker,
)
from ypl.backend.llm.routing.policy import SelectionCriteria, decayed_random_fraction
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.rule_router import RoutingRuleFilter, RoutingRuleProposer
from ypl.backend.utils.monitoring import metric_inc_by

# Begin pro router logic and routine
ROUTING_LLM: BaseChatModel | None = None
USE_GEMINI_FOR_ROUTING = False


def _get_good_and_bad_models(preference: RoutingPreference, has_pdf: bool) -> tuple[set[str], set[str]]:
    """
    Go through all previous turns and split models into a good set and a bad set based on their user evaluation.
    good = preferred
    bad = downvoted or failed or user-stopped
    """
    all_good_models = set()
    all_bad_models = set()
    for turn in preference.turns or []:
        if turn.preferred:
            all_good_models.add(turn.preferred)
        if turn.failed_models and not has_pdf:
            all_bad_models.update(turn.failed_models)
        if turn.downvoted and not has_pdf:
            all_bad_models.update(turn.downvoted)
    all_good_models = all_good_models - all_bad_models
    return all_good_models, all_bad_models


async def get_simple_pro_router(
    prompt: str,
    num_models: int,
    preference: RoutingPreference,
    reputable_providers: set[str] | None = None,
    user_selected_models: list[str] | None = None,
    inherited_models: list[str] | None = None,
    same_turn_shown_models: list[str] | None = None,
    provided_categories: list[str] | None = None,
    chat_id: str | None = None,
    turn_id: str | None = None,
    intent: SelectIntent = SelectIntent.NEW_CHAT,
    with_fallback: bool = False,
) -> RouterModule:
    """
    The main routing function.
    Args:
        prompt: the prompt to route
        num_models: the number of models needed for the UI, not including the fallback models
        preference: the chat history information (past PREFs, NOPEs, used models, etc)
        reputable_providers: the reputable providers to propose models from (TODO(Tian): deprecate this)
        user_selected_models: models selected explicitly by the user in the model picker
        inherited_models: models used and need to be reused from the past turns of the same chat
        same_turn_shown_models: models already shown in the current turn (if this is a SMM round)
        provided_categories: categories detected from the prompt plus those passed in in request
        chat_id: the chat ID
        turn_id: the turn ID
        intent: the intent of the request
        with_fallback: whether to include fallback models, if true, will return 2*num_models

    Returns:
        A RouterModule for the routing chain
    """
    num_models_to_return = num_models * 2 if with_fallback else num_models
    first_turn = intent == SelectIntent.NEW_CHAT
    follow_up_turns = intent == SelectIntent.NEW_TURN
    show_me_more = intent == SelectIntent.SHOW_ME_MORE

    categories = provided_categories or []
    same_turn_shown_providers = (
        set(deduce_original_providers(tuple(same_turn_shown_models)).values()) if same_turn_shown_models else set()
    )

    reputable_proposer = RandomModelProposer(
        for_criteria=SelectionCriteria.RANDOM_REPUTABLE,
        providers=reputable_providers or set(settings.ROUTING_REPUTABLE_PROVIDERS),
    )

    # Abilities needed.
    has_image = IMAGE_CATEGORY in categories
    has_pdf = PDF_CATEGORY in categories
    has_attachment = has_image or has_pdf
    needs_online_access = ONLINE_CATEGORY in categories
    needs_image_gen = IMAGE_GEN_CATEGORY in categories
    image_filter = SupportsImageAttachmentModelFilter() if IMAGE_CATEGORY in categories else Passthrough()
    pdf_filter = SupportsPdfAttachmentModelFilter() if PDF_CATEGORY in categories else Passthrough()
    attachment_filter = image_filter | pdf_filter
    live_model_filter = LiveModelFilter() if needs_online_access and not has_attachment else Passthrough()

    rule_proposer = RoutingRuleProposer(*categories)
    rule_filter = RoutingRuleFilter(*categories)
    error_filter = HighErrorRateFilter() if not has_pdf else HighErrorRateFilter(soft_threshold=0.2, hard_threshold=0.4)

    include_internal_models = preference.user_id is not None and (await is_user_internal(preference.user_id))

    required_models = (user_selected_models or []) + (inherited_models or [])

    def get_preprocessing_stage(prompt: str) -> RouterModule:
        """
        All necessary preprocessing steps, filtering out models we definitely cannot use.
        As of 3/11/25, only groundedvertexai/gemini-2.0-flash-001-online can handle both attachments and online access.
        """
        return (
            # we inject inherited models early as they are subject to capabilities filtering, if they
            # get through this stage, their high score will help them win the ranking.
            Inject(inherited_models or [], score=49_000_000)
            | rule_filter
            | ContextLengthFilter(prompt)
            | attachment_filter
            | live_model_filter
            | error_filter
        )

    async def get_postprocessing_stage(exclude_models: set[str] | None = None, prefix: str = "first") -> RouterModule:
        """
        Common post-processing stages that's shared in first-turn and non-first-turn routers
        """
        semantic_group_filter = OnePerSemanticGroupFilter(priority_models=required_models)
        return (
            # -- filter stage --
            (  # Exclude already shown providers, except for image gen, where only the shown models are excluded.
                Exclude(name="-exclBad", providers=same_turn_shown_providers, exclude_models=exclude_models)
                if not needs_image_gen
                else Exclude(
                    name="-exclBad",
                    providers=set(),
                    exclude_models=(exclude_models or set()) | set(same_turn_shown_models or []),
                )
            )
            # Inject required models, even if they don't have attachment capabilities.
            | Inject(user_selected_models or [], score=50_000_000)
            # exclude inactive models after injection, this is necessary in case we are injecting models inferred
            # from the history of the chat but they are no longer active.
            | Exclude(name="-inactive", whitelisted_models=await get_all_active_models(include_internal_models))
            # exclude Yapp models in SMM rounds.
            | (Exclude(name="-yapp", providers={"Yapp"}) if show_me_more else Passthrough())
            # Don't apply semantic group filter for image turns, since we don't have many supporting models.
            | (semantic_group_filter if not has_attachment else Passthrough())
            | (
                ProviderFilter(one_per_provider=True, priority_models=required_models)
                if not has_attachment and not needs_image_gen
                else Passthrough()
            )  # dedupe by provider, relax for image/pdf needs
            # -- ranking stage --
            | ScoreReranker()
            | PromotionModelReranker()
            | (
                SemanticGroupScatterer(min_dist=num_models_to_return) if has_attachment else Passthrough()
            )  # scatter models with same semantic group
            | (ProviderScatterer(min_dist=num_models_to_return) if has_attachment else Passthrough())
            | YappReranker(num_models)  # yapp models should never be in the fallback
            | FirstK(num_models_to_return, num_primary_models=num_models, name="final")
            # Final tweaks of the order after trimming down to num_models_to_return
            | ReasoningModelReranker(num_models)
            | (SpeedReranker(num_models) if not first_turn else Passthrough())  # rerank by speed only in new turns
            | (ProAndStrongReranker(exempt_models=required_models) if first_turn else Passthrough())
            | (PositionMatchReranker(preference, num_models) if follow_up_turns else Passthrough())
            # -- logging stage --
            | RoutingDecisionLogger(
                enabled=settings.ROUTING_DO_LOGGING,
                prefix=f"{intent}",
                preference=preference,
                required_models=required_models,
                metadata={
                    "user_id": preference.user_id,
                    "chat_id": chat_id,
                    "turn_id": turn_id,
                    "categories": categories,
                },
            )
        )

    if required_models is not None and len(required_models) > 0:
        metric_inc_by("routing/num_required_models_for_routing", len(required_models))

    def propose_type(proposer: type[ModelProposer], name: str, offset: int = 50_000) -> RouterModule:
        return proposer() | error_filter | TopK(1, name=name).with_flags(always_include=True, offset=offset)

    if intent == SelectIntent.NEW_CHAT:
        # --- First Turn (NEW_CHAT) ---
        # Construct a first-turn router guaranteeing at least one pro model and one reputable model.
        router: RouterModule = (
            # -- candidate prep stage --
            get_preprocessing_stage(prompt)
            # -- proposal stage --
            | (
                # propose through routing table rules
                (rule_proposer.with_flags(always_include=True) | RandomJitter(jitter_range=1))
                # Propose image gen models explicitly so that IMAGE_GEN_CATEGORY is their selection criteria.
                & (ImageGenModelsProposer() if needs_image_gen else Passthrough())
                # always try to have something pro and strong in the first turn
                & (ProAndStrongModelProposer() | error_filter | TopK(1, name="pro_and_strong")).with_flags(
                    always_include=True, offset=1_000_000
                )
                & (  # diversification
                    propose_type(StrongModelProposer, "strong", 50_000)
                    ^ propose_type(FastModelProposer, "fast", 50_000)
                    ^ propose_type(LiveModelProposer, "live", 50_000)
                    ^ propose_type(ReasoningModelProposer, "reasoning", 50_000)
                )
                # propose promoted models
                & (PromotionModelProposer() | error_filter).with_flags(always_include=True)
                # propose reputable models
                & (
                    reputable_proposer
                    | error_filter
                    | StreamableModelFilter()
                    | MaxSpeedProposer()
                    | ContextLengthFilter(prompt)
                    | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                    | ProviderFilter(one_per_provider=True)
                ).with_flags(always_include=True, offset=5_000)
            )
            | error_filter  # removes models with high error rate
            # -- post processing stage --
            | await get_postprocessing_stage(exclude_models=None, prefix="first")
        )
    else:
        # --- Non-First Turn (NEW_TURN or SHOW_ME_MORE) ---
        all_good_models, all_bad_models = _get_good_and_bad_models(preference, has_pdf=has_pdf)
        rule_filter.exempt_models = all_good_models
        no_proposal_prob = 1.0 / (len(all_good_models) + 1)

        router: RouterModule = (  # type: ignore[no-redef]
            # -- preprocessing stage --
            get_preprocessing_stage(prompt)
            # -- proposal stage --
            | (
                # propose all previous good models (preferred models)
                (
                    (RandomModelProposer(models=all_good_models) | error_filter | TopK(1, name="rnd-NF")).with_flags(
                        always_include=True, offset=1_0000
                    )
                    ^ Passthrough()
                ).with_probs(1 - no_proposal_prob, no_proposal_prob)
                # propose through routing table rules
                & (rule_proposer.with_flags(always_include=True) | error_filter | RandomJitter(jitter_range=1))
                # Propose image gen models explicitly so that IMAGE_GEN_CATEGORY is their selection criteria.
                & (ImageGenModelsProposer() if needs_image_gen else Passthrough())
                # propose pro OR reputable OR random models, choosing them random using the probabilities in with_probs
                & (
                    (
                        ProModelProposer()
                        | Exclude(name="-ex-NF", exclude_models=all_bad_models)
                        | error_filter
                        | TopK(1, name="pro-NF")
                    ).with_flags(always_include=True, offset=10_000)
                    ^ (
                        reputable_proposer
                        | StreamableModelFilter()
                        | error_filter
                        | image_filter
                        | MaxSpeedProposer()
                        | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                        | ProviderFilter(one_per_provider=True)
                    ).with_flags(always_include=True, offset=10_000)
                    ^ RandomModelProposer().with_flags(offset=10_000, always_include=True)
                ).with_probs(
                    settings.ROUTING_WEIGHTS.get("pro", 0.5),
                    settings.ROUTING_WEIGHTS.get("reputable", 0.3),
                    settings.ROUTING_WEIGHTS.get("random", 0.2),
                )
                # propose promoted models
                & (PromotionModelProposer() | error_filter).with_flags(always_include=True)
                # propose even more random models but low score
                & RandomModelProposer().with_flags(offset=-1_000, always_include=True)
            )
            | error_filter  # removes models with high error rate
            # -- post processing stage --
            | await get_postprocessing_stage(exclude_models=all_bad_models, prefix="nonfirst")
        )

    return router


async def get_default_routing_llm() -> BaseChatModel:
    if USE_GEMINI_FOR_ROUTING:
        return await get_internal_provider_client("gemini-2.0-flash-001", max_tokens=32)
    else:
        return await get_internal_provider_client("gpt-4o-mini", max_tokens=40)


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

    router = (
        router
        | TopK(2, name="ranker")
        | RoutingDecisionLogger(enabled=settings.ROUTING_DO_LOGGING, prefix="default-router")
    )

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
