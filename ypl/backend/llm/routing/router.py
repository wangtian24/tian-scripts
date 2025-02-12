from google.auth import default
from google.cloud import run_v2
from langchain_core.language_models.chat_models import BaseChatModel

from ypl.backend.config import settings
from ypl.backend.llm.constants import IMAGE_CATEGORY, ONLINE_CATEGORY, PDF_CATEGORY, ChatProvider
from ypl.backend.llm.db_helpers import deduce_original_providers
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.promotions import PromotionModelProposer
from ypl.backend.llm.ranking import Ranker, get_ranker
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.modules.decision import RoutingDecisionLogger
from ypl.backend.llm.routing.modules.filters import (
    ContextLengthFilter,
    Exclude,
    FirstK,
    HighErrorRateFilter,
    Inject,
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
    ImageProModelProposer,
    MaxSpeedProposer,
    PdfProModelProposer,
    ProModelProposer,
    RandomModelProposer,
    StrongModelProposer,
)
from ypl.backend.llm.routing.modules.rankers import (
    PositionMatchReranker,
    PromotionModelReranker,
    ScoreReranker,
    SemanticGroupReranker,
    SpeedReranker,
    YappReranker,
)
from ypl.backend.llm.routing.policy import SelectionCriteria, decayed_random_fraction
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.rule_router import RoutingRuleFilter, RoutingRuleProposer
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter, OpenAILangChainAdapter
from ypl.backend.utils.monitoring import metric_inc_by

# Begin pro router logic and routine
ROUTING_LLM: BaseChatModel | None = None
USE_GEMINI_FOR_ROUTING = False


def needs_attachment(categories: list[str]) -> bool:
    return IMAGE_CATEGORY in categories or PDF_CATEGORY in categories


def needs_special_ability(categories: list[str]) -> bool:
    # TODO(tian) - this might be a bit hacky, here we manually maintain a list we know that the routing chain is using.
    # we will change this to a more proper model ability check in the future.
    return needs_attachment(categories) or categories in ["coding", ONLINE_CATEGORY]


async def get_simple_pro_router(
    prompt: str,
    num_models: int,
    preference: RoutingPreference,
    reputable_providers: set[str] | None = None,
    required_models: list[str] | None = None,
    show_me_more_models: list[str] | None = None,
    provided_categories: list[str] | None = None,
    chat_id: str | None = None,
    is_new_turn: bool = False,
    with_fallback: bool = False,
) -> RouterModule:
    """
    The main routing function.
    """
    num_models_to_return = num_models * 2 if with_fallback else num_models

    categories = provided_categories or []
    show_me_more_providers = (
        set(deduce_original_providers(tuple(show_me_more_models)).values()) if show_me_more_models else set()
    )

    reputable_proposer = RandomModelProposer(
        for_criteria=SelectionCriteria.RANDOM_REPUTABLE,
        providers=reputable_providers or set(settings.ROUTING_REPUTABLE_PROVIDERS),
    )

    has_image = IMAGE_CATEGORY in categories
    has_pdf = PDF_CATEGORY in categories
    has_attachment = has_image or has_pdf

    pro_proposer = ImageProModelProposer() if has_image else PdfProModelProposer() if has_pdf else ProModelProposer()

    rule_proposer = RoutingRuleProposer(*categories)
    rule_filter = RoutingRuleFilter(*categories)
    error_filter = HighErrorRateFilter()
    num_pro = int(pro_proposer.get_rng().random() * 2 + 1)

    image_proposer = ImageProModelProposer() if IMAGE_CATEGORY in categories else Passthrough()
    pdf_proposer = PdfProModelProposer() if PDF_CATEGORY in categories else Passthrough()
    attachment_proposer = image_proposer | pdf_proposer

    image_filter = SupportsImageAttachmentModelFilter() if IMAGE_CATEGORY in categories else Passthrough()
    pdf_filter = SupportsPdfAttachmentModelFilter() if PDF_CATEGORY in categories else Passthrough()
    attachment_filter = image_filter | pdf_filter

    def get_postprocessing_stage(exclude_models: set[str] | None = None, prefix: str = "first") -> RouterModule:
        """
        Common post-processing stages that's shared in first-turn and non-first-turn routers
        """
        semantic_group_filter = OnePerSemanticGroupFilter(priority_models=required_models)
        return (
            # -- filter stage --
            Exclude(name="-exSMM", providers=show_me_more_providers, models=exclude_models)
            # Inject required models, even if they don't have attachment capabilities.
            | Inject(required_models or [], score=50000000)
            # remove models that don't support needed attachment
            | attachment_filter
            # Don't apply semantic group filter for image turns, since we don't have many supporting models.
            | (semantic_group_filter if not has_attachment else Passthrough())
            | (
                ProviderFilter(one_per_provider=True, priority_models=required_models)
                if not has_attachment
                else Passthrough()
            )  # dedupe by provider, relax for image/pdf needs
            # -- ranking stage --
            | ScoreReranker()
            | PromotionModelReranker()
            | (
                SemanticGroupReranker(min_dist=num_models_to_return) if has_attachment else Passthrough()
            )  # scatter models with same semantic group
            | YappReranker(num_models)  # yapp models should never be in the fallback
            | FirstK(num_models_to_return, num_primary_models=num_models, name="final")
            | SpeedReranker(num_models)  # rerank final results with speed, the fastest models always in the front
            | (PositionMatchReranker(preference, num_models) if is_new_turn else Passthrough())
            # -- logging stage --
            | RoutingDecisionLogger(
                enabled=settings.ROUTING_DO_LOGGING,
                prefix=f"{prefix}-prompt-simple-pro-router",
                preference=preference,
                required_models=required_models,
                metadata={
                    "user_id": preference.user_id,
                    "chat_id": chat_id,
                    "categories": categories,
                },
            )
        )

    if required_models is not None and len(required_models) > 0:
        metric_inc_by("routing/num_user_selected_models", len(required_models))

    if not preference.turns:
        # --- First Turn (NEW_TURN) ---
        # Construct a first-turn router guaranteeing at least one pro model and one reputable model.
        router: RouterModule = (
            # -- candidate prep stage --
            rule_filter  # apply rules for an initial pass clean up on all candidate models, reject based on prompt
            # -- proposal stage --
            | (
                # propose through routing table rules
                (rule_proposer.with_flags(always_include=True) | RandomJitter(jitter_range=1))
                # propose pro models
                & (pro_proposer | error_filter | TopK(num_pro, name="pro")).with_flags(
                    always_include=True, offset=100000
                )
                # propose strong models
                & (StrongModelProposer() | error_filter | TopK(1, name="strong")).with_flags(
                    always_include=True, offset=50000
                )
                # propose promoted models
                & (PromotionModelProposer() | error_filter).with_flags(always_include=True)
                # propose models with image or pdf capabilities, strong offset
                & (attachment_proposer | error_filter).with_flags(always_include=True, offset=200000)
                # propose reputable models
                & (
                    reputable_proposer
                    | error_filter
                    | StreamableModelFilter()
                    | MaxSpeedProposer()
                    | ContextLengthFilter(prompt)
                    | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                    | ProviderFilter(one_per_provider=True)
                ).with_flags(always_include=True, offset=5000)
            )
            | error_filter  # removes models with high error rate
            # -- post processing stage --
            | get_postprocessing_stage(exclude_models=None, prefix="first")
        )
    else:
        # --- Non-First Turn (NEW_TURN or SHOW_ME_MORE) ---
        all_good_models, all_bad_models = _get_good_and_bad_models(preference)
        rule_filter.exempt_models = all_good_models
        no_proposal_prob = 1.0 / (len(all_good_models) + 1)

        router: RouterModule = (  # type: ignore[no-redef]
            # -- preprocessing stage --
            rule_filter
            # -- proposal stage --
            | (
                # propose all previous good models (preferred models)
                (
                    (RandomModelProposer(models=all_good_models) | error_filter | TopK(1, name="rnd-NF")).with_flags(
                        always_include=True, offset=10000
                    )
                    ^ Passthrough()
                ).with_probs(1 - no_proposal_prob, no_proposal_prob)
                # propose through routing table rules
                & (rule_proposer.with_flags(always_include=True) | error_filter | RandomJitter(jitter_range=1))
                # propose models with image capabilities, strong offset
                & (
                    image_proposer.with_flags(always_include=True, offset=20000000)
                    | error_filter
                    | RandomJitter(jitter_range=1)
                )
                # propose pro OR reputable OR random models, choosing them random using the probabilities in with_probs
                & (
                    (
                        pro_proposer
                        | Exclude(name="-ex-NF", models=all_bad_models)
                        | error_filter
                        | TopK(1, name="pro-NF")
                    ).with_flags(always_include=True, offset=10000)
                    ^ (
                        reputable_proposer
                        | StreamableModelFilter()
                        | error_filter
                        | image_filter
                        | MaxSpeedProposer()
                        | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                        | ProviderFilter(one_per_provider=True)
                    ).with_flags(always_include=True, offset=10000)
                    ^ RandomModelProposer().with_flags(offset=10000, always_include=True)
                ).with_probs(
                    settings.ROUTING_WEIGHTS.get("pro", 0.5),
                    settings.ROUTING_WEIGHTS.get("reputable", 0.3),
                    settings.ROUTING_WEIGHTS.get("random", 0.2),
                )
                # propose promoted models
                & (PromotionModelProposer() | error_filter).with_flags(always_include=True)
                # propose even more random models but low score
                & RandomModelProposer().with_flags(offset=-1000, always_include=True)
            )
            | error_filter  # removes models with high error rate
            # -- post processing stage --
            | get_postprocessing_stage(exclude_models=all_bad_models, prefix="nonfirst")
        )

    return router


def _get_good_and_bad_models(preference: RoutingPreference) -> tuple[set[str], set[str]]:
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
        if turn.failed_models:
            all_bad_models.update(turn.failed_models)
        if turn.downvoted:
            all_bad_models.update(turn.downvoted)
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
