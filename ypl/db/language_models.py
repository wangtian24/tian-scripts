import uuid
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import BigInteger, Boolean, Column, Integer, Numeric, String, UniqueConstraint
from sqlalchemy import Enum as sa_Enum
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.users import User

if TYPE_CHECKING:
    from ypl.db.chats import ChatMessage, TurnQuality
    from ypl.db.promotions import ModelPromotion
    from ypl.db.ratings import Rating, RatingHistory


class LanguageModelStatusEnum(Enum):
    SUBMITTED = "SUBMITTED"
    VERIFIED_PENDING_ACTIVATION = "VERIFIED_PENDING_ACTIVATION"
    REJECTED = "REJECTED"
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class LicenseEnum(Enum):
    unknown = "Unknown"
    apache_2_0 = "Apache license 2.0"
    mit = "MIT"
    openrail = "OpenRAIL license family"
    bigscience_openrail_m = "BigScience OpenRAIL-M"
    creativeml_openrail_m = "CreativeML OpenRAIL-M"
    bigscience_bloom_rail_1_0 = "BigScience BLOOM RAIL 1.0"
    bigcode_openrail_m = "BigCode Open RAIL-M v1"
    afl_3_0 = "Academic Free License v3.0"
    artistic_2_0 = "Artistic license 2.0"
    bsl_1_0 = "Boost Software License 1.0"
    bsd = "BSD license family"
    bsd_2_clause = "BSD 2-clause “Simplified” license"
    bsd_3_clause = "BSD 3-clause “New” or “Revised” license"
    bsd_3_clause_clear = "BSD 3-clause Clear license"
    c_uda = "Computational Use of Data Agreement"
    cc = "Creative Commons license family"
    cc0_1_0 = "Creative Commons Zero v1.0 Universal"
    cc_by_2_0 = "Creative Commons Attribution 2.0"
    cc_by_2_5 = "Creative Commons Attribution 2.5"
    cc_by_3_0 = "Creative Commons Attribution 3.0"
    cc_by_4_0 = "Creative Commons Attribution 4.0"
    cc_by_sa_3_0 = "Creative Commons Attribution Share Alike 3.0"
    cc_by_sa_4_0 = "Creative Commons Attribution Share Alike 4.0"
    cc_by_nc_2_0 = "Creative Commons Attribution Non Commercial 2.0"
    cc_by_nc_3_0 = "Creative Commons Attribution Non Commercial 3.0"
    cc_by_nc_4_0 = "Creative Commons Attribution Non Commercial 4.0"
    cc_by_nd_4_0 = "Creative Commons Attribution No Derivatives 4.0"
    cc_by_nc_nd_3_0 = "Creative Commons Attribution Non Commercial No Derivatives 3.0"
    cc_by_nc_nd_4_0 = "Creative Commons Attribution Non Commercial No Derivatives 4.0"
    cc_by_nc_sa_2_0 = "Creative Commons Attribution Non Commercial Share Alike 2.0"
    cc_by_nc_sa_3_0 = "Creative Commons Attribution Non Commercial Share Alike 3.0"
    cc_by_nc_sa_4_0 = "Creative Commons Attribution Non Commercial Share Alike 4.0"
    cdla_sharing_1_0 = "Community Data License Agreement – Sharing, Version 1.0"
    cdla_permissive_1_0 = "Community Data License Agreement – Permissive, Version 1.0"
    cdla_permissive_2_0 = "Community Data License Agreement – Permissive, Version 2.0"
    wtfpl = "Do What The F*ck You Want To Public License"
    ecl_2_0 = "Educational Community License v2.0"
    epl_1_0 = "Eclipse Public License 1.0"
    epl_2_0 = "Eclipse Public License 2.0"
    etalab_2_0 = "Etalab Open License 2.0"
    eupl_1_1 = "European Union Public License 1.1"
    agpl_3_0 = "GNU Affero General Public License v3.0"
    gfdl = "GNU Free Documentation License family"
    gpl = "GNU General Public License family"
    gpl_2_0 = "GNU General Public License v2.0"
    gpl_3_0 = "GNU General Public License v3.0"
    lgpl = "GNU Lesser General Public License family"
    lgpl_2_1 = "GNU Lesser General Public License v2.1"
    lgpl_3_0 = "GNU Lesser General Public License v3.0"
    isc = "ISC"
    lppl_1_3c = "LaTeX Project Public License v1.3c"
    ms_pl = "Microsoft Public License"
    apple_ascl = "Apple Sample Code license"
    mpl_2_0 = "Mozilla Public License 2.0"
    odc_by = "Open Data Commons License Attribution family"
    odbl = "Open Database License family"
    openrail_pp = "Open Rail++-M License"
    osl_3_0 = "Open Software License 3.0"
    postgresql = "PostgreSQL License"
    ofl_1_1 = "SIL Open Font License 1.1"
    ncsa = "University of Illinois/NCSA Open Source License"
    unlicense = "The Unlicense"
    zlib = "zLib License"
    pddl = "Open Data Commons Public Domain Dedication and License"
    lgpl_lr = "Lesser General Public License For Linguistic Resources"
    deepfloyd_if_license = "DeepFloyd IF Research License Agreement"
    llama2 = "Llama 2 Community License Agreement"
    llama3 = "Llama 3 Community License Agreement"
    gemma = "Gemma Terms of Use"
    other = "Other"


class LanguageModelLicense(BaseModel, table=True):
    __tablename__ = "language_model_licenses"

    language_model_license_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(index=True, nullable=False, unique=True)
    models: list["LanguageModel"] = Relationship(back_populates="language_model_license")


class LanguageModel(BaseModel, table=True):
    __tablename__ = "language_models"

    __table_args__ = (
        UniqueConstraint("name", "provider_id"),
        UniqueConstraint("internal_name", "provider_id"),
    )

    language_model_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # This is the name displayed to the user, e.g. "gpt-4o-2024-05-13".
    # This name can be pseudonymous, e.g. "anonymous-model" with internal_name
    # "gpt-4o-2024-05-13". This is useful when Model Providers want to train
    # their models anonymously.
    # This is unique per provider.
    name: str = Field(index=True)

    # This is the "real" name of the model as given by the Model Provider,
    # e.g. "gpt-4o-2024-05-13".
    # This is unique per provider, and is sent to the model provider for identification.
    internal_name: str = Field(sa_column=Column("internal_name", sa.VARCHAR(), nullable=False, index=True))

    # This is a human-readable name for the model, e.g. "GPT 4o".
    label: str | None = Field(default=None)

    license: LicenseEnum = Field(
        default=LicenseEnum.unknown, sa_column=Column(sa_Enum(LicenseEnum), server_default=LicenseEnum.unknown.name)
    )
    family: str | None = Field(default=None)
    avatar_url: str | None = Field(default=None)

    # This is the number of parameters in the model.
    parameter_count: int | None = Field(sa_column=Column(BigInteger(), nullable=True), default=None)

    # This is the context window of the model.
    context_window_tokens: int | None = Field(sa_column=Column(BigInteger(), nullable=True), default=None)

    # This is the knowledge cutoff of the model in yyyy mm dd format.
    # For example, a knowledge cutoff of 2024 06 15 means the model was trained on data up to June 15, 2024.
    knowledge_cutoff_date: date | None = Field(default=None, nullable=True)

    # Input cost in USD per million tokens, stored with 6 decimal places.
    input_cost_usd_per_million_tokens: Decimal | None = Field(
        sa_column=Column(Numeric(precision=10, scale=6), nullable=True), default=None
    )
    # Output cost in USD per million tokens, stored with 6 decimal places.
    output_cost_usd_per_million_tokens: Decimal | None = Field(
        sa_column=Column(Numeric(precision=10, scale=6), nullable=True), default=None
    )

    # Median time in milliseconds to emit the first token
    first_token_avg_latency_ms: float | None = Field(default=None, nullable=True)
    first_token_p50_latency_ms: float | None = Field(default=None, nullable=True)

    # P90 time in milliseconds to emit the first token
    first_token_p90_latency_ms: float | None = Field(default=None, nullable=True)

    # Median tokens per second to write the output tokens
    output_avg_tps: float | None = Field(default=None, nullable=True)
    output_p50_tps: float | None = Field(default=None, nullable=True)

    # P90 tokens per second to write the output tokens
    output_p90_tps: float | None = Field(default=None, nullable=True)

    # This is the status of the language model. Once a new model is created, its status is SUBMITTED.
    # If the model is rejected, it is not made available to the public and status is set to REJECTED.
    # After the model has been verified by the automatic checks, the status is set to VERIFIED_PENDING_ACTIVATION.
    # If the model is verified and some manual work is needed then post completion of the same,
    # the status is set to ACTIVE.
    # If the model is no longer available to the public, the status is set to INACTIVE.
    # Don't softdelete as we need to preserve history for ranking and leaderboard.
    status: LanguageModelStatusEnum = Field(
        default=LanguageModelStatusEnum.SUBMITTED,
        sa_column=Column(sa_Enum(LanguageModelStatusEnum), server_default=LanguageModelStatusEnum.SUBMITTED.name),
    )

    # Whether the model is considered a pro model.
    is_pro: bool | None = Field(nullable=True, default=None, index=True)
    # Whether the model is considered a strong model, not exposed to users but used for routing.
    is_strong: bool | None = Field(nullable=True, default=None, index=True)
    # Whether the model supports real time queries.
    is_live: bool | None = Field(nullable=True, default=None, index=True)

    # Eventually we will link to our own model info pages. But this is a short term solution.
    external_model_info_url: str | None = Field(nullable=True, default=None)

    # This is the organization that owns the language model.
    organization_id: uuid.UUID | None = Field(foreign_key="organizations.organization_id", nullable=True, default=None)
    organization: "Organization" = Relationship(back_populates="language_models")

    # This is the semantic group of the language model, e.g. "gpt-4o-2024-05-13" is in the group "gpt-4o",
    # as are all the LLaMA models. It differs from `family` in that it can dictated by arbitrary business
    # logic, e.g., if we wanted "llama-3" to be a separate group from "llama-2".
    semantic_group: str | None = Field(sa_column=Column(sa.VARCHAR(), nullable=True, index=True))

    # This is the user that created the language model.
    creator_user_id: str = Field(foreign_key="users.user_id", nullable=False)
    language_model_creator: "User" = Relationship(
        back_populates="created_language_models", sa_relationship_kwargs={"remote_side": "User.user_id"}
    )

    ratings: list["Rating"] = Relationship(back_populates="model")
    ratings_history: list["RatingHistory"] = Relationship(back_populates="model")

    provider_id: uuid.UUID | None = Field(foreign_key="providers.provider_id", nullable=True, default=None, index=True)
    provider: "Provider" = Relationship(back_populates="language_models")

    # Model-specific settings, sent to the model provider as part of the request body.
    # Example settings are `provider.ignore`, used by OpenRouter to ignore certain providers.
    provider_settings: dict | None = Field(default_factory=dict, sa_type=sa.JSON)

    turn_qualities: list["TurnQuality"] = Relationship(back_populates="prompt_difficulty_judge_model")

    chat_messages: list["ChatMessage"] = Relationship(back_populates="assistant_language_model")

    language_model_license_id: uuid.UUID | None = Field(
        foreign_key="language_model_licenses.language_model_license_id", nullable=True, default=None
    )
    language_model_license: LanguageModelLicense = Relationship(back_populates="models")
    supported_attachment_mime_types: list[str] | None = Field(
        default=None, sa_column=Column(ARRAY(String), nullable=True)
    )

    # Number of requests used to calculate the metrics. Included here mainly for debugging.
    # This is not used for speed score calculation at runtime.
    num_requests_in_metric_window: int | None = Field(default=None, nullable=True)
    # Average number of tokens in requests used to calculate the metrics. Used for speed score calculation.
    avg_token_count: float | None = Field(default=None, nullable=True)

    # --- Relationships ---
    promotions: list["ModelPromotion"] = Relationship(back_populates="language_model")

    def supports_mime_type(self, mime_type: str) -> bool:
        import re

        if self.supported_attachment_mime_types is None:
            return False
        pattern = "|".join([m.replace("*", ".*") for m in self.supported_attachment_mime_types])
        return re.match(pattern, mime_type) is not None


class EmbeddingModel(BaseModel, table=True):
    """
    Represents an embedding model. These models can be used to generate embeddings
    for various data types (text, etc.). The `dimension` field helps ensure
    that the vector field in related tables has the correct dimensionality.
    """

    __tablename__ = "embedding_models"

    embedding_model_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    name: str = Field(sa_column=Column(sa.String, nullable=False))
    dimension: int = Field(
        sa_column=Column(sa.Integer, nullable=False), gt=0, le=1536
    )  # We only support up to embedding size 1536


# Provider is a service that can be used to access a model, e.g. OpenAI, Anthropic, Together AI, etc.
class Provider(BaseModel, table=True):
    __tablename__ = "providers"

    provider_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # Provider's human-readable name, e.g. "OpenAI", "Together AI".
    name: str = Field(default=None, index=True, unique=True)

    # Provider's base URL for their API, e.g. "https://api.openai.com/v1".
    base_api_url: str = Field(default=None)

    # Environment variable name that contains the API key for the provider.
    # This is used to retrieve the API key at runtime.
    # TODO(arawind): Think of a better way to store secrets.
    api_key_env_name: str = Field(default=None)

    # Whether the provider is active and should be used for new evaluations.
    # Using this instead of deleting, as it is more semantic while temporarily disabling a provider.
    # For permanent removal of the provider, set deleted_at.
    is_active: bool = Field(default=True)

    language_models: list[LanguageModel] = Relationship(back_populates="provider")


# Organization is a group of entities that own the rights to a language model.
class Organization(BaseModel, table=True):
    __tablename__ = "organizations"

    organization_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    organization_name: str = Field(default=None, index=True, unique=True)

    language_models: list[LanguageModel] = Relationship(back_populates="organization")


class RoutingAction(Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    NOOP = "NOOP"

    def opposite(self) -> "RoutingAction":
        return RoutingAction.REJECT if self == RoutingAction.ACCEPT else RoutingAction.ACCEPT

    def noop(self) -> bool:
        return self == RoutingAction.NOOP


# Rules that govern how to route to various models, roughly based on iptables.
class RoutingRule(BaseModel, table=True):
    __tablename__ = "routing_rules"

    routing_rule_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # Whether the rule is active and should be applied.
    is_active: bool = Field(sa_column=Column(Boolean(), server_default="TRUE", nullable=False))

    # The z-index of the rule, used to resolve conflicts. Higher values take precedence.
    z_index: int = Field(sa_column=Column(Integer(), server_default="0"))

    # The category of the source prompt of the form "category" or "*". Categories prepended with
    # "-" means to match the negation of the category, e.g. "-category" matches any prompt that is not
    # in the category "category".
    source_category: str = Field(nullable=False, index=True)

    # A destination of the form "provider/model_name", "provider/*", or "*"
    destination: str = Field(nullable=False, index=True)

    # The destination policy
    target: RoutingAction = Field(default=RoutingAction.ACCEPT)

    # The probability of this rule being applied; (probability * 100)% of the time, this rule is
    # applied; otherwise, the next matching rule is applied.
    probability: float = Field(default=1.0, sa_column=Column(Numeric(precision=10, scale=6), server_default="1.0"))

    __table_args__ = (UniqueConstraint("source_category", "destination", name="uq_cat_dest"),)


class LanguageModelResponseStatusEnum(str, Enum):
    OK = "OK"
    SLOW_RESPONSE = "SLOW_RESPONSE"
    TIMEOUT = "TIMEOUT"
    CONNECTION_REFUSED = "CONNECTION_REFUSED"
    STREAMING_INTERRUPTED = "STREAMING_INTERRUPTED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INFERENCE_SUCCEEDED = "INFERENCE_SUCCEEDED"
    OTHER = "OTHER"

    def is_ok(self) -> bool:
        return self in (LanguageModelResponseStatusEnum.OK, LanguageModelResponseStatusEnum.INFERENCE_SUCCEEDED)


class LanguageModelResponseStatus(BaseModel, table=True):
    __tablename__ = "language_model_response_statuses"

    language_model_response_status_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    language_model_id: uuid.UUID = Field(foreign_key="language_models.language_model_id", nullable=False)
    http_response_code: int | None = Field(sa_column=Column(Integer(), nullable=True))

    status_type: LanguageModelResponseStatusEnum = Field(
        sa_column=Column(
            sa_Enum(LanguageModelResponseStatusEnum),
            nullable=False,
            server_default=LanguageModelResponseStatusEnum.OTHER.name,
            index=True,
        )
    )

    status_message: str | None = Field(sa_column=Column(sa.TEXT(), nullable=True))
