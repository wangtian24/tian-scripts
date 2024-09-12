import uuid
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import BigInteger, Column, Numeric, UniqueConstraint
from sqlmodel import Field, Relationship

from db.base import BaseModel

if TYPE_CHECKING:
    from db.ratings import Rating, RatingHistory


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


# Associates a language model with a provider. A language model can be accessed through multiple providers.
class LanguageModelProviderAssociation(BaseModel, table=True):
    __tablename__ = "language_model_provider_associations"

    language_model_id: uuid.UUID = Field(primary_key=True, foreign_key="language_models.language_model_id")
    provider_id: uuid.UUID = Field(primary_key=True, foreign_key="providers.provider_id")

    # Whether the model should be accessed via the provider.
    # Using this instead of deleting, as it is more semantic while temporarily disabling a model on a provider.
    # For permanent removal of the model on a provider, set deleted_at.
    is_active: bool = Field(default=True)

    # Input cost in USD per million tokens, stored with 2 decimal places.
    input_cost_usd_per_million_tokens: Decimal | None = Field(
        sa_column=Column(Numeric(precision=10, scale=2), nullable=True), default=None
    )
    # Output cost in USD per million tokens, stored with 2 decimal places.
    output_cost_usd_per_million_tokens: Decimal | None = Field(
        sa_column=Column(Numeric(precision=10, scale=2), nullable=True), default=None
    )


class LanguageModel(BaseModel, table=True):
    __tablename__ = "language_models"

    language_model_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # This is the name displayed to the user, e.g. "gpt-4o-2024-05-13".
    # This name can be pseudonymous, e.g. "anonymous-model" with internal_name
    # "gpt-4o-2024-05-13". This is useful when Model Providers want to train
    # their models anonymously.
    name: str = Field(index=True, unique=True)

    # This is the "real" name of the model as given by the Model Provider,
    # e.g. "gpt-4o-2024-05-13".
    internal_name: str = Field(sa_column=Column("internal_name", sa.VARCHAR(), nullable=False))
    # Forcing the pre-convention constraint name for backwards compatibility.
    __table_args__ = (UniqueConstraint("internal_name", name="language_models_internal_name_key"),)

    # This is a human-readable name for the model, e.g. "GPT 4o".
    label: str | None = Field(default=None)

    license: LicenseEnum = Field(default=LicenseEnum.unknown)
    family: str | None = Field(default=None)
    avatar_url: str | None = Field(default=None)

    # This is the number of parameters in the model.
    parameter_count: int | None = Field(sa_column=Column(BigInteger(), nullable=True), default=None)

    # This is the context window of the model.
    context_window_tokens: int | None = Field(sa_column=Column(BigInteger(), nullable=True), default=None)

    # This is the knowledge cutoff of the model in yyyy mm dd format.
    # For example, a knowledge cutoff of 2024 06 15 means the model was trained on data up to June 15, 2024.
    knowledge_cutoff_date: date | None = Field(default=None, nullable=True)

    # This is the organization that owns the language model.
    organization_id: uuid.UUID | None = Field(foreign_key="organizations.organization_id", nullable=True, default=None)
    organization: "Organization" = Relationship(back_populates="language_models")

    ratings: list["Rating"] = Relationship(back_populates="model")
    ratings_history: list["RatingHistory"] = Relationship(back_populates="model")

    providers: list["Provider"] = Relationship(
        back_populates="language_models", link_model=LanguageModelProviderAssociation
    )


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

    language_models: list[LanguageModel] = Relationship(
        back_populates="providers", link_model=LanguageModelProviderAssociation
    )


# Organization is a group of entities that own the rights to a language model.
class Organization(BaseModel, table=True):
    __tablename__ = "organizations"

    organization_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    organization_name: str = Field(default=None, index=True, unique=True)

    language_models: list[LanguageModel] = Relationship(back_populates="organization")
