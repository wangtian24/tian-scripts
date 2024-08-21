import uuid
from enum import Enum

from sqlalchemy import String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from db.base import BaseModel


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


class LanguageModel(BaseModel):
    __tablename__ = "language_models"

    model_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # This is the name displayed to the user, e.g. "gpt-4o-2024-05-13".
    # This name can be pseudonymous, e.g. "anonymous-model" with internal_name
    # "gpt-4o-2024-05-13". This is useful when Model Providers want to train
    # their models anonymously.
    name: Mapped[str] = mapped_column(String, nullable=False, index=True, unique=True)

    # This is the "real" name of the model as given by the Model Provider,
    # e.g. "gpt-4o-2024-05-13".
    internal_name: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    # This is a human-readable name for the model, e.g. "GPT 4o".
    label: Mapped[str] = mapped_column(String, nullable=True)

    license: Mapped[LicenseEnum] = mapped_column(nullable=False, default=LicenseEnum.unknown)
    family: Mapped[str] = mapped_column(nullable=True)
    avatar_url: Mapped[str] = mapped_column(nullable=True)
