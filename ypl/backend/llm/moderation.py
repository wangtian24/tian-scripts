import enum
from dataclasses import dataclass
from functools import cache

from together import Together

LLAMA_GUARD_2_8B_MODEL_NAME = "meta-llama/LlamaGuard-2-8b"
LLAMA_GUARD_3_8B_MODEL_NAME = "meta-llama/Meta-Llama-Guard-3-8B"


class ModerationReason(enum.Enum):
    OTHER = "Other"
    VIOLENT_CRIMES = "Violent Crimes"
    NON_VIOLENT_CRIMES = "Non-Violent Crimes"
    SEX_CRIMES = "Sex Crimes"
    CHILD_EXPLOITATION = "Child Exploitation"
    DEFAMATION = "Defamation"
    SPECIALIZED_ADVICE = "Specialized Advice"
    PRIVACY = "Privacy"
    INTELLECTUAL_PROPERTY = "Intellectual Property"
    INDISCERNATE_WEAPONS = "Indiscriminate Weapons"
    HATE = "Hate"
    SELF_HARM = "Self-Harm"
    SEXUAL_CONTENT = "Sexual Content"
    ELECTIONS = "Elections"

    @classmethod
    def from_llamaguard_code(cls, code: str) -> "ModerationReason":
        """Get ModerationReason from LlamaGuard category code."""
        mapping = {
            "S1": cls.VIOLENT_CRIMES,
            "S2": cls.NON_VIOLENT_CRIMES,
            "S3": cls.SEX_CRIMES,
            "S4": cls.CHILD_EXPLOITATION,
            "S5": cls.DEFAMATION,
            "S6": cls.SPECIALIZED_ADVICE,
            "S7": cls.PRIVACY,
            "S8": cls.INTELLECTUAL_PROPERTY,
            "S9": cls.INDISCERNATE_WEAPONS,
            "S10": cls.HATE,
            "S11": cls.SELF_HARM,
            "S12": cls.SEXUAL_CONTENT,
            "S13": cls.ELECTIONS,
        }
        return mapping.get(code, cls.OTHER)


@cache
def get_together_client() -> Together:
    return Together()


@dataclass
class ModerationResult:
    model_name: str | None
    safe: bool
    reasons: list[ModerationReason] | None


def moderate(text: str, model_name: str = LLAMA_GUARD_3_8B_MODEL_NAME) -> ModerationResult:
    if model_name not in (LLAMA_GUARD_2_8B_MODEL_NAME, LLAMA_GUARD_3_8B_MODEL_NAME):
        raise ValueError(f"Model {model_name} is not supported")

    response = get_together_client().chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": text}],
        seed=123,
    )
    content = response.choices[0].message.content
    if content == "safe":
        return ModerationResult(safe=True, reasons=None, model_name=model_name)
    elif content.startswith("unsafe"):
        unsafe_codes = content.split("\n", 1)[1].split(",")
        reasons = [ModerationReason.from_llamaguard_code(code) for code in unsafe_codes]
        return ModerationResult(safe=False, reasons=reasons, model_name=model_name)
    else:
        raise ValueError(f"Unknown response from model {model_name}: {content}")
