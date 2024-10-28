import enum
from dataclasses import dataclass

from together import AsyncTogether, Together
from together.types import ChatCompletionResponse

LLAMA_GUARD_2_8B_MODEL_NAME = "meta-llama/LlamaGuard-2-8b"
LLAMA_GUARD_3_8B_MODEL_NAME = "meta-llama/Meta-Llama-Guard-3-8B"

# TODO(YUP-717): migrate to LangChain/LLMLabeler.


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


@dataclass
class ModerationResult:
    model_name: str | None
    safe: bool
    reasons: list[ModerationReason] | None


def _check_model_name(model_name: str) -> None:
    if model_name not in (LLAMA_GUARD_2_8B_MODEL_NAME, LLAMA_GUARD_3_8B_MODEL_NAME):
        raise ValueError(f"Model {model_name} is not supported")


def moderate(text: str, model_name: str = LLAMA_GUARD_3_8B_MODEL_NAME) -> ModerationResult:
    _check_model_name(model_name)
    response = Together().chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": text}],
        seed=123,
    )
    return _parse_response(response, model_name)


async def amoderate(text: str, model_name: str = LLAMA_GUARD_3_8B_MODEL_NAME) -> ModerationResult:
    _check_model_name(model_name)
    response = await AsyncTogether().chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": text}],
        seed=123,
    )
    return _parse_response(response, model_name)


def _parse_response(response: ChatCompletionResponse, model_name: str) -> ModerationResult:
    content = response.choices[0].message.content
    if content == "safe":
        return ModerationResult(safe=True, reasons=None, model_name=model_name)
    elif content.startswith("unsafe"):
        unsafe_codes = content.split("\n", 1)[1].split(",")
        reasons = [ModerationReason.from_llamaguard_code(code) for code in unsafe_codes]
        return ModerationResult(safe=False, reasons=reasons, model_name=model_name)
    else:
        raise ValueError(f"Unknown response from model {model_name}: {content}")
