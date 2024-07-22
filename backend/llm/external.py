from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from backend import prompts


def prompt_openai(api_key: str, model: str = "gpt-4o-mini", prompt: str = "") -> BaseMessage:
    sample_responses = {
        "prompt": "What is the difference between marriage license and marriage certificate?",
        "response_a": "A marriage license is a legal document that allows a couple to get married. It is issued by a government agency, such as a county clerk's office or a state government, and is valid for a certain period of time, usually one year. After the marriage has taken place, the couple must obtain a marriage certificate, which is a document that records the marriage and is used to prove that the marriage took place. The marriage certificate is usually issued by the same government agency that issued the marriage license, and it is typically used for legal purposes, such as to change a name on a driver's license or to prove that a couple is married when applying for government benefits.",  # noqa
        "response_b": "A marriage license and a marriage certificate are two different legal documents that have separate purposes.\n\n1. Marriage License: A marriage license is a legal document that gives a couple permission to get married. It's usually obtained from local government or court officials before the wedding ceremony takes place. The couple is required to meet certain criteria, such as being of a certain age or not being closely related. Once the license is issued, there's often a waiting period before the marriage ceremony can take place. The marriage license has to be signed by the couple, their witnesses, and the officiant conducting the marriage ceremony, then returned to the license issuer for recording.\n\n2. Marriage Certificate: A marriage certificate, on the other hand, is a document that proves a marriage has legally taken place. It's issued after the marriage ceremony, once the signed marriage license has been returned and recorded. The marriage certificate includes details about the couple, like their names, the date and location of their wedding, and the names of their witnesses. This document serves as the official record of the marriage and is often needed for legal transactions like changing a name, adding a spouse to insurance, or proving marital status.",  # noqa
    }
    ai = ChatOpenAI(api_key=SecretStr(api_key), model=model)
    messages = [
        SystemMessage(content=prompts.COMPARE_RESPONSES_SYSTEM_PROMPT),
        HumanMessage(content=str(sample_responses)),
    ]
    return ai.invoke(messages)

