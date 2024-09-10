import json

import click

from backend.config import settings
from backend.llm.chat import (
    ChatProvider,
    compare_llm_responses,
    highlight_llm_similarities,
    highlight_llm_similarities_with_embeddings,
)
from backend.llm.ranking import get_ranker, init_ranking

SAMPLE: dict = {
    "prompt": "What is the difference between marriage license and marriage certificate?",
    "responses": {
        "response_a": "A marriage license is a legal document that allows a couple to get married. It is issued by a government agency, such as a county clerk's office or a state government, and is valid for a certain period of time, usually one year. After the marriage has taken place, the couple must obtain a marriage certificate, which is a document that records the marriage and is used to prove that the marriage took place. The marriage certificate is usually issued by the same government agency that issued the marriage license, and it is typically used for legal purposes, such as to change a name on a driver's license or to prove that a couple is married when applying for government benefits.",  # noqa
        "response_b": "A marriage license and a marriage certificate are two different legal documents that have separate purposes.\n\n1. Marriage License: A marriage license is a legal document that gives a couple permission to get married. It's usually obtained from local government or court officials before the wedding ceremony takes place. The couple is required to meet certain criteria, such as being of a certain age or not being closely related. Once the license is issued, there's often a waiting period before the marriage ceremony can take place. The marriage license has to be signed by the couple, their witnesses, and the officiant conducting the marriage ceremony, then returned to the license issuer for recording.\n\n2. Marriage Certificate: A marriage certificate, on the other hand, is a document that proves a marriage has legally taken place. It's issued after the marriage ceremony, once the signed marriage license has been returned and recorded. The marriage certificate includes details about the couple, like their names, the date and location of their wedding, and the names of their witnesses. This document serves as the official record of the marriage and is often needed for legal transactions like changing a name, adding a spouse to insurance, or proving marital status.",  # noqa
    },
}


@click.group()
def cli() -> None:
    """Main."""
    pass


def _set_api_key(api_key: str) -> str:
    if not api_key:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("An API key should be provided using --api-key or the env variable OPENAI_API_KEY")
    return api_key


@cli.command()
@click.option("--prompt", required=True, help="The prompt to send to OpenAI")
@click.option("--api-key", help="API key")
@click.option(
    "--provider",
    type=click.Choice([provider.name.lower() for provider in ChatProvider], case_sensitive=False),
    default="openai",
    help="LLM provider",
)
@click.option("--model", default="gpt-4o-mini", help="The provider model to use")
def compare_responses(prompt: str, api_key: str, provider: str, model: str) -> None:
    api_key = _set_api_key(api_key)

    response = compare_llm_responses(
        provider=provider,
        api_key=api_key,
        model=model,
        prompt=SAMPLE["prompt"],
        responses=SAMPLE["responses"],
    )
    print(response.json(indent=2))


@cli.command()
@click.option("--api-key", help="API key")
@click.option(
    "--provider",
    type=click.Choice([provider.name.lower() for provider in ChatProvider], case_sensitive=False),
    default="openai",
    help="LLM provider",
)
@click.option("--model", default="gpt-4o-mini", help="The provider model to use")
def highlight_similarities(api_key: str, provider: str, model: str) -> None:
    api_key = _set_api_key(api_key)

    response = highlight_llm_similarities(
        provider=provider,
        api_key=api_key,
        model=model,
        responses=SAMPLE["responses"],
    )
    print(response.content)


@cli.command()
def highlight_similarities_embeddings() -> None:
    response = highlight_llm_similarities_with_embeddings(
        response_a=SAMPLE["responses"]["response_a"],
        response_b=SAMPLE["responses"]["response_b"],
    )
    print(json.dumps(response, indent=2))


@cli.command()
def update_ranking() -> None:
    init_ranking()
    ranker = get_ranker()
    print(ranker.leaderboard())
    ranker.to_db()


if __name__ == "__main__":
    cli()
