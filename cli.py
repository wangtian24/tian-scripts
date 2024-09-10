import asyncio
import json
from collections import Counter
from typing import Any

import click
import numpy as np

from backend.config import settings
from backend.llm.chat import (
    ChatProvider,
    JsonChatIO,
    ModelInfo,
    YuppChatIO,
    compare_llm_responses,
    get_chat_model,
    highlight_llm_similarities,
    highlight_llm_similarities_with_embeddings,
)
from backend.llm.constants import COSTS_BY_MODEL
from backend.llm.embedding import get_embedding_model
from backend.llm.judge import WildChatRealismJudge
from backend.llm.ranking import get_default_ranker
from backend.llm.synthesize import SynthesizerConfig, SyntheticUserGenerator, asynthesize_chats


def click_provider_option(*args: str, **kwargs: Any | None) -> Any:
    return click.option(
        *args,
        type=click.Choice([provider.name.lower() for provider in ChatProvider], case_sensitive=False),
        default="openai",
        **kwargs,
    )


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
@click_provider_option("--provider", help="LLM provider")
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
@click_provider_option("--provider", help="LLM provider")
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


@cli.command(
    help="Estimate the cost of a given model from a list of JSON objects passed through stdin. Each JSON \
          object should contain input and output strings with the keys `--input` and `--output`, respectively."
)
@click.option("--model", help="The model to use for cost estimation.", default=None)
@click.option("--input", help="The key to use for extracting input strings.", default="input")
@click.option("--output", help="The key to use for extracting output strings.", default="output")
def estimate_cost(model: str | None, input: str, output: str) -> None:
    """
    Estimate the cost of a given model from a list of JSON objects composed of input and output strings with the keys
    `--input` and `--output`, respectively. The JSON objects are read from stdin. If no `--model` is provided, the
    model is taken from the `model` key in the JSON objects.
    """
    costs = []

    for line in click.get_text_stream("stdin"):
        data = json.loads(line)
        model = model or data.get("model")

        if not model:
            raise ValueError("No model provided.")

        cost = COSTS_BY_MODEL[model].compute_cost(input_string=data.get(input, ""), output_string=data.get(output, ""))
        costs.append(cost)

    costs_arr = np.array(costs)
    print(f"Average cost per example: ${np.mean(costs_arr):.2f}")
    print(f"Total cost: ${np.sum(costs_arr):.2f}")


@cli.command()
@click.option("-c", "--config", required=True, help="The configuration used for Yuppfill")
@click.option("-t", "--output-type", default="json", type=click.Choice(["json", "db"], case_sensitive=False))
@click.option("-o", "--output-path", help="The output path of the file")
@click.option("-j", "--num-parallel", default=16, help="The number of jobs to run in parallel")
@click.option("-n", "--num-chats-per-user", default=10, help="The number of chats to generate per user")
def synthesize_backfill_data(
    config: str, output_type: str, output_path: str, num_parallel: int, num_chats_per_user: int
) -> None:
    async def asynthesize_backfill_data() -> None:
        synth_config = SynthesizerConfig.parse_file(config)
        user_generator = SyntheticUserGenerator(synth_config)
        writer: YuppChatIO | None = None

        match output_type:
            case "json":
                writer = JsonChatIO(output_path)
            case "db":
                raise NotImplementedError("Database output is not implemented yet")
            case _:
                raise ValueError(f"Invalid output type: {output_type}")

        for user in user_generator.generate_users():
            chats = await asynthesize_chats(synth_config, user, num_chats_per_user, num_parallel=num_parallel)

            for chat in chats:
                writer.append_chat(chat)

            writer.flush()

    asyncio.run(asynthesize_backfill_data())


@cli.command()
@click.option("--category", multiple=True, help="Category to include (can be specified multiple times)")
@click.option("--exclude-ties", is_flag=True, help="Exclude ties")
@click.option("--language", help="Language")
@click.option("--model-names", multiple=True, help="Model name (can be specified multiple times)")
def update_ranking(
    category: list[str] | None = None,
    exclude_ties: bool = False,
    language: str | None = None,
    model_names: list[str] | None = None,
) -> None:
    ranker = get_default_ranker()
    ranker.add_evals_from_db(
        category_names=category,
        exclude_ties=exclude_ties,
        language=language,
        model_names=model_names,
    )
    for ranked_model in ranker.leaderboard():
        print(ranked_model)
    ranker.to_db()


@cli.command(help="Judge the realism of user prompts read in from a JSON file, relative to WildChat")
@click.option("-i", "--json-file", required=True, help="The JSON file to read prompts from")
@click_provider_option("--provider", help="LLM and embeddings provider")
@click.option("--api-key", help="API key", required=True)
@click.option("--language-model", default="gpt-4o-mini", help="The LLM to use for judging realism")
@click.option("--embeddings-model", default="text-embedding-ada-002", help="The embeddings model to use")
@click.option("--limit", default=200, help="The number of prompts to judge")
def judge_wildchat_realism(
    json_file: str, provider: str, api_key: str, language_model: str, embeddings_model: str, limit: int
) -> None:
    llm = get_chat_model(ModelInfo(provider=provider, model=language_model, api_key=api_key), temperature=0.0)
    embedding_model = get_embedding_model(ModelInfo(provider=provider, model=embeddings_model, api_key=api_key))
    judge = WildChatRealismJudge(llm, embedding_model)
    prompts = []

    for idx, line in enumerate(open(json_file)):
        if idx >= limit:
            break

        try:
            prompts.append(json.loads(line)["messages"][0][0]["content"])
        except KeyError:
            continue

    realistic_flags = asyncio.run(judge.abatch_judge(prompts))
    counter = Counter(realistic_flags)

    print("Proportion realistic:", counter[True] / (counter[True] + counter[False]))
    print("Sample size:", counter[True] + counter[False])


if __name__ == "__main__":
    cli()
