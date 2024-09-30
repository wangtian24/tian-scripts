import asyncio
import functools
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from typing import Any

import click
import git
import numpy as np
from dotenv import load_dotenv
from sqlmodel import Session, select, text
from tqdm.asyncio import tqdm_asyncio

from ypl.backend.config import settings
from ypl.backend.db import get_engine
from ypl.backend.llm.chat import (
    AIMessage,
    ChatProvider,
    HumanMessage,
    JsonChatIO,
    ModelInfo,
    YuppChatIO,
    YuppChatMessageHistory,
    YuppMessageRow,
    compare_llm_responses,
    get_chat_model,
    highlight_llm_similarities,
    highlight_llm_similarities_with_embeddings,
)
from ypl.backend.llm.constants import COSTS_BY_MODEL
from ypl.backend.llm.embedding import get_embedding_model
from ypl.backend.llm.judge import JudgeConfig, YuppEvaluationLabeler, YuppPromptDifficultyLabeler, choose_llm
from ypl.backend.llm.labeler import WildChatRealismLabeler
from ypl.backend.llm.prompt_classifiers import categorize_user_messages
from ypl.backend.llm.ranking import get_default_ranker
from ypl.backend.llm.synthesize import SQLChatIO, SynthesizerConfig, SyntheticUserGenerator, asynthesize_chats
from ypl.backend.llm.utils import fetch_categories_with_descriptions_from_db
from ypl.db.chats import Chat

logging.getLogger().setLevel(logging.INFO)


def db_cmd(f: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        load_dotenv()
        get_approval_on_environment()
        return f(*args, **kwargs)

    return wrapper


def get_approval_on_environment() -> bool:
    if settings.ENVIRONMENT.lower() != "local":
        # If running in a non-interactive environment (cloud run job)
        if not sys.stdin.isatty():
            return True

        print(f"WARNING: Command will be run on the {settings.ENVIRONMENT.upper()} database!")
        approval = input("Type 'yupp' to continue: ").strip().lower()
        if approval != "yupp":
            print("Aborted!")
            exit(1)
    return True


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
@click.option(
    "-s",
    "--sql",
    required=True,
    help="a SQL query to fetch the ids of the chats to dump; "
    "example: 'SELECT chat_id from chats ORDER BY created_at DESC limit 5'",
)
@click.option("-o", "--output-path", required=True, help="The output path of the file")
def chats_to_json(sql: str, output_path: str) -> None:
    messages: list[YuppMessageRow] = []
    writer = JsonChatIO(output_path)
    with Session(get_engine()) as session:
        ids = [str(row[0]) for row in session.exec(text(sql)).all()]  # type: ignore
        chats = session.exec(select(Chat).where(Chat.chat_id.in_(ids))).all()  # type: ignore
        for chat in chats:
            for turn in chat.turns:
                if len(turn.chat_messages) < 3:
                    continue
                prompt, response1, response2 = turn.chat_messages[:3]
                messages.append([HumanMessage(content=prompt.content)])
                messages.append(
                    [
                        AIMessage(content=response1.content),
                        AIMessage(content=response2.content),
                    ]
                )
            writer.append_chat(YuppChatMessageHistory(messages=messages))
    writer.flush()


@cli.command(help="Evaluate the difficulty of the initial prompt of a chat.")
@click.option("-c", "--config", required=True, help="The judge config to use")
@click_provider_option("--provider", help="LLM and embeddings provider")
@click.option("--api-key", help="API key", required=True)
@click.option("--language-model", default="gpt-4o-mini", help="The LLM to use for judging realism")
@click.option("-i", "--input-file", type=str, required=True, help="The JSON file containing conversations")
@click.option("-t", "--truncate-inputs", type=int, default=500, help="Truncate prompts/responses after N chars")
@click.option(
    "-o",
    "--output-file",
    type=str,
    default=None,
    help="The JSON file to write the results to." " Defaults to the input file.",
)
@click.option(
    "-j",
    "--num-parallel",
    default=1,
    help="The number of jobs to run in parallel. Optimal " "value depends on the rate limit and CPU cores.",
)
def judge_yupp_prompt_difficulty(
    input_file: str,
    output_file: str,
    num_parallel: int,
    provider: str,
    api_key: str,
    language_model: str,
    truncate_inputs: int,
) -> None:
    llm = get_chat_model(ModelInfo(provider=provider, model=language_model, api_key=api_key), temperature=0.0)

    async def arun_batch(inputs: list[tuple[str, str, str, str]]) -> list[tuple[str, str]]:
        async def ajudge_yupp_output(id: str, user_msg: str, llm1_msg: str, llm2_msg: str) -> tuple[str, str]:
            async with sem:
                judge = YuppPromptDifficultyLabeler(llm)

                return await judge.alabel((user_msg, llm1_msg, llm2_msg)), id

        sem = asyncio.Semaphore(num_parallel)

        return await tqdm_asyncio.gather(*[ajudge_yupp_output(*x) for x in inputs])  # type: ignore

    output_file = output_file or input_file

    chats = JsonChatIO(input_file).read_chats()
    inputs: list[tuple[str, str, str, str]] = []

    def truncate(s: str) -> str:
        if len(s) < truncate_inputs:
            return s
        # remove the last partially-truncated word
        return re.sub(r"\S+$", "...", s[:truncate_inputs])

    for row_idx, chat in enumerate(chats):
        try:
            chat_id, user_message, llm_responses = chat.initial_prompt_and_responses()
            if chat_id is None:
                raise ValueError(f"Chat {row_idx} has no ID")
            assert len(llm_responses) >= 2
            llm1_response, llm2_response = llm_responses[:2]
            inputs.append((chat_id, truncate(user_message), truncate(llm1_response), truncate(llm2_response)))
        except ValueError:
            logging.exception(f"Error processing chat row {row_idx}")
            raise

    results = asyncio.run(arun_batch(inputs))
    prefix, suffix = "```json\n", "\n```"
    with open(output_file, "w") as outf:
        for res in results:
            judgement_json_str, chat_id = res
            judgement = None
            try:
                judgement = json.loads(judgement_json_str)
            except json.JSONDecodeError:
                if not (judgement_json_str.startswith(prefix) and judgement_json_str.endswith(suffix)):
                    print(f"Can't parse judgement for {chat_id}: {judgement}")
                    continue
                try:
                    judgement = json.loads(judgement_json_str[len(prefix) : len(judgement_json_str) - len(suffix)])
                except json.JSONDecodeError:
                    print(f"Can't parse judgement for {chat_id}: {judgement}")
                    continue
            outf.write(json.dumps(dict(judgement=judgement, chat_id=chat_id)))
            outf.write("\n")


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
                writer = sql_chat_io = SQLChatIO()
                repo = git.Repo(search_parent_directories=True)
                git_commit_sha = repo.head.object.hexsha
                sql_chat_io.populate_backfill_attributes(synth_config, num_chats_per_user, git_commit_sha)

                writer.flush()
            case _:
                raise ValueError(f"Invalid output type: {output_type}")

        for user in user_generator.generate_users():
            chats = await asynthesize_chats(synth_config, user, num_chats_per_user, num_parallel=num_parallel)

            for chat in chats:
                writer.append_chat(chat)

            writer.flush()

    asyncio.run(asynthesize_backfill_data())


@cli.command(help="Converts backfill data from one format to another (e.g., JSON to SQL).")
@click.option("-sc", "--synthesizer-config", required=True, help="The config used for synthesizing YF")
@click.option("-jc", "--judge-config", required=True, help="The config used for judging YF")
@click.option("-it", "--input-type", required=True, type=click.Choice(["json", "db"], case_sensitive=False))
@click.option("-i", "--input-path", help="The path to the input file, if the type is `json`")
@click.option("-ot", "--output-type", required=True, type=click.Choice(["json", "db"], case_sensitive=False))
@click.option("-o", "--output-path", help="The path to the output file, if the type is `json`")
@click.option("-n", "--num-attempted-chats-per-user", default=10, help="The number of attempted chats per user")
def convert_backfill_data(
    synthesizer_config: str,
    judge_config: str,
    input_type: str,
    input_path: str,
    output_type: str,
    output_path: str,
    num_attempted_chats_per_user: int,
) -> None:
    synth_config = SynthesizerConfig.parse_file(synthesizer_config)
    judge_config_ = JudgeConfig.parse_file(judge_config)

    if input_type == "json" and output_type == "db":
        input_io = JsonChatIO(input_path)
        output_io = sql_io = SQLChatIO()
        git_commit_sha = git.Repo(search_parent_directories=True).head.object.hexsha
        sql_io.populate_backfill_attributes(
            synth_config,
            num_attempted_chats_per_user=num_attempted_chats_per_user,
            git_commit_sha=git_commit_sha,
            judge_models=[x.model for x in judge_config_.llms],
            judge_model_temperatures=[x.temperature or 0.0 for x in judge_config_.llms],
        )

        output_io.flush()
    elif input_type == "db" and output_type == "json":
        raise NotImplementedError("Conversion from SQL to JSON is not yet supported.")
    else:
        raise ValueError(f"Unsupported conversion from {input_type} to {output_type}")

    for chat in input_io.read_chats():
        output_io.append_chat(chat)

    output_io.flush()


def _update_ranking(
    category_names: list[str] | None = None,
    exclude_ties: bool = False,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    user_from_date: datetime | None = None,
    user_to_date: datetime | None = None,
    language_codes: list[str] | None = None,
) -> None:
    params = locals()
    ranker = get_default_ranker()
    ranker.add_evals_from_db(**params)
    for ranked_model in ranker.leaderboard():
        logging.info(ranked_model)
    ranker.to_db()


@cli.command()
@click.option("--category-names", multiple=True, help="Categories to include (can be specified multiple times)")
@click.option("--exclude-ties", is_flag=True, help="Exclude ties")
@click.option("--from-date", help="The prompt start date to filter by")
@click.option("--to-date", help="The prompt end date to filter by")
@click.option("--user-from-date", help="The user start date to filter by")
@click.option("--user-to-date", help="The user end date to filter by")
@click.option("--language-codes", multiple=True, help="The language codes to filter by")
def update_ranking(
    category_names: list[str] | None = None,
    exclude_ties: bool = False,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    user_from_date: datetime | None = None,
    user_to_date: datetime | None = None,
    language_codes: list[str] | None = None,
    all_categories: bool = False,
) -> None:
    _update_ranking(category_names, exclude_ties, from_date, to_date, user_from_date, user_to_date, language_codes)


@cli.command()
def update_ranking_all_categories() -> None:
    for category_name in fetch_categories_with_descriptions_from_db():
        logging.info(f"Updating ranking for category: {category_name}")
        _update_ranking(category_names=[category_name])
    logging.info("Updating ranking with no categories")
    _update_ranking()


@cli.command(help="Label the realism of user prompts read in from a JSON file, relative to WildChat")
@click.option("-i", "--json-file", required=True, help="The JSON file to read prompts from")
@click_provider_option("--provider", help="LLM and embeddings provider")
@click.option("--api-key", help="API key", required=True)
@click.option("--language-model", default="gpt-4o-mini", help="The LLM to use for judging realism")
@click.option("--embeddings-model", default="text-embedding-ada-002", help="The embeddings model to use")
@click.option("--limit", default=200, help="The number of prompts to judge")
def label_wildchat_realism(
    json_file: str, provider: str, api_key: str, language_model: str, embeddings_model: str, limit: int
) -> None:
    llm = get_chat_model(ModelInfo(provider=provider, model=language_model, api_key=api_key), temperature=0.0)
    embedding_model = get_embedding_model(ModelInfo(provider=provider, model=embeddings_model, api_key=api_key))
    judge = WildChatRealismLabeler(llm, embedding_model)
    prompts = []

    for idx, line in enumerate(open(json_file)):
        if idx >= limit:
            break

        try:
            prompts.append(json.loads(line)["messages"][0][0]["content"])
        except KeyError:
            continue

    realistic_flags = asyncio.run(judge.abatch_label(prompts))
    counter = Counter(realistic_flags)

    print("Proportion realistic:", counter[True] / (counter[True] + counter[False]))
    print("Sample size:", counter[True] + counter[False])


@cli.command(help="Evaluate pairs of LLM generations read in from a JSON file, acting like a real Yupp user")
@click.option("-c", "--config", required=True, help="The judge config to use")
@click.option("--limit", default=200, help="The number of examples to judge")
@click.option("-i", "--input-file", type=str, required=True, help="The JSON file containing conversations")
@click.option(
    "-o",
    "--output-file",
    type=str,
    default=None,
    help="The JSON file to write the results to." " Defaults to the input file.",
)
@click.option(
    "-j",
    "--num-parallel",
    default=4,
    help="The number of jobs to run in parallel. Optimal " "value depends on the rate limit and CPU cores.",
)
def judge_yupp_llm_outputs(input_file: str, output_file: str, limit: int, config: str, num_parallel: int) -> None:
    async def arun_batch(inputs: list[tuple[int, str, str, str, str, str]]) -> list[tuple[str, int]]:
        async def ajudge_yupp_output(
            row_idx: int, user_msg: str, llm1_msg: str, llm2_msg: str, llm1: str, llm2: str
        ) -> tuple[str, int]:
            async with sem:
                llm_info = choose_llm(cfg.llms, exclude_models={llm1, llm2}, seed=row_idx)
                llm = get_chat_model(llm_info, temperature=0.0)
                judge = YuppEvaluationLabeler(llm)

                return llm_info.model, await judge.alabel((user_msg, llm1_msg, llm2_msg))

        sem = asyncio.Semaphore(num_parallel)

        return await tqdm_asyncio.gather(*[ajudge_yupp_output(*x) for x in inputs])  # type: ignore

    cfg = JudgeConfig.parse_file(config)
    output_file = output_file or input_file

    chats = JsonChatIO(input_file).read_chats()
    coro_inputs: list[tuple[int, str, str, str, str, str]] = []

    for row_idx, chat in enumerate(chats[:limit]):
        try:
            for user_message, llm1_response, llm2_response in chat.triplet_blocks():
                llm1 = chat.eval_llms[0]
                llm2 = chat.eval_llms[1]

                coro_inputs.append(
                    (
                        row_idx,
                        str(user_message.content),
                        str(llm1_response.content),
                        str(llm2_response.content),
                        llm1,
                        llm2,
                    )
                )
        except ValueError:
            logging.exception(f"Error processing chat row {row_idx}")
            raise

    results = asyncio.run(arun_batch(coro_inputs))

    for (chosen_llm, result), (idx, _, _, _, _, _) in zip(results, coro_inputs, strict=False):
        if result is None or result < 0:
            new_result = None
        else:
            new_result = 100 - int(20 * ((result - 1) * 5 / 4))  # inverted to match FE semantics

        chats[idx].judgements += [None, new_result]  # no judgements associated with user messages
        chats[idx].judge_llm = chosen_llm

    chat_io = JsonChatIO(output_file)
    chat_io.write_all_chats(chats)
    chat_io.flush()


@cli.command()
@click.option(
    "--update-all-messages",
    is_flag=True,
    default=False,
    help="Categorize all messages, not just those without a category",
)
@db_cmd
def categorize_messages(update_all_messages: bool) -> None:
    """Categorize user chat messages."""
    categorize_user_messages(update_all_messages)


if __name__ == "__main__":
    cli()
