import asyncio
import concurrent.futures
import functools
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import click
import git
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import and_, case, func, or_
from sqlalchemy.orm import load_only, selectinload
from sqlmodel import Session, select, text
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from ypl.backend.config import settings
from ypl.backend.db import get_async_session, get_engine
from ypl.backend.email.marketing import send_marketing_emails_async, send_monthly_summary_emails_async
from ypl.backend.email.send_email import EmailConfig, send_email_async
from ypl.backend.llm.chat import (
    AIMessage,
    ChatProvider,
    HumanMessage,
    JsonChatIO,
    YuppChatIO,
    YuppChatMessageHistory,
    YuppMessageRow,
)
from ypl.backend.llm.constants import MODEL_HEURISTICS
from ypl.backend.llm.db_helpers import get_chat_model
from ypl.backend.llm.embedding import get_embedding_model
from ypl.backend.llm.judge import (
    JudgeConfig,
    QuickResponseQualityLabeler,
    ResponseRefusalLabeler,
    SpeedAwareYuppEvaluationLabeler,
    YuppEvaluationLabeler,
    YuppMultilabelClassifier,
    YuppOnlinePromptLabeler,
    YuppPromptDifficultyLabeler,
    YuppQualityLabeler,
    YuppSingleDifficultyLabeler,
    choose_llm,
)
from ypl.backend.llm.labeler import SummarizingQuicktakeLabeler, WildChatRealismLabeler
from ypl.backend.llm.model.model_management import validate_active_onboarded_models
from ypl.backend.llm.model.model_onboarding import verify_onboard_submitted_models
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.moderation import LLAMA_GUARD_3_8B_MODEL_NAME, ModerationReason, moderate
from ypl.backend.llm.prompt_classifiers import categorize_user_messages
from ypl.backend.llm.prompt_suggestions import refresh_conversation_starters
from ypl.backend.llm.ranking import get_default_ranker
from ypl.backend.llm.synthesize import SQLChatIO, SynthesizerConfig, SyntheticUserGenerator, asynthesize_chats
from ypl.backend.llm.utils import fetch_categories_with_descriptions_from_db
from ypl.backend.payment.crypto.crypto_payout import process_pending_crypto_rewards
from ypl.backend.payment.crypto.crypto_wallet import create_wallet
from ypl.backend.payment.payment import (
    store_coinbase_retail_wallet_balances,
    store_wallet_balances,
    validate_ledger_balance_all_users,
)
from ypl.backend.payment.payout_utils import validate_pending_cashouts_async
from ypl.backend.payment.plaid.plaid_payout import PlaidPayout, process_plaid_payout
from ypl.backend.utils.analytics import post_analytics_to_slack
from ypl.backend.utils.generate_referral_codes import generate_invite_codes_for_yuppster_async
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import CapabilityType
from ypl.db.chats import (
    Chat,
    ChatMessage,
    LanguageCode,
    MessageType,
    Turn,
    TurnQuality,
    User,
)
from ypl.db.language_models import LanguageModel, Provider
from ypl.db.message_annotations import (
    IS_REFUSAL_ANNOTATION_NAME,
    QUICK_RESPONSE_QUALITY_ANNOTATION_NAME,
    get_turns_to_evaluate,
    update_message_annotations_in_chunks,
)
from ypl.db.oneoffs.reset_points import reset_points
from ypl.db.rewards import RewardActionEnum, RewardAmountRule, RewardProbabilityRule, RewardRule
from ypl.db.users import (
    SYSTEM_USER_ID,
    Capability,
    CapabilityStatus,
    UserCapabilityOverride,
    UserCapabilityStatus,
    UserStatus,
)

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


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

        cost = MODEL_HEURISTICS[model].compute_cost(
            input_string=data.get(input, ""), output_string=data.get(output, "")
        )
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
    writer = JsonChatIO(output_path)
    with Session(get_engine()) as session:
        ids = [str(row[0]) for row in session.exec(text(sql)).all()]  # type: ignore

        query = (
            select(ChatMessage.content, ChatMessage.message_type, ChatMessage.message_id, Turn.turn_id)  # type: ignore
            .join(Turn, ChatMessage.turn_id == Turn.turn_id)
            .join(Chat, Turn.chat_id == Chat.chat_id)
            .where(
                Chat.deleted_at.is_(None),  # type: ignore
                Chat.chat_id.in_(ids),  # type: ignore
                ChatMessage.message_type.in_([MessageType.USER_MESSAGE, MessageType.ASSISTANT_MESSAGE]),  # type: ignore
            )
            .order_by(Turn.turn_id)
        )

        results = session.exec(query).all()

        # Organize the results by turns.
        grouped_results = defaultdict(list)
        for content, message_type, message_id, turn_id in results:
            grouped_results[turn_id].append((content, message_type, message_id))

        for turn_id, message_tuples in grouped_results.items():
            messages: list[YuppMessageRow] = []
            user_messages = [
                (content, message_id)
                for (content, message_type, message_id) in message_tuples
                if message_type == MessageType.USER_MESSAGE
            ]
            assert len(user_messages) == 1, f"turn_id: {turn_id} has {len(user_messages)} user messages"
            prompt = user_messages[0]
            assistant_messages = [
                (content, message_id)
                for (content, message_type, message_id) in message_tuples
                if message_type == MessageType.ASSISTANT_MESSAGE
            ]
            if len(assistant_messages) < 2:
                continue

            messages.append([HumanMessage(content=prompt[0], id=str(prompt[1]))])
            messages.append([AIMessage(content=message[0], id=str(message[1])) for message in assistant_messages])
            writer.append_chat(YuppChatMessageHistory(messages=messages, chat_id=str(turn_id)))
    writer.flush(mode="w")


@cli.command(help="Evaluate the difficulty of the initial prompt of a chat.")
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
    help="The JSON file to write the results to. Defaults to the input file.",
)
@click.option(
    "-j",
    "--num-parallel",
    default=1,
    help="The number of jobs to run in parallel. Optimal value depends on the rate limit and CPU cores.",
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

    async def arun_batch(inputs: list[tuple[str, str, str, str]]) -> list[tuple[tuple[int, str], str]]:
        async def ajudge_yupp_output(
            id: str, user_msg: str, llm1_msg: str, llm2_msg: str
        ) -> tuple[tuple[int, str], str]:
            async with sem:
                judge = YuppPromptDifficultyLabeler(llm)

                return await judge.alabel_full((user_msg, llm1_msg, llm2_msg)), id

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
    with open(output_file, "w") as outf:
        for res in results:
            full_judgement, chat_id = res
            judgement, judgement_details = full_judgement
            outf.write(json.dumps(dict(judgement=judgement, chat_id=chat_id, judgement_details=judgement_details)))
            outf.write("\n")


@cli.command()
@click.option("-i", "--input-path", help="The output path of the file")
@click_provider_option("--provider", help="LLM and embeddings provider")
@click.option("--language-model", default="gpt-4o-mini", help="The LLM to use for judging realism")
def store_prompt_difficulty(input_path: str, provider: str, language_model: str) -> None:
    df = pd.read_json(input_path, lines=True)

    with Session(get_engine()) as session:
        query = (
            select(LanguageModel.language_model_id)
            .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
            .where(
                func.lower(Provider.name) == provider.lower(),
                LanguageModel.internal_name == language_model,
                LanguageModel.deleted_at.is_(None),  # type: ignore
            )
        )
        llm_id = session.exec(query).first()
        if not llm_id:
            raise ValueError(f"Model {language_model} not found")

        for i, row in tqdm(list(df.iterrows())):
            row_id = row["chat_id"]
            prompt_difficulty = float(row["judgement"])
            turn_quality = session.exec(select(TurnQuality).where(TurnQuality.turn_id == row_id)).first()
            if turn_quality is None:
                turn_quality = TurnQuality(
                    turn_id=row_id,
                    prompt_difficulty=prompt_difficulty,
                    prompt_difficulty_judge_model_id=llm_id,
                    prompt_difficulty_details=row["judgement_details"],
                )
            else:
                turn_quality.prompt_difficulty = prompt_difficulty
                turn_quality.prompt_difficulty_judge_model_id = llm_id
                turn_quality.prompt_difficulty_details = row["judgement_details"]
            session.add(turn_quality)
            if i % 500 == 0:
                session.commit()
                print(f"Committed {i} rows")
        session.commit()


@cli.command()
@click.option("-i", "--input-path", required=True, help="The path to the input CSV file")
@click.option("-o", "--output-path", help="The path to the output CSV file", default=None)
@click_provider_option("--provider", help="LLM provider")
@click.option("--api-key", help="API key", required=True)
@click.option("-lm", "--language-model", default="gpt-4o-mini", help="The LLM to use")
@click.option("-j", "--num-parallel", default=16, help="The number of jobs to run in parallel", type=int)
@click.option("-lim", "--limit", default=None, help="The number of prompts to label", type=int)
def label_yupp_online_prompts(
    input_path: str,
    output_path: str | None,
    provider: str,
    api_key: str,
    language_model: str,
    num_parallel: int,
    limit: int | None,
) -> None:
    llm = get_chat_model(ModelInfo(provider=provider, model=language_model, api_key=api_key), temperature=0.0)
    judge = YuppOnlinePromptLabeler(llm)

    output_path = output_path or input_path
    df = pd.read_csv(input_path)
    limit_ = limit or len(df)

    # Needs -1 because .loc includes the endpoint in the slice
    df.loc[: limit_ - 1, "is_online"] = [
        ["offline", "online"][int(x)] if x else None
        for x in asyncio.run(judge.abatch_label(df.loc[: limit_ - 1, "prompt"].tolist(), num_parallel=num_parallel))
    ]
    df.to_csv(output_path, index=False)


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
    asyncio.run(ranker.add_evals_from_db(**params))
    for ranked_model in ranker.leaderboard():
        logging.info(ranked_model)
    if category_names:
        for category_name in category_names:
            ranker.to_db(category_name=category_name)
    else:
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
) -> None:
    _update_ranking(category_names, exclude_ties, from_date, to_date, user_from_date, user_to_date, language_codes)


@cli.command()
def update_ranking_all_categories() -> None:
    for category_name in fetch_categories_with_descriptions_from_db():
        logging.info(f"Updating ranking for category: {category_name}")
        _update_ranking(category_names=[category_name])
    logging.info("Updating ranking with no categories")
    _update_ranking()


@cli.command(help="Generate QuickTakes from a list of prompts")
@click.option("-i", "--json-file", required=True, help="The JSON file to read prompts from")
@click.option("-o", "--output-file", required=True, help="The JSON file to write quicktakes to")
@click_provider_option("--provider", help="LLM and embeddings provider")
@click.option("--api-key", help="API key", required=True)
@click.option("--language-model", default="gpt-4o", help="The LLM to use for generating quicktakes")
@click.option("--limit", default=200, help="The number of prompts to generate quicktakes for")
@click.option("--num-parallel", default=16, help="The number of jobs to run in parallel")
def generate_quicktakes(
    json_file: str, output_file: str, provider: str, api_key: str, language_model: str, limit: int, num_parallel: int
) -> None:
    llm = get_chat_model(ModelInfo(provider=provider, model=language_model, api_key=api_key), temperature=0.0)
    judge = SummarizingQuicktakeLabeler(llm)

    prompts = []
    objects = []

    for idx, line in enumerate(open(json_file)):
        if idx >= limit:
            break

        try:
            data = json.loads(line)
            prompts.append(data["prompt"])
            objects.append(data)
        except KeyError:
            continue

    quicktakes = asyncio.run(judge.abatch_label(prompts, num_parallel=num_parallel))

    with open(output_file, "w") as outf:
        for obj, quicktake in zip(objects, quicktakes, strict=False):
            if not quicktake:
                continue

            obj["quicktake"] = quicktake
            print(json.dumps(obj), file=outf)


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
    help="The JSON file to write the results to. Defaults to the input file.",
)
@click.option(
    "-j",
    "--num-parallel",
    default=4,
    help="The number of jobs to run in parallel. Optimal value depends on the rate limit and CPU cores.",
)
@click.option("--speed-aware", is_flag=True, help="Use speed-aware Yupp evaluation")
@click.option("--no-exclude", is_flag=True, help="Do not exclude the two LLMs from the random selection")
def judge_yupp_llm_outputs(
    input_file: str, output_file: str, limit: int, config: str, num_parallel: int, speed_aware: bool, no_exclude: bool
) -> None:
    async def arun_batch(
        inputs: list[tuple[int, str, str, str, str, str, float | None, float | None]],
    ) -> list[tuple[str, int]]:
        async def ajudge_yupp_output(
            row_idx: int,
            user_msg: str,
            llm1_msg: str,
            llm2_msg: str,
            llm1: str,
            llm2: str,
            time1: float | None = None,
            time2: float | None = None,
        ) -> tuple[str, int]:
            async with sem:
                llm_info = choose_llm(cfg.llms, exclude_models=None if no_exclude else {llm1, llm2}, seed=row_idx)
                llm = get_chat_model(llm_info, temperature=0.0)

                if time1 is not None and time2 is not None:
                    judge1 = SpeedAwareYuppEvaluationLabeler(llm)
                    in1 = (user_msg, llm1_msg, llm2_msg, time1, time2)

                    return llm_info.model, await judge1.alabel(in1)  # for MyPy
                else:
                    judge2 = YuppEvaluationLabeler(llm)
                    in2 = (user_msg, llm1_msg, llm2_msg)

                    return llm_info.model, await judge2.alabel(in2)  # for MyPy

        sem = asyncio.Semaphore(num_parallel)

        return await tqdm_asyncio.gather(*[ajudge_yupp_output(*x) for x in inputs])  # type: ignore

    cfg = JudgeConfig.parse_file(config)
    output_file = output_file or input_file

    chats = JsonChatIO(input_file).read_chats()
    coro_inputs: list[tuple[int, str, str, str, str, str, float | None, float | None]] = []
    default_cost = MODEL_HEURISTICS["gpt-4o-mini"]

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
                        MODEL_HEURISTICS.get(llm1, default_cost).compute_time(output_string=str(llm1_response.content))
                        if speed_aware
                        else None,
                        MODEL_HEURISTICS.get(llm2, default_cost).compute_time(output_string=str(llm2_response.content))
                        if speed_aware
                        else None,
                    )
                )
        except ValueError:
            logging.exception(f"Error processing chat row {row_idx}")
            raise

    results = asyncio.run(arun_batch(coro_inputs))

    for (chosen_llm, result), inp in zip(results, coro_inputs, strict=False):
        idx = inp[0]

        if result is None or result < 0:
            new_result = None
        else:
            new_result = 100 - int(20 * ((result - 1) * 5 / 4))  # inverted to match FE semantics

        chats[idx].judgements += [None, new_result]  # no judgements associated with user messages
        chats[idx].judge_llm = chosen_llm

    chat_io = JsonChatIO(output_file)
    chat_io.write_all_chats(chats)
    chat_io.flush()


@cli.command(help="Evaluate the traits of prompts and LLM responses read in from a JSON file")
@click.option("-c", "--config", required=True, help="The judge config to use")
@click.option("--limit", default=200, help="The number of examples to judge")
@click.option(
    "-i",
    "--input-file",
    type=str,
    required=True,
    help='A file containing conversations of [{"content": "...", "role": "..."}, ...] on each line',
)
@click.option(
    "-o",
    "--output-file",
    type=str,
    default=None,
    help="Defaults to the input file.",
)
@click.option(
    "-j",
    "--num-parallel",
    default=64,
    help="The number of jobs to run in parallel. Optimal value depends on the rate limit and CPU cores.",
)
@click.option("--always-label-difficulty", is_flag=True, help="Always label difficulty")
@click.option("--always-label-categories", is_flag=True, help="Always label categories")
@click.option("--always-label-quality", is_flag=True, help="Always label quality")
@click.option("--no-label-quality", is_flag=True, help="Do not label quality")
@click.option("--no-label-difficulty", is_flag=True, help="Do not label difficulty")
@click.option("--no-label-categories", is_flag=True, help="Do not label categories")
def judge_prompt_traits(
    input_file: str,
    output_file: str,
    limit: int,
    config: str,
    num_parallel: int,
    always_label_difficulty: bool,
    always_label_categories: bool,
    always_label_quality: bool,
    no_label_quality: bool,
    no_label_difficulty: bool,
    no_label_categories: bool,
) -> None:
    lines = Path(input_file).read_text().splitlines()
    batch = []
    orig_batch = []

    for idx, line in enumerate(lines):
        if idx >= limit:
            break

        try:
            data = json.loads(line)
            data = [x for x in data if x["role"] in {"user", "assistant"}]

            if len(data) > 1:
                user_turn = next(x for x in data if x["role"] == "user")
                asst_turn = next(x for x in data if x["role"] == "assistant")
                batch.append((user_turn["content"], asst_turn["content"]))
                orig_batch.append((user_turn, asst_turn))
        except:  # noqa: E722
            continue

    cfg = JudgeConfig.parse_file(config)
    llm_info = cfg.llms[0]
    llm = get_chat_model(llm_info, temperature=0.0)
    labels_list: list[dict[str, Any]] = [{} for _ in orig_batch]

    if "quality" not in orig_batch[0][1] or always_label_quality and not no_label_quality:
        logging.info("Labeling quality...")
        quality_labeler = YuppQualityLabeler(llm, timeout_secs=cfg.timeout)
        quality_results = asyncio.run(quality_labeler.abatch_label(batch, num_parallel=num_parallel))

        for labels, result in zip(labels_list, quality_results, strict=True):
            labels["quality"] = result

    if "difficulty" not in orig_batch[0][1] or always_label_difficulty and not no_label_difficulty:
        logging.info("Labeling difficulty...")
        difficulty_labeler = YuppSingleDifficultyLabeler(llm, timeout_secs=cfg.timeout)
        difficulty_results = asyncio.run(
            difficulty_labeler.abatch_label([x[1] for x in batch], num_parallel=num_parallel)
        )

        for labels, result in zip(labels_list, difficulty_results, strict=True):
            labels["difficulty"] = result

    if "categories" not in orig_batch[0][1] or always_label_categories and not no_label_categories:
        logging.info("Labeling categories...")
        categorizer = YuppMultilabelClassifier(llm, timeout_secs=cfg.timeout)
        category_results = asyncio.run(categorizer.abatch_label([x[1] for x in batch], num_parallel=num_parallel))

        for labels, cat_result in zip(labels_list, category_results, strict=True):
            labels["categories"] = cat_result

    output_file = output_file or input_file

    with Path(output_file).open("w") as f:
        for (orig_user_turn, orig_asst_turn), data, labels in zip(orig_batch, batch, labels_list, strict=True):
            print(
                json.dumps(
                    [
                        {**orig_user_turn, "content": data[0], "role": "user"},
                        {**orig_asst_turn, "content": data[1], "role": "assistant", **labels},
                    ]
                ),
                file=f,
            )


@cli.command()
@click.option(
    "-i",
    "--input-file",
    type=str,
    required=True,
    help="The input file to read from",
)
@click.option(
    "-o",
    "--output-file",
    type=str,
    help="The output file to write to",
)
@click.option(
    "-acp",
    "--attention-check-proportion",
    type=float,
    default=0.1,
)
def create_mturk_prompt_quality_data(
    input_file: str, output_file: str | None, attention_check_proportion: float
) -> None:
    df = pd.read_csv(input_file)
    output_file_ = output_file or input_file
    categories = list(df.category.unique())
    attn_rows = []

    for _ in range(int(attention_check_proportion * len(df))):
        category = random.choice(categories)
        diff = random.choice(["Beginner", "Intermediate", "Expert"])
        rating_gen_fn = functools.partial(random.choice, ["Poor", "Good", "Excellent"])

        attn_rows.append(
            dict(
                prompt=f"This is a {category} prompt with {diff} difficulty.",
                response1=f"{rating_gen_fn()} response",
                response2=f"{rating_gen_fn()} response",
                response3=f"{rating_gen_fn()} response",
            )
        )

    df_new = pd.DataFrame(attn_rows)
    df = pd.concat((df[["prompt", "response1", "response2", "response3"]], df_new))
    df = df.sample(frac=1, replace=False)
    df.to_csv(output_file_, index=False)


@cli.command()
@click.option(
    "-i",
    "--input-file",
    type=str,
    required=True,
    help="The input file to read from",
)
@click.option(
    "-o",
    "--output-file",
    type=str,
    help="The output file to write to",
)
def review_mturk_prompt_quality_data(input_file: str, output_file: str | None) -> None:
    df = pd.read_csv(input_file)
    output_file_ = output_file or input_file
    total_workers = set()
    blocked_workers = set()

    for _, row in df.iterrows():
        prompt = row["Input.prompt"]
        total_workers.add(row["WorkerId"])

        if m := re.match(r"This is a (.+?) prompt with (.+?) difficulty.", prompt):
            category = m.group(1)
            difficulty = m.group(2)

            if category == "nan" or difficulty == "nan":
                continue

            category = category.title()
            cat_response = row[f"Answer.category.{category}.prompt.{category}"]
            diff_response = row[f"Answer.expertiseLevel.{difficulty.capitalize()}.prompt.{difficulty.lower()}"]

            if not cat_response or not diff_response:
                blocked_workers.add(row["WorkerId"])

    c: Counter[str] = Counter()

    for idx, row in df.iterrows():
        if row["WorkerId"] in blocked_workers:
            df.loc[idx, "Reject"] = "x"  # type: ignore[index]
            c["HIT Rejected"] += 1
        else:
            df.loc[idx, "Approve"] = "x"  # type: ignore[index]

    df.to_csv(output_file_, index=False)

    print(f"HIT Rejection Rate: {c['HIT Rejected'] / len(df)}")
    print(f"Worker Rejection Rate: {len(blocked_workers) / len(total_workers)}")
    print("Workers to ban: ", blocked_workers)


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


@cli.command()
@db_cmd
def verify_submitted_models() -> None:
    """Verify and onboard submitted language models."""
    asyncio.run(verify_onboard_submitted_models())


@cli.command()
@db_cmd
def validate_active_models() -> None:
    """Validate active language models."""
    asyncio.run(validate_active_onboarded_models())


@cli.command()
@db_cmd
def post_analytics() -> None:
    """Fetch metrics from Amplitude and post to Slack."""
    asyncio.run(post_analytics_to_slack())


@cli.command()
@click.option(
    "--update-all-messages",
    is_flag=True,
    default=False,
    help="Update language for all messages, not just those without a language code",
)
@db_cmd
def store_prompt_language(update_all_messages: bool) -> None:
    """Detect and store the language of chat messages."""
    from fast_langdetect import detect  # Moved the import here so that the model doesn't download for every command

    CHUNK_SIZE = 100

    with Session(get_engine()) as session:
        query = select(ChatMessage).options(
            load_only(ChatMessage.message_id, ChatMessage.language_code, ChatMessage.content)  # type: ignore
        )
        if not update_all_messages:
            query = query.where(ChatMessage.language_code.is_(None))  # type: ignore
        messages = session.scalars(query).all()
        logging.info(f"Total messages to process: {len(messages)}")
        total_processed = 0
        total_failed = 0

        for i, message in enumerate(messages):
            try:
                input_text = message.content.replace("\n", " ").strip()
                detected_language = detect(input_text)
                lang_code = detected_language["lang"]
                try:
                    message.language_code = LanguageCode(lang_code)
                except ValueError:
                    logging.warning(f"Unsupported language code: {lang_code}")
                total_processed += 1
            except Exception as e:
                logging.error(f"Language detection failed for message {message.message_id}: {e}")
                logging.error(f"Message content: {message.content[:100]}...")
                total_failed += 1

            if (i + 1) % CHUNK_SIZE == 0:
                session.commit()
                logging.info(f"Processed {i + 1} messages. Committed chunk of {CHUNK_SIZE} messages")

        session.commit()
        logging.info("Committed final chunk of messages")
        logging.info(
            f"Language field population complete. "
            f"Total messages processed: {total_processed}. "
            f"Total messages failed: {total_failed}."
        )


@cli.command()
@click.option("--moderation-model", default=LLAMA_GUARD_3_8B_MODEL_NAME, help="The moderation model to use")
@db_cmd
def add_moderation_flags(moderation_model: str) -> None:
    """Add moderation flags to all user messages without them."""

    CHUNK_SIZE = 100
    MAX_MESSAGES_TO_PROCESS = 50000

    def process_message(message_content: str) -> tuple[bool, list[ModerationReason] | None]:
        moderation_result = moderate(message_content, moderation_model)
        return moderation_result.safe, moderation_result.reasons

    completed_messages = 0
    with Session(get_engine()) as session:
        while True:
            query = (
                select(ChatMessage)
                .join(Turn)
                .join(TurnQuality, isouter=True)
                .options(selectinload(ChatMessage.turn).selectinload(Turn.turn_quality))  # type: ignore
                .where(
                    (TurnQuality.prompt_is_safe.is_(None) | (TurnQuality.turn_id.is_(None))),  # type: ignore
                    ChatMessage.message_type == MessageType.USER_MESSAGE,
                )
                .limit(CHUNK_SIZE)
            )

            messages = session.exec(query).all()
            if not messages:
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                future_to_message = {executor.submit(process_message, message.content): message for message in messages}

                for future in concurrent.futures.as_completed(future_to_message):
                    message = future_to_message[future]
                    is_safe, reasons = future.result()

                    turn_quality = message.turn.turn_quality
                    if turn_quality is None:
                        turn_quality = TurnQuality(turn_id=message.turn.turn_id)
                        message.turn.turn_quality = turn_quality

                    turn_quality.prompt_moderation_model_name = moderation_model
                    turn_quality.prompt_is_safe = is_safe
                    if reasons:
                        turn_quality.prompt_unsafe_reasons = reasons
                        print(turn_quality.prompt_unsafe_reasons, message.message_id, message.content)

            session.commit()
            completed_messages += len(messages)
            now = datetime.now().strftime("%H:%M:%S")
            print(f"Committed {len(messages)} rows (total {completed_messages}) at {now}")
            if completed_messages >= MAX_MESSAGES_TO_PROCESS:
                break

    print(f"Finished processing {completed_messages} messages.")


@cli.command()
@click.option("--rules-file", default="data/reward_rules.yml", help="The rules file to use")
@click.option("--dry-run", is_flag=True, default=False, help="Whether to apply the changes")
@db_cmd
def refresh_rewards_rules(rules_file: str, dry_run: bool) -> None:
    """Refresh the rewards rules."""

    with open(rules_file) as f:
        rules = yaml.safe_load(f)

    def parse_action_type(rule: dict[str, Any]) -> dict[str, Any]:
        return {**rule, "action_type": RewardActionEnum(rule["action_type"].lower())}

    amount_rules = [parse_action_type(rule) for rule in rules.get("amount_rules", [])]
    probability_rules = [parse_action_type(rule) for rule in rules.get("probability_rules", [])]

    new_amount_rules = [RewardAmountRule(**rule) for rule in amount_rules]
    new_probability_rules = [RewardProbabilityRule(**rule) for rule in probability_rules]

    with Session(get_engine()) as session:
        existing_amount_rules = session.exec(select(RewardAmountRule)).all()
        existing_probability_rules = session.exec(select(RewardProbabilityRule)).all()

    def matching_existing_rule(new_rule: RewardRule, existing_rules: list[RewardRule]) -> RewardRule | None:
        # Return the first rule that matches the new rule, or None if no match is found.
        return next((rule for rule in existing_rules if rule == new_rule), None)

    existing_rules_to_keep: list[RewardRule] = []
    new_rules_to_add: list[RewardRule] = []
    existing_rules_to_update: list[RewardRule] = []

    for amount_rule in new_amount_rules:
        matching_rule = matching_existing_rule(amount_rule, existing_amount_rules)  # type: ignore[arg-type]
        if matching_rule:
            existing_rules_to_keep.append(matching_rule)
        else:
            new_rules_to_add.append(amount_rule)

    for probability_rule in new_probability_rules:
        matching_rule = matching_existing_rule(probability_rule, existing_probability_rules)  # type: ignore[arg-type]
        if matching_rule:
            existing_rules_to_keep.append(matching_rule)
        else:
            new_rules_to_add.append(probability_rule)

    all_existing_rules = list(existing_amount_rules) + list(existing_probability_rules)
    for rule in all_existing_rules:
        if rule not in existing_rules_to_keep and rule.is_active:
            # We don't want to delete rules, since they may be associated with past rewards; just set them as inactive.
            rule.is_active = False
            existing_rules_to_update.append(rule)

    def names(rules: list[RewardRule]) -> list[str]:
        return [rule.name for rule in rules]

    print(f"Keeping {len(existing_rules_to_keep)} existing rules: {names(existing_rules_to_keep)}")
    print(f"Setting {len(existing_rules_to_update)} existing rules as inactive: {names(existing_rules_to_update)}")
    print(f"Adding {len(new_rules_to_add)} new rules: {names(new_rules_to_add)}")

    if dry_run:
        print("Dry run, not committing changes.")
    else:
        with Session(get_engine()) as session:
            session.add_all(existing_rules_to_update)
            session.add_all(new_rules_to_add)
            session.commit()


@cli.command(help="Annotate LLM refusals")
@click.option("-j", "--num-parallel", default=8, help="The number of jobs to run in parallel.")
@click.option("-n", "--max-num-turns", default=1000, help="The maximum number of unannotated turns to process")
def judge_refusals(
    num_parallel: int,
    max_num_turns: int,
) -> None:
    prompts_responses = []
    response_message_ids = []

    with Session(get_engine()) as session:
        # Get unannotated turns, up to the max number of turns.
        turn_ids = get_turns_to_evaluate(
            session,
            [MessageType.ASSISTANT_MESSAGE, MessageType.QUICK_RESPONSE_MESSAGE],
            IS_REFUSAL_ANNOTATION_NAME,
            max_num_turns,
        )

        # Get the messages for each turn as pairs of prompt and response.
        for turn_id in turn_ids:
            messages = session.exec(
                select(ChatMessage.message_id, ChatMessage.content, ChatMessage.message_type)  # type: ignore
                .where(ChatMessage.turn_id == turn_id)
                .order_by(ChatMessage.created_at)
            ).all()

            prompt = None
            for _, content, message_type in messages:
                if message_type == MessageType.USER_MESSAGE:
                    prompt = content
                    break

            if prompt is None:
                continue

            for message_id, content, message_type in messages:
                if message_type in (MessageType.ASSISTANT_MESSAGE, MessageType.QUICK_RESPONSE_MESSAGE):
                    prompts_responses.append((prompt, content))
                    response_message_ids.append(message_id)

    if not prompts_responses:
        logging.info("No responses to label")
        return

    logging.info(f"Collected {len(prompts_responses)} prompt-response pairs to label")

    judge_llm = get_chat_model(
        ModelInfo(provider=ChatProvider.OPENAI, model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY),
        temperature=0.0,
    )
    labeler = ResponseRefusalLabeler(judge_llm)
    results = asyncio.run(labeler.abatch_label(prompts_responses, num_parallel=num_parallel))

    update_values = [
        {
            "message_id": message_id,
            "key": IS_REFUSAL_ANNOTATION_NAME,
            "value": result,
        }
        for message_id, result in zip(response_message_ids, results, strict=True)
        if result not in (None, labeler.error_value)
    ]

    num_refusals = sum(result == 1 for result in results)
    num_messages = len(results)
    logging.info(f"{num_refusals} refusals found for {num_messages} messages ({num_refusals / num_messages:.2%})")

    update_message_annotations_in_chunks(update_values)


@cli.command(help="Annotate quick response quality")
@click.option("-j", "--num-parallel", default=8, help="The number of jobs to run in parallel.")
@click.option("-n", "--max-num-turns", default=1000, help="The maximum number of unannotated turns to process")
def judge_quick_response_quality(
    num_parallel: int,
    max_num_turns: int,
) -> None:
    def is_qt_refusal(content: Any) -> Any:
        return or_(content == "", func.length(content) > 140)

    prompts = []
    responses = []
    response_message_ids = []
    chat_histories = []

    with Session(get_engine()) as session:
        turns_to_evaluate = get_turns_to_evaluate(
            session,
            [MessageType.QUICK_RESPONSE_MESSAGE],
            QUICK_RESPONSE_QUALITY_ANNOTATION_NAME,
            max_num_turns,
            additional_fields=[Turn.chat_id, Turn.sequence_id],
        )

        # For a given turn, get the chat history up to and including this turn
        for _, chat_id, sequence_id in turns_to_evaluate:  # type: ignore
            messages = session.execute(
                select(  # type: ignore
                    ChatMessage.message_id,
                    case(
                        (
                            and_(
                                (ChatMessage.message_type == MessageType.QUICK_RESPONSE_MESSAGE),  # type: ignore
                                is_qt_refusal(ChatMessage.content),
                            ),
                            "<CANT_ANSWER>",  # QT refusal text
                        ),
                        else_=ChatMessage.content,
                    ).label("content"),
                    ChatMessage.message_type,
                    Turn.sequence_id,
                )
                .join(Turn)
                .where(
                    Turn.chat_id == chat_id,
                    Turn.sequence_id <= int(sequence_id),
                    ChatMessage.message_type.in_([MessageType.USER_MESSAGE, MessageType.QUICK_RESPONSE_MESSAGE]),  # type: ignore
                )
                .order_by(Turn.sequence_id, ChatMessage.created_at)
            ).all()

            if len(messages) < 2:  # Need at least a prompt and response
                continue

            chat_history = []
            for message in messages:
                if message.sequence_id < sequence_id:
                    chat_history.append(
                        {
                            "role": "user" if message.message_type == MessageType.USER_MESSAGE else "assistant",
                            "content": message.content,
                        }
                    )
                elif message.sequence_id == sequence_id:
                    if message.message_type == MessageType.USER_MESSAGE:
                        prompt_msg = message
                    elif message.message_type == MessageType.QUICK_RESPONSE_MESSAGE:
                        response_msg = message

            prompts.append(prompt_msg.content)
            responses.append(response_msg.content)
            response_message_ids.append(response_msg.message_id)
            chat_histories.append(chat_history)

    if not prompts:
        logging.info("No quick responses to label")
        return

    logging.info(f"Found {len(prompts)} prompt-response pairs to evaluate")

    inputs = list(zip(prompts, responses, chat_histories, strict=True))

    judge_llm = get_chat_model(
        ModelInfo(provider=ChatProvider.OPENAI, model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY),
        temperature=0.0,
    )
    labeler = QuickResponseQualityLabeler(judge_llm)
    results = asyncio.run(labeler.abatch_label(inputs, num_parallel=num_parallel))

    update_values = [
        {
            "message_id": message_id,
            "key": QUICK_RESPONSE_QUALITY_ANNOTATION_NAME,
            "value": result,
        }
        for message_id, result in zip(response_message_ids, results, strict=True)
        if result not in (None, labeler.error_value)
    ]

    quality_counts = Counter(results)
    num_messages = len(results)
    logging.info(f"Quality distribution for {num_messages} messages:")
    for quality, count in quality_counts.items():
        if quality not in (None, labeler.error_value):
            logging.info(f"{quality}: {count} ({count / num_messages:.2%})")

    update_message_annotations_in_chunks(update_values)


@cli.command()
@db_cmd
def process_crypto_rewards() -> None:
    """Process pending crypto rewards."""
    asyncio.run(process_pending_crypto_rewards())


@cli.command()
def create_a_wallet() -> None:
    """Create a new wallet."""
    asyncio.run(create_wallet())


@cli.command()
def process_a_plaid_payout() -> None:
    """Make a Plaid payment."""
    payout = PlaidPayout(
        user_id="1",
        user_name="Ansuman Behera",
        amount=Decimal("1.00"),
        account_number="100000000",
        routing_number="121122676",
        account_type="checking",
    )

    asyncio.run(process_plaid_payout(payout))


@cli.command()
def get_the_coinbase_retail_wallet_balance() -> None:
    """Get the balance of a Coinbase wallet."""
    from ypl.backend.payment.coinbase.coinbase_payout import get_coinbase_retail_wallet_account_details

    asyncio.run(get_coinbase_retail_wallet_account_details())


@cli.command()
@click.option("--to-address", required=True, help="The recipient address for the payout")
def process_a_coinbase_retail_payout(to_address: str) -> None:
    """Make a Coinbase retail payout."""

    from uuid import uuid4

    from ypl.backend.payment.coinbase.coinbase_payout import CoinbaseRetailPayout, process_coinbase_retail_payout
    from ypl.db.payments import CurrencyEnum

    payout = CoinbaseRetailPayout(
        user_id="1",
        amount=Decimal("0.001"),
        to_address=to_address,
        currency=CurrencyEnum.ETH,
        payment_transaction_id=uuid4(),
    )
    asyncio.run(process_coinbase_retail_payout(payout))


@cli.command()
@db_cmd
def validate_pending_cashouts() -> None:
    """Validate pending cashouts."""
    asyncio.run(validate_pending_cashouts_async())


@cli.command()
@db_cmd
def generate_invite_codes_for_yuppster() -> None:
    """Generate invite codes for all users with emails ending in specified domain."""
    asyncio.run(generate_invite_codes_for_yuppster_async())


@cli.command()
@click.option("--init-value", default=2500, help="Initial value to reset points to")
@db_cmd
def reset_yuppster_points(init_value: int) -> None:
    """Reset points for Yupp employees to the specified initial value."""
    reset_points(init_value=init_value)


@cli.command()
def calculate_coinbase_signature_for_test() -> None:
    """Calculate the Coinbase webhook signature."""
    from ypl.backend.utils.json import json_dumps
    from ypl.webhooks.routes.v1.coinbase import calculate_coinbase_signature

    async def run() -> None:
        # create a test payload
        msg = b"""{"blockHash":"0x5ad18a709e15238f0dc9f24dd1fdd50402104d047413caa071f5a447c464ac27","blockNumber":"25019532","blockTime":"1736828411","contractAddress":"","cumulativeGasUsed":"27409922","effectiveGasPrice":"8609108","eventType":"transaction","from":"0x0e88a5b4622552edb523f64f16f45a7333cb1c46","gas":"30000","gasPrice":"8609108","gasUsed":"21000","input":"0x","l1Fee":"10058601563","l1FeeScalar":"","l1GasPrice":"1948738794","l1GasUsed":"1600","logsBloom":"0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000","maxFeePerGas":"8609108","maxPriorityFeePerGas":"1000000","network":"base-mainnet","nonce":"7","priorityFeePerGas":"995009","root":"","status":"1","to":"0x87a6e5b175125c9ed273fa3c03c2612da61b436c","transactionHash":"0x7e2566c8dbd26926b1a3d573e16297bcf8f6a11c0ac10af67a606d214e1fa467","transactionIndex":"161","transactionType":"2","type":"2","value":"126380509597800","valueString":"126380509597800","webhookId":"676ce841dd3e6d230439ca52"}"""  # noqa
        calculated_signature = await calculate_coinbase_signature(msg)
        log_dict = {
            "message": "Coinbase Webhook: Calculated signature",
            "calculated_signature": calculated_signature,
            "expected_signature": "75a94d4b8f86672342a83608df7cad88e006477d2f51b23d0ddb7852552bc5d0",
        }
        logging.info(json_dumps(log_dict))

    asyncio.run(run())


@cli.command()
def post_source_account_balances() -> None:
    """Get funding source account balances."""
    from ypl.backend.llm.utils import post_to_slack
    from ypl.backend.payment.coinbase.coinbase_payout import get_coinbase_retail_wallet_account_details
    from ypl.backend.payment.crypto.crypto_wallet import SLACK_WEBHOOK_CASHOUT, get_wallet_balance
    from ypl.backend.payment.payment import store_axis_upi_balance
    from ypl.backend.payment.upi.axis.facilitator import get_balance

    async def format_and_post_balances() -> None:
        #  get self custodial wallet balance
        wallet_data = await get_wallet_balance()
        await store_wallet_balances(wallet_data)

        message = "*Self Custodial Wallet Balance*\n"
        message += "```\n"
        message += "| Asset | Balance |\n"
        message += "|-------|----------|\n"
        for balance in wallet_data["balances"]:
            currency = balance["currency"]
            amount = balance["balance"]
            message += f"| {currency:<5} | {amount:>9.8f} |\n"
        message += "```\n\n"

        # Get offchain balance
        accounts = await get_coinbase_retail_wallet_account_details()
        await store_coinbase_retail_wallet_balances(accounts)

        message += "*Coinbase Retail Wallet Balance*\n"
        message += "```\n"
        message += "| Asset | Balance |\n"
        message += "|-------|----------|\n"
        for currency, details in accounts.items():
            balance = details.get("balance", 0)
            message += f"| {currency:<5} | {balance:>9.8f} |\n"
        message += "```"

        #  Get Axis UPI balance
        balance = await get_balance()
        await store_axis_upi_balance(balance)
        message += "\n*Axis UPI Balance*\n"
        message += "```\n"
        message += f"| Balance | {balance:>9.8f} |\n"
        message += "```"

        await post_to_slack(message, SLACK_WEBHOOK_CASHOUT)

    asyncio.run(format_and_post_balances())


@cli.command()
@click.option("--max-recent-chats", default=20, help="Maximum number of recent chats to consider")
@click.option("--max-turns-per-chat", default=15, help="Maximum turns to include per chat")
@click.option("--max-message-length", default=1000, help="Maximum length of each message")
@click.option("--min-new-chats", default=2, help="Minimum number of new chats required")
@click.option("--num-days-for-user-activity", default=1, help="Limit to users with chats in the last N days")
@click.option("--num-parallel", default=4, help="Number number of users to refresh in parallel")
@db_cmd
def refresh_all_users_conversation_starters(
    max_recent_chats: int,
    max_turns_per_chat: int,
    max_message_length: int,
    min_new_chats: int,
    num_days_for_user_activity: int,
    num_parallel: int,
) -> None:
    """Refresh conversation starters for all users."""

    # Get users to refresh: all active users with at least 1 chat in the last 24 hours
    with Session(get_engine()) as session:
        user_ids = (
            session.execute(
                select(User.user_id)
                .where(User.status == UserStatus.ACTIVE)
                .join(
                    Chat,
                    and_(
                        User.user_id == Chat.creator_user_id,  # type: ignore
                        Chat.created_at >= datetime.now(UTC) - timedelta(days=num_days_for_user_activity),  # type: ignore
                        Chat.deleted_at.is_(None),  # type: ignore
                    ),
                )
                .distinct()
            )
            .scalars()
            .all()
        )

        logging.info(f"Found {len(user_ids)} users to refresh conversation starters for")

        sem = asyncio.Semaphore(num_parallel)

        async def refresh_with_semaphore(user_id: str) -> None:
            async with sem:
                await refresh_conversation_starters(
                    user_id,
                    max_recent_chats=max_recent_chats,
                    max_turns_per_chat=max_turns_per_chat,
                    max_message_length=max_message_length,
                    min_new_chats=min_new_chats,
                )

        async def run_tasks() -> None:
            tasks = [refresh_with_semaphore(user_id) for user_id in user_ids]
            await asyncio.gather(*tasks)

        asyncio.run(run_tasks())
        logging.info(f"Completed refreshing conversation starters for {len(user_ids)} users")


@cli.command()
@click.option("--user-id", required=True, help="The ID of the user to create the override for")
@click.option(
    "--status",
    type=click.Choice(["ENABLED", "DISABLED"], case_sensitive=True),
    required=True,
    help="The status of the override",
)
@click.option("--reason", required=True, help="The reason for creating this override")
@click.option("--first-time-limit", type=int, help="Override for first time cashout limit")
@click.option("--daily-count", type=int, help="Override for daily cashout count limit")
@click.option("--weekly-count", type=int, help="Override for weekly cashout count limit")
@click.option("--monthly-count", type=int, help="Override for monthly cashout count limit")
@click.option("--daily-credits", type=int, help="Override for daily cashout credits limit")
@click.option("--weekly-credits", type=int, help="Override for weekly cashout credits limit")
@click.option("--monthly-credits", type=int, help="Override for monthly cashout credits limit")
@db_cmd
def create_cashout_override(
    user_email: str,
    status: str,
    reason: str,
    first_time_limit: int | None = None,
    daily_count: int | None = None,
    weekly_count: int | None = None,
    monthly_count: int | None = None,
    daily_credits: int | None = None,
    weekly_credits: int | None = None,
    monthly_credits: int | None = None,
) -> None:
    """Create a cashout capability override for a user."""

    async def _create_override() -> None:
        async with get_async_session() as session:
            capability_stmt = select(Capability).where(
                Capability.capability_name == CapabilityType.CASHOUT.value,
                Capability.deleted_at.is_(None),  # type: ignore
                Capability.status == CapabilityStatus.ACTIVE,
            )
            capability = (await session.exec(capability_stmt)).first()

            if not capability:
                log_dict = {
                    "message": "Error: Cashout capability not found for cashout override",
                    "user_email": user_email,
                }
                logging.error(json_dumps(log_dict))
                return

            override_config = {}
            if first_time_limit is not None:
                override_config["first_time_limit"] = first_time_limit
            if daily_count is not None:
                override_config["daily_count"] = daily_count
            if weekly_count is not None:
                override_config["weekly_count"] = weekly_count
            if monthly_count is not None:
                override_config["monthly_count"] = monthly_count
            if daily_credits is not None:
                override_config["daily_credits"] = daily_credits
            if weekly_credits is not None:
                override_config["weekly_credits"] = weekly_credits
            if monthly_credits is not None:
                override_config["monthly_credits"] = monthly_credits

            user = (await session.exec(select(User).where(User.email == user_email))).first()
            if not user:
                log_dict = {
                    "message": "Error: User not found for cashout override",
                    "user_email": user_email,
                }
                logging.error(json_dumps(log_dict))
                return
            user_id = user.user_id
            override = UserCapabilityOverride(
                user_id=user_id,
                capability_id=capability.capability_id,
                creator_user_id=SYSTEM_USER_ID,
                status=UserCapabilityStatus[status],
                reason=reason,
                effective_start_date=datetime.now(UTC),
                override_config=override_config if override_config else None,
            )

            session.add(override)
            await session.commit()

            log_dict = {
                "message": "Successfully created cashout override for user",
                "user_id": user_id,
                "status": status,
                "reason": reason,
                "first_time_limit": str(first_time_limit) if first_time_limit is not None else "None",
                "daily_count": str(daily_count) if daily_count is not None else "None",
                "weekly_count": str(weekly_count) if weekly_count is not None else "None",
                "monthly_count": str(monthly_count) if monthly_count is not None else "None",
                "daily_credits": str(daily_credits) if daily_credits is not None else "None",
                "weekly_credits": str(weekly_credits) if weekly_credits is not None else "None",
                "monthly_credits": str(monthly_credits) if monthly_credits is not None else "None",
            }
            logging.info(json_dumps(log_dict))

    asyncio.run(_create_override())


@cli.command()
@click.option("--account-id", required=True, help="The ID of the Coinbase account")
@click.option("--transaction-id", required=True, help="The ID of the Coinbase transaction")
def get_coinbase_retail_transaction_status(account_id: str, transaction_id: str) -> None:
    """Get the status of a Coinbase retail transaction."""
    from ypl.backend.payment.coinbase.coinbase_payout import get_transaction_status

    txn_status = asyncio.run(get_transaction_status(account_id, transaction_id))
    print(txn_status)


@cli.command()
@click.option("--metric-window-hours", default=24, help="Scan this many hours of chat messages for metrics")
@click.option(
    "--max-requests-in-metric-window",
    default=100,
    help="Consider up to this many most recent chat messages for metrics in the window",
)
@click.option(
    "--min-requests-in-metric-window",
    default=10,
    help="Min threshold to set metrics just based on recent chat messages. Otherwise take "
    + "average of new and existing metrics",
)
@click.option("--dry-run", is_flag=True, help="Do not update the table")
def update_model_metrics(
    metric_window_hours: int, max_requests_in_metric_window: int, min_requests_in_metric_window: int, dry_run: bool
) -> None:
    """Calculates performance metrics for all the models based on recent chat messages"""
    from ypl.backend.llm.db_update_model_metrics import update_active_model_metrics

    update_active_model_metrics(
        metric_window_hours=metric_window_hours,
        max_requests_in_metric_window=max_requests_in_metric_window,
        min_requests_in_metric_window=min_requests_in_metric_window,
        dry_run=dry_run,
    )


@cli.command()
def validate_ledger_balance() -> None:
    """Validate ledger balance for all users."""
    asyncio.run(validate_ledger_balance_all_users())


@cli.command()
@click.option("--dry-run", is_flag=True, help="Print emails that would be sent without actually sending them")
@db_cmd
def send_monthly_summary_emails(dry_run: bool) -> None:
    """Schedule and send monthly summary emails to users.

    Example usage:
        poetry run python -m ypl.cli send-monthly-summary-emails
        poetry run python -m ypl.cli send-monthly-summary-emails --dry-run
    """

    with Session(get_engine()) as session:
        asyncio.run(send_monthly_summary_emails_async(session, dry_run))


@cli.command()
@click.option("--dry-run", is_flag=True, help="Print emails that would be sent without actually sending them")
@db_cmd
def send_marketing_emails(dry_run: bool) -> None:
    """Schedule and send email campaigns to users.

    Example usage:
        poetry run python -m ypl.cli send-marketing-emails --dry-run
    """

    with Session(get_engine()) as session:
        asyncio.run(send_marketing_emails_async(session, dry_run))


@cli.command()
@click.option("--campaign", required=True, help="The campaign to send the email for")
@click.option("--to-address", default="delivered@resend.dev", help="The email address to send the email to")
@click.option("--print-only", is_flag=True, help="Print emails that would be sent without actually sending them")
@db_cmd
def test_send_email(campaign: str, to_address: str, print_only: bool) -> None:
    """Test sending email campaign to test address.

    Example usage:
        poetry run python -m ypl.cli test-send-email --campaign signup
        poetry run python -m ypl.cli test-send-email --campaign sic_availability --print-only
        poetry run python -m ypl.cli test-send-email --campaign sic_availability --to-address spherecollider@gmail.com
    """

    asyncio.run(
        send_email_async(
            EmailConfig(
                campaign=campaign,
                to_address=to_address,
                template_params={
                    "email_recipient_name": "Rumplestiltskin",
                    "referee_name": "Friend of Rumpy",
                    "credits": 100,
                    "unsubscribe_link": "https://gg.yupp.ai/unsubscribe?user_id=asdfzxcv",
                },
            ),
            print_only=print_only,
        )
    )


@cli.command()
def migrate_wallet_credentials() -> None:
    """Migrate wallet credentials."""
    # This is a one time script to migrate wallet credentials from old to new
    # and won't be needed in the future. This is just captured here for incident reference
    from ypl.backend.payment.crypto.crypto_wallet import migrate_wallet_credentials_for_wallet_id

    try:
        old_api_key_name = os.getenv("OLD_CDP_API_KEY_NAME")
        old_api_private_key = os.getenv("OLD_CDP_API_PRIVATE_KEY")
        new_api_key_name = os.getenv("NEW_CDP_API_KEY_NAME")
        new_api_private_key = os.getenv("NEW_CDP_API_PRIVATE_KEY")
        wallet_id = os.getenv("CDP_WALLET_ID")

        if (
            not old_api_key_name
            or not old_api_private_key
            or not new_api_key_name
            or not new_api_private_key
            or not wallet_id
        ):
            raise ValueError("Missing required environment variables")

        asyncio.run(
            migrate_wallet_credentials_for_wallet_id(
                old_api_key_name=old_api_key_name,
                old_api_private_key=old_api_private_key,
                new_api_key_name=new_api_key_name,
                new_api_private_key=new_api_private_key,
                wallet_id=wallet_id,
            )
        )

        log_dict = {
            "message": "Successfully migrated crypto wallet",
        }
        logging.info(json_dumps(log_dict))

    except Exception as e:
        log_dict = {"message": "Failed to migrate crypto wallet", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise


if __name__ == "__main__":
    cli()
