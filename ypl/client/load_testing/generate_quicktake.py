import argparse
import json
import random
from argparse import Namespace
from typing import Any

from locust import HttpUser, between, events, task
from locust.env import Environment
from ypl.backend.config import settings

EXAMPLE_PROMPTS = [
    "When was Albert Einstein born?",
    "What is the capital of France?",
    "What is the weather in San Francisco?",
    "Generate a 1000-word essay on the history of the internet.",
    "What is the meaning of life?",
    "Draw a picture of a cat.",
]

LOADED_PROMPTS = EXAMPLE_PROMPTS


@events.init_command_line_parser.add_listener
def _(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt-file", type=str, default=None, help="JSONL containing prompts in OpenAI format")
    parser.add_argument("--chat-id", type=str, required=True)


@events.init.add_listener
def on_locust_init(environment: Environment, **kwargs: Any) -> None:
    options: Namespace = environment.parsed_options  # type: ignore

    if options.prompt_file:
        with open(options.prompt_file) as f:
            global LOADED_PROMPTS
            LOADED_PROMPTS = [json.loads(line)["content"] for line in f]


class QuickTakeUser(HttpUser):
    wait_time = between(10, 20)

    @task
    def generate_quicktake(self) -> None:
        chat_id = self.environment.parsed_options.chat_id
        prompt = random.choice(LOADED_PROMPTS)

        self.client.post(
            f"/api/v1/chats/{chat_id}/generate_quicktake",
            json={"prompt": prompt},
            headers={"X-API-KEY": settings.X_API_KEY},
        )
