from collections.abc import Callable
from typing import Literal

import numpy as np
import tiktoken
from pydantic.v1 import BaseModel as BaseModelV1
from transformers import AutoTokenizer
from vertexai.preview import tokenization


class ModelHeuristics(BaseModelV1):
    dollars_per_million_input_tokens: float
    dollars_per_million_output_tokens: float
    tokens_per_second: float = 0.0  # number of tokens output per second

    # categories mapped to highest difficulty (1-10) the model can handle. "all" means general purpose.
    skills: dict[str, float] = {}

    tokenizer_type: Literal["huggingface", "tiktoken", "google"] = "tiktoken"
    tokenizer_name: str = "gpt-4o-mini"

    def get_tokenizer_counter(self) -> Callable[[str], int]:
        if self.tokenizer_type == "huggingface":
            return lambda string: len(AutoTokenizer.from_pretrained(self.tokenizer_name).encode(string))
        elif self.tokenizer_type == "tiktoken":
            return lambda string: len(tiktoken.encoding_for_model(self.tokenizer_name).encode(string))
        elif self.tokenizer_type == "google":
            tokenizer = tokenization.get_tokenizer_for_model(self.tokenizer_name)
            return lambda string: tokenizer.count_tokens(string).total_tokens
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def estimate_quality(self, category: str | list[str], difficulty: int) -> float:
        """
        Estimates the quality of the model in the given category and difficulty.

        Args:
            category: The category of the prompt.
            difficulty: The difficulty of the prompt.

        Returns:
          the difference between the model's skill in that category and the difficulty, i.e., negative values is the
          amount the model needs to improve to reach the difficulty and positive values is the amount the model
          exceeds the difficulty.
        """
        if isinstance(category, list):
            return float(np.mean([self.estimate_quality(c, difficulty) for c in category]))

        c = category.lower()

        if c in self.skills:
            return self.skills[c] - difficulty
        elif "all" in self.skills:
            return self.skills["all"] - difficulty

        return -difficulty

    def compute_cost(
        self,
        num_input_tokens: int = 0,
        num_output_tokens: int = 0,
        input_string: str = "",
        output_string: str = "",
    ) -> float:
        if num_input_tokens and input_string:
            raise ValueError("Only one of `num_input_tokens` and `input_string` should be provided")
        if num_output_tokens and output_string:
            raise ValueError("Only one of `num_output_tokens` and `output_string` should be provided")

        cost: float = 0.0
        tokenizer_counter = self.get_tokenizer_counter()

        if num_input_tokens:
            cost += num_input_tokens * self.dollars_per_million_input_tokens / 1_000_000
        elif input_string:
            cost += tokenizer_counter(input_string) * self.dollars_per_million_input_tokens / 1_000_000

        if num_output_tokens:
            cost += num_output_tokens * self.dollars_per_million_output_tokens / 1_000_000
        elif output_string:
            cost += tokenizer_counter(output_string) * self.dollars_per_million_output_tokens / 1_000_000

        return cost

    def compute_time(
        self,
        num_input_tokens: int = 0,
        num_output_tokens: int = 0,
        input_string: str = "",
        output_string: str = "",
    ) -> float:
        if num_input_tokens and input_string:
            raise ValueError("Only one of `num_input_tokens` and `input_string` should be provided")
        if num_output_tokens and output_string:
            raise ValueError("Only one of `num_output_tokens` and `output_string` should be provided")

        tokenizer_counter = self.get_tokenizer_counter()
        time: float = 0.0

        if num_input_tokens:
            time += num_input_tokens / self.tokens_per_second
        elif input_string:
            time += tokenizer_counter(input_string) / self.tokens_per_second

        if num_output_tokens:
            time += num_output_tokens / self.tokens_per_second
        elif output_string:
            time += tokenizer_counter(output_string) / self.tokens_per_second

        return time
