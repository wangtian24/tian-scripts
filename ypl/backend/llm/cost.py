from collections.abc import Callable
from typing import Literal

import tiktoken
from pydantic.v1 import BaseModel as BaseModelV1
from transformers import AutoTokenizer
from vertexai.preview import tokenization


class ModelCost(BaseModelV1):
    dollars_per_million_input_tokens: float
    dollars_per_million_output_tokens: float

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
