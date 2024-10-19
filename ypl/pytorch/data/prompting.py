import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoTokenizer

from ypl.pytorch.data.base import PandasDataset, TokenizerCollator
from ypl.pytorch.torch_utils import StrTensorDict


class PromptExample(BaseModel):
    prompt: str
    response: str
    model: str = ""
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class ResponseLengthExample(PromptExample):
    response_length: int  # the length of the response in words


class ResponseLengthDataset(PandasDataset[ResponseLengthExample]):
    def __getitem__(self, index: int) -> ResponseLengthExample:
        return self.process(self.df.iloc[index])

    def process(self, example: pd.Series) -> ResponseLengthExample:
        raise NotImplementedError


class ResponseLengthCollator(TokenizerCollator[ResponseLengthExample]):
    """
    Collator for response length examples that prepares batches for model input. This collator encodes prompts and
    maps response lengths to their corresponding indices.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        buckets: list[int] = [26, 60, 99, 149, 203, 267, 343, 445, 597, 2609],  # noqa: B006
    ) -> None:
        super().__init__(tokenizer)
        self.buckets = buckets

    def collate(self, batch: list[ResponseLengthExample]) -> StrTensorDict:
        tokenizer_output = self.tokenizer.batch_encode_plus(
            [example.prompt for example in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
        ).data

        tokenizer_output["response_lengths"] = torch.tensor(
            [self.bucket_index(example.response_length) for example in batch]
        )

        return tokenizer_output  # type: ignore[no-any-return]

    def bucket_index(self, response_length: int) -> int:
        for i, bucket in enumerate(self.buckets):
            if response_length <= bucket:
                return i

        return len(self.buckets) - 1


class YuppResponseLengthDataset(ResponseLengthDataset):
    def post_init(self) -> None:
        self.df = self.df[
            self.df["eval_llms"].apply(lambda x: any(llm.startswith("gpt-4") for llm in x))
            & self.df["messages"].apply(lambda x: len(x) > 1 and len(x[1]) > 1)
        ]

    @classmethod
    def from_file(cls, path: str | Path) -> "YuppResponseLengthDataset":
        df = pd.read_json(path, orient="records", lines=True)
        return cls(df)

    def process(self, example: pd.Series) -> ResponseLengthExample:
        eval_idx = next(i for i, llm in enumerate(example["eval_llms"]) if llm.startswith("gpt-4"))

        return ResponseLengthExample(
            prompt=example["messages"][0][0]["content"],
            response=example["messages"][1][eval_idx]["content"],
            model=example["eval_llms"][eval_idx],
            response_length=len(example["messages"][1][eval_idx]["content"].split()),
        )


class WildChatResponseLengthDataset(
    ResponseLengthDataset, default_invocation="hf", default_invocation_args=("allenai/WildChat",)
):
    def post_init(self) -> None:
        def include_row(row: pd.Series) -> bool:
            conversation = row["conversation"]

            if len(conversation) < 2:
                return False

            content = conversation[0]["content"]

            if pattern.match(content) or len(content.split()) >= 500:
                return False

            return True

        pattern = re.compile(
            (
                r"^.*(midjourney|/imagine|\s\[1\] =|MC:|Natsuki:|\[player\]:|Yuri:|"
                r"Sayori:|Monika:|\n\n[A-z]+:|Day 1|give me a response to ```).*$"
            ),
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )  # this removes a lot of the redundancies known to be in WildChat

        self.df = self.df[self.df["language"] == "English"]
        self.df = self.df[self.df["model"].str.startswith("gpt-4")]

        include_mask = joblib.Parallel(n_jobs=24)(
            joblib.delayed(include_row)(row)
            for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Filtering rows")
        )
        self.df = self.df[include_mask]

    def process(self, example: pd.Series) -> ResponseLengthExample:
        conversation = example["conversation"]
        prompt = ""
        response = ""
        model = example["model"]
        response_length = 0
        has_user = False
        has_asst = False

        for message in conversation:
            if message["role"] == "user" and not has_user:
                prompt = message["content"]
                has_user = True
            elif message["role"] == "assistant" and not has_asst:
                response = message["content"]
                response_length = len(response.split())
                has_asst = True

            if has_user and has_asst:
                break

        return ResponseLengthExample(
            prompt=prompt,
            response=response,
            model=model,
            response_length=response_length,
        )
