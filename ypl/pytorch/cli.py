import ast
import logging

import click
from transformers import Trainer, TrainingArguments

from ypl.pytorch.data.base import ConcatDataset, PandasDataset, TokenizerCollator
from ypl.pytorch.data.categorizer import CategorizerCollator, CategorizerDataset
from ypl.pytorch.data.prompting import ResponseLengthCollator, WildChatResponseLengthDataset, YuppResponseLengthDataset
from ypl.pytorch.data.routing import RoutingCollator, RoutingDataset
from ypl.pytorch.model.base import YuppClassificationModel
from ypl.pytorch.model.categorizer import CategorizerClassificationModel
from ypl.pytorch.model.response_length import TransformerResponseLengthModel
from ypl.pytorch.model.routing import RoutingMultilabelClassificationModel
from ypl.pytorch.trainer import CategorizerTrainer, ResponseLengthTrainer, RoutingMultilabelTrainer

logging.getLogger().setLevel(logging.INFO)


@click.group()
def cli() -> None:
    """Main."""
    pass


def do_simple_classification_training(
    dataset_cls: type[PandasDataset],
    collator_cls: type[TokenizerCollator],
    model_cls: type[YuppClassificationModel],
    trainer_cls: type[Trainer],
    input_file: str,
    model_name: str,
    seed: int,
    training_pct: int,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    output_folder: str,
    load_from: str | None,
    multilabel: bool = False,
) -> None:
    dataset = dataset_cls.from_csv(input_file, sep="\t")
    dataset.set_seed(seed)

    train_dataset, val_dataset = dataset.split(percentage=training_pct)
    label_map = dataset.create_label_map(multilabel=multilabel)
    model = model_cls(model_name=model_name, label_map=label_map, multilabel=multilabel)
    kwargs = {}

    if multilabel:
        pos_weights = dataset.compute_label_pos_weights(label_map)  # type: ignore[attr-defined]
        kwargs["pos_weights"] = pos_weights

    if load_from is not None:
        model = model_cls.from_pretrained(load_from)

    trainer = trainer_cls(
        args=TrainingArguments(
            output_dir="output",
            max_steps=max_steps,
            optim="schedule_free_adamw",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            fp16=True,
        ),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator_cls(tokenizer=model.tokenizer, label_map=label_map, multilabel=multilabel),  # type: ignore[call-arg]
        **kwargs,
    )

    try:
        if load_from is None:
            trainer.train()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected, evaluating model...")

    print(trainer.evaluate(val_dataset, multilabel=multilabel))

    if load_from is None:
        model.save_pretrained(output_folder)


@cli.command()
@click.option("-yi", "--yupp-input-file", required=True, help="Path to the Yupp input file")
@click.option("-m", "--model-name", required=False, default="bert-base-uncased", help="Name of the model to train")
@click.option("--seed", required=False, default=0, help="Random seed")
@click.option("--training-pct", required=False, default=98, help="Percentage of data to use for training")
@click.option("--max-steps", required=False, default=1000, help="Maximum number of steps to train for")
@click.option("-lr", "--learning-rate", required=False, default=5e-5, help="Learning rate")
@click.option("-bsz", "--batch-size", required=False, default=8, help="Batch size")
@click.option("-o", "--output-folder", required=False, default="model", help="Output folder")
@click.option("--load-from", required=False, default=None, help="Path to load model from")
@click.option(
    "--buckets",
    required=False,
    default="[26, 60, 99, 149, 203, 267, 343, 445, 597, 2609]",
    help="Buckets to use for response length, specified as a Python list string.",
    type=str,
)
def train_response_length(
    yupp_input_file: str,
    model_name: str,
    seed: int,
    training_pct: int,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    output_folder: str,
    load_from: str | None,
    buckets: str,
) -> None:
    buckets_ = ast.literal_eval(buckets)
    logging.info("Creating dataset...")
    wc_dataset = WildChatResponseLengthDataset.create_default()
    yupp_dataset = YuppResponseLengthDataset.from_file(yupp_input_file)

    wc_dataset.set_seed(seed)
    yupp_dataset.set_seed(seed)

    logging.info(f"WildChat dataset length: {len(wc_dataset)}")
    logging.info(f"Yupp dataset length: {len(yupp_dataset)}")
    logging.info("Splitting datasets...")
    wc_train_dataset, wc_val_dataset = wc_dataset.split(percentage=training_pct)
    yupp_train_dataset, yupp_val_dataset = yupp_dataset.split(percentage=training_pct)

    train_dataset = ConcatDataset([wc_train_dataset, yupp_train_dataset])
    model = TransformerResponseLengthModel(model_name, buckets=buckets_)
    collator = ResponseLengthCollator(tokenizer=model.tokenizer, buckets=buckets_)

    trainer = ResponseLengthTrainer(
        args=TrainingArguments(
            output_dir="output",
            max_steps=max_steps,
            optim="schedule_free_adamw",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            fp16=True,
            dataloader_num_workers=8,
        ),
        model=model,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    try:
        if load_from is None:
            trainer.train()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected, evaluating model...")

    print(trainer.evaluate(wc_val_dataset))
    print(trainer.evaluate(yupp_val_dataset))
    model.save_pretrained(output_folder)


@cli.command()
@click.option("-i", "--input-file", required=True, help="Path to the input TSV file")
@click.option("-m", "--model-name", required=False, default="bert-base-uncased", help="Name of the model to train")
@click.option("--seed", required=False, default=0, help="Random seed")
@click.option("--training-pct", required=False, default=90, help="Percentage of data to use for training")
@click.option("--max-steps", required=False, default=1000, help="Maximum number of steps to train for")
@click.option("-lr", "--learning-rate", required=False, default=5e-5, help="Learning rate")
@click.option("-bsz", "--batch-size", required=False, default=8, help="Batch size")
@click.option("-o", "--output-folder", required=False, default="model", help="Output folder")
@click.option("--load-from", required=False, default=None, help="Path to load model from")
def train_routing(
    input_file: str,
    model_name: str,
    seed: int,
    training_pct: int,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    output_folder: str,
    load_from: str | None,
) -> None:
    """Train a routing model."""
    do_simple_classification_training(
        dataset_cls=RoutingDataset,
        collator_cls=RoutingCollator,
        model_cls=RoutingMultilabelClassificationModel,
        trainer_cls=RoutingMultilabelTrainer,
        input_file=input_file,
        model_name=model_name,
        seed=seed,
        training_pct=training_pct,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        output_folder=output_folder,
        load_from=load_from,
    )


@cli.command()
@click.option("-i", "--input-file", required=True, help="Path to the input TSV file")
@click.option("-m", "--model-name", required=False, default="bert-base-uncased", help="Name of the model to train")
@click.option("--seed", required=False, default=0, help="Random seed")
@click.option("--training-pct", required=False, default=90, help="Percentage of data to use for training")
@click.option("--max-steps", required=False, default=1000, help="Maximum number of steps to train for")
@click.option("-lr", "--learning-rate", required=False, default=5e-5, help="Learning rate")
@click.option("-bsz", "--batch-size", required=False, default=8, help="Batch size")
@click.option("-o", "--output-folder", required=False, default="model", help="Output folder")
@click.option("--load-from", required=False, default=None, help="Path to load model from")
@click.option("--multilabel", required=False, default=False, help="Whether to use multilabel categorization")
def train_categorizer(
    input_file: str,
    model_name: str,
    seed: int,
    training_pct: int,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    output_folder: str,
    load_from: str | None,
    multilabel: bool,
) -> None:
    """Train a routing model."""
    do_simple_classification_training(
        dataset_cls=CategorizerDataset,
        collator_cls=CategorizerCollator,
        model_cls=CategorizerClassificationModel,
        trainer_cls=CategorizerTrainer,
        input_file=input_file,
        model_name=model_name,
        seed=seed,
        training_pct=training_pct,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        output_folder=output_folder,
        load_from=load_from,
        multilabel=multilabel,
    )


if __name__ == "__main__":
    cli()
