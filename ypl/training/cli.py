import logging

import click
from transformers import TrainingArguments

from ypl.training.data.routing import RoutingCollator, RoutingDataset
from ypl.training.model.routing import RoutingMultilabelClassificationModel
from ypl.training.trainer import RoutingMultilabelTrainer

logging.getLogger().setLevel(logging.INFO)


@click.group()
def cli() -> None:
    """Main."""
    pass


@cli.command()
@click.option("-i", "--input-file", required=True, help="Path to the input TSV file")
@click.option("-m", "--model-name", required=False, default="bert-base-uncased", help="Name of the model to train")
@click.option("--seed", required=False, default=0, help="Random seed")
@click.option("--training-pct", required=False, default=90, help="Percentage of data to use for training")
@click.option("--max-steps", required=False, default=1000, help="Maximum number of steps to train for")
@click.option("-lr", "--learning-rate", required=False, default=5e-5, help="Learning rate")
def train_routing(
    input_file: str,
    model_name: str,
    seed: int,
    training_pct: int,
    max_steps: int,
    learning_rate: float,
) -> None:
    """Train a routing model."""
    dataset = RoutingDataset.from_csv(input_file, sep="\t")
    dataset.set_seed(seed)

    train_dataset, val_dataset = dataset.split(percentage=training_pct)
    model_map = dataset.create_model_map()
    model = RoutingMultilabelClassificationModel(model_name=model_name, model_map=model_map)

    trainer = RoutingMultilabelTrainer(
        args=TrainingArguments(
            output_dir="output",
            max_steps=max_steps,
            optim="schedule_free_adamw",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=learning_rate,
            logging_steps=10,
        ),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=RoutingCollator(tokenizer=model.tokenizer, model_map=model_map),
    )

    trainer.train()


if __name__ == "__main__":
    cli()
