import logging

import click

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
def train_routing(input_file: str, model_name: str, seed: int, training_pct: int) -> None:
    """Train a routing model."""
    dataset = RoutingDataset.from_csv(input_file, sep="\t")
    dataset.set_seed(seed)

    train_dataset, val_dataset = dataset.split(percentage=training_pct)
    model_map = dataset.create_model_map()
    model = RoutingMultilabelClassificationModel(model_name=model_name, model_map=model_map)

    trainer = RoutingMultilabelTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=RoutingCollator(tokenizer=model.tokenizer, model_map=model_map),
    )

    trainer.train()


if __name__ == "__main__":
    cli()
