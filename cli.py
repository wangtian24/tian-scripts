
import click

from backend.config import settings
from backend.llm.constants import Provider
from backend.llm.external import prompt_openai


@click.group()
def cli():
    """Main."""
    pass

@cli.command()
@click.option('--prompt', required=True, help="The prompt to send to OpenAI")
@click.option('--api-key', help="API key")
@click.option('--provider', help="LLM provider")
@click.option('--model', default="gpt-4o-mini", help="The provider model to use")
def simple_prompt(prompt: str, api_key: str, provider: Provider, model: str):
    """Send a prompt to OpenAI and get a response."""
    if not api_key:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("An API key should be provided using --api-key or the env variable OPENAI_API_KEY")

    response = prompt_openai(
        api_key=api_key,
        prompt=prompt,
        model=model
    )
    print(response.json(indent=2))

if __name__ == "__main__":
    cli()
