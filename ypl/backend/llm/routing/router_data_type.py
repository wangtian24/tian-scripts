from pydantic import BaseModel, Field


class PreferredModel(BaseModel):
    models: list[str] = Field(description="List of models presented to the user for a given turn.")
    preferred: str | None = Field(description="Which model was preferred by the user, or None if all are bad")


class RoutingPreference(BaseModel):
    turns: list[PreferredModel] | None = Field(
        description=(
            "The preference for each of the past turns in the chat context "
            "in chronological order (first turn is the oldest). "
            "An empty list indicates that there were no prior turns."
        )
    )
