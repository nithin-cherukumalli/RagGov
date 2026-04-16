"""Pydantic models for stresslab runtime profiles."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class RuntimeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    llm_base_url: HttpUrl
    embedding_url: HttpUrl
    answer_model: str
    embedding_model: str
    top_k: Annotated[int, Field(gt=0)] = 5
    artifact_dir: str = "stresslab/artifacts"
