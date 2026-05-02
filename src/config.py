from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource


class ModelConfig(BaseModel):
    path: str
    conf_threshold: float
    nms_threshold: float


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(toml_file="config.toml")

    model: ModelConfig

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (TomlConfigSettingsSource(settings_cls),)
