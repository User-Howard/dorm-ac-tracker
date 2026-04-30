from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource


class ModelConfig(BaseModel):
    path: str
    conf_threshold: float
    nms_threshold: float


class ZoneConfig(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(toml_file="config.toml")

    model: ModelConfig
    zone: ZoneConfig

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (TomlConfigSettingsSource(settings_cls),)
