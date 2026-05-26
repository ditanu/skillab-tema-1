from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    prompt: str
    description: str = ""
    variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptRegistry:
    def __init__(self, folder: str):
        self.folder = folder
        self._templates = self._load(folder)

    def _load(self, folder: str) -> dict[str, PromptTemplate]:
        templates = {}

        for path in Path(folder).rglob("*.yaml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            template = PromptTemplate(**data)
            templates[template.name] = template

        return templates

    def get(self, name: str) -> PromptTemplate:
        if name not in self._templates:
            raise KeyError(f"Promptul '{name}' nu există în registry.")

        return self._templates[name]

    def render(self, name: str, **variables) -> str:
        template = self.get(name)
        return Template(template.prompt).render(**variables)

    def list_templates(self) -> list[str]:
        return list(self._templates.keys())

    def reload(self) -> None:
        self._templates = self._load(self.folder)