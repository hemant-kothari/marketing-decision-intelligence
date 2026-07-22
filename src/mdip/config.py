"""Typed configuration loader.

We load ``conf/config.yaml`` into frozen dataclasses. Why dataclasses instead of
passing raw dicts around?
  * Autocomplete + type checking catch typos (``cfg.stats.alpha``, not ``cfg["stats"]["alpah"]``).
  * ``frozen=True`` makes config immutable, so no stage can secretly mutate it mid-run.
  * It reads clearly in an interview: "config is validated and typed at load time."

This is intentionally lightweight — no Hydra, no Pydantic — because a project this
size does not need them, and simpler code is easier to explain and defend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Project root = two levels up from this file (src/mdip/config.py -> project root).
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    raw_data: str
    processed_dir: str
    duckdb: str
    artifacts_dir: str
    figures_dir: str


@dataclass(frozen=True)
class DataCfg:
    treatment_col: str
    control_label: str
    treated_labels: list[str]
    primary_outcome: str
    outcomes: list[str]


@dataclass(frozen=True)
class StatsCfg:
    alpha: float
    n_bootstrap: int
    mde_lift: float


@dataclass(frozen=True)
class LoggingCfg:
    level: str


@dataclass(frozen=True)
class Config:
    project_name: str
    seed: int
    paths: Paths
    data: DataCfg
    stats: StatsCfg
    logging: LoggingCfg

    def path(self, key: str) -> Path:
        """Return an absolute path for a configured relative path (by attribute name)."""
        return ROOT / getattr(self.paths, key)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load and validate the project configuration.

    Args:
        config_path: Optional path to a YAML config. Defaults to ``conf/config.yaml``.

    Returns:
        A frozen, fully-typed :class:`Config`.
    """
    path = Path(config_path) if config_path else ROOT / "conf" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        project_name=raw["project_name"],
        seed=int(raw["seed"]),
        paths=Paths(**raw["paths"]),
        data=DataCfg(**raw["data"]),
        stats=StatsCfg(**raw["stats"]),
        logging=LoggingCfg(**raw["logging"]),
    )
