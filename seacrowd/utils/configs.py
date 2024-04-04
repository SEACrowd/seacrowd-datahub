from dataclasses import dataclass

import datasets


@dataclass
class SEACrowdConfig(datasets.BuilderConfig):
    """BuilderConfig for SEACrowd."""

    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None
