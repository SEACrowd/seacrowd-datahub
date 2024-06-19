from pathlib import Path
from typing import List

import datasets
import pandas as pd
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# No paper citation found.
_CITATION = """\
@article{bai2022training,
  title={Training a helpful and harmless assistant with reinforcement learning from human feedback},
  author={Bai, Yuntao and Jones, Andy and Ndousse, Kamal and Askell, Amanda and Chen, Anna and DasSarma, Nova and Drain, Dawn and Fort, Stanislav and Ganguli, Deep and Henighan, Tom and others},
  journal={arXiv preprint arXiv:2204.05862},
  year={2022}
}

"""

_LOCAL = False
_LANGUAGES = ["tha"]
_DATASETNAME = "thai_hh_rlhf"
_DESCRIPTION = """\
This is a Thai-translated dataset based on Anthropic/hh-rlhf using Google Cloud Translation. This repository
provides access to 161K Train dataset Anthropic/hh-rlhf (Thai-translated).

The original repository provides access to two different kinds of data:

1. Human preference data about helpfulness and harmlessness from Training a Helpful and Harmless Assistant with
Reinforcement Learning from Human Feedback. These data are meant to train preference (or reward) models for
subsequent RLHF training. These data are not meant for supervised training of dialogue agents. Training dialogue
agents on these data is likely to lead to harmful models and this shold be avoided.

2. Human-generated and annotated red teaming dialogues from Red Teaming Language Models to Reduce Harms: Methods,
Scaling Behaviors, and Lessons Learned. These data are meant to understand how crowdworkers red team models and
what types of red team attacks are succesful or not. The data are not meant for fine-tuning or preference modeling
(use the data above for preference modeling). These data are entire transcripts of conversations that are derived
from the harmlessness preference modeling data described above, where only the chosen response is incorporated into
the overall transcript. Furthermore, the transcripts are annotated with human and automated measurements of how
harmful the overall dialogues are.

The translated data corresponds to the first kind of data.
"""

_HOMEPAGE = "	https://huggingface.co/datasets/Thaweewat/hh-rlhf-th"
_LICENSE = Licenses.MIT.value
_URL = "https://huggingface.co/datasets/Thaweewat/hh-rlhf-th/resolve/main/hh-rlhf-train-161k-th.parquet"
_SUPPORTED_TASKS = [Tasks.REINFORCEMENT_LEARNING_WITH_HUMAN_FEEDBACK]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ThaiHhRlhfDataset(datasets.GeneratorBasedBuilder):
    """Thai_HH_RLHF Dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "text"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="Thai_HH_RLHF source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description="Thai_HH_RLHF SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "chosen": datasets.Value("string"),
                    "rejected": datasets.Value("string"),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text.features(label_names=["rejected", "chosen"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_file = Path(dl_manager.download_and_extract(_URL))
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_file})]

    def _generate_examples(self, filepath: Path):
        """Yield examples as (key, example) tuples"""
        df = pd.read_parquet(filepath)
        for idx, row in df.iterrows():
            if self.config.schema == "source":
                example = {"chosen": row.get("chosen"), "rejected": row.get("rejected")}

                yield idx, example

            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                for i, label in enumerate(["chosen", "rejected"]):
                    text = row.get(label)

                    example = {"id": str(idx), "text": text, "label": label}

                    yield (idx * 2) + i, example
