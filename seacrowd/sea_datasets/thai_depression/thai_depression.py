import json
from pathlib import Path
from typing import List

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import DEFAULT_SEACROWD_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Licenses, Tasks

_DATASETNAME = "thai_depression"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

_LANGUAGES = ["tha"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{hamalainen-etal-2021-detecting,
    title = "Detecting Depression in Thai Blog Posts: a Dataset and a Baseline",
    author = {H{\"a}m{\"a}l{\"a}inen, Mika  and
      Patpong, Pattama  and
      Alnajjar, Khalid  and
      Partanen, Niko  and
      Rueter, Jack},
    editor = "Xu, Wei  and
      Ritter, Alan  and
      Baldwin, Tim  and
      Rahimi, Afshin",
    booktitle = "Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wnut-1.3",
    doi = "10.18653/v1/2021.wnut-1.3",
    pages = "20--25",
    abstract = "We present the first openly available corpus for detecting depression in Thai. Our corpus is compiled by expert verified cases of depression in several online blogs.
    We experiment with two different LSTM based models and two different BERT based models. We achieve a 77.53%% accuracy with a Thai BERT model in detecting depression.
    This establishes a good baseline for future researcher on the same corpus. Furthermore, we identify a need for Thai embeddings that have been trained on a more varied corpus than Wikipedia.
    Our corpus, code and trained models have been released openly on Zenodo.",
}
"""

_DESCRIPTION = """\
We present the first openly available corpus for detecting depression in Thai. Our corpus is compiled by expert verified cases of depression in several online blogs.
We experiment with two different LSTM based models and two different BERT based models. We achieve a 77.53%% accuracy with a Thai BERT model in detecting depression.
This establishes a good baseline for future researcher on the same corpus. Furthermore, we identify a need for Thai embeddings that have been trained on a more varied corpus than Wikipedia.
Our corpus, code and trained models have been released openly on Zenodo.
"""

_HOMEPAGE = "https://zenodo.org/records/4734552"

_LICENSE = Licenses.CC_BY_NC_ND_4_0.value

_URLs = "https://zenodo.org/records/4734552/files/data.zip?download=1"

_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ThaiDepressionDataset(datasets.GeneratorBasedBuilder):
    """Thai depression detection dataset."""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} seacrowd schema",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(["depression", "no_depression"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        path = Path(dl_manager.download_and_extract(_URLs))
        data_files = {
            "train": path / "splits/train.json",
            "test": path / "splits/test.json",
            "valid": path / "splits/valid.json",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["valid"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"]},
            ),
        ]

    def _parse_and_label(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        parsed_data = []
        for item in data:
            parsed_data.append({"text": item[0], "label": item[1]})

        return parsed_data

    def _generate_examples(self, filepath: Path):
        print("Reading ", filepath)
        for id, row in enumerate(self._parse_and_label(filepath)):
            if self.config.schema == "source":
                yield id, {"text": row["text"], "label": row["label"]}
            elif self.config.schema == "seacrowd_text":
                yield id, {"id": str(id), "text": row["text"], "label": row["label"]}
            else:
                raise ValueError(f"Invalid config: {self.config.name}")
