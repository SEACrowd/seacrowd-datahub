from pathlib import Path

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{clark-etal-2023-seahorse,
    title = "{SEAHORSE}: A Multilingual, Multifaceted Dataset for Summarization Evaluation",
    author = "Clark, Elizabeth  and
      Rijhwani, Shruti  and
      Gehrmann, Sebastian  and
      Maynez, Joshua  and
      Aharoni, Roee  and
      Nikolaev, Vitaly  and
      Sellam, Thibault  and
      Siddhant, Aditya  and
      Das, Dipanjan  and
      Parikh, Ankur",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.584",
    doi = "10.18653/v1/2023.emnlp-main.584",
    pages = "9397--9413",
}
"""

_DATASETNAME = "seahorse"

_DESCRIPTION = """
SEAHORSE is a dataset for multilingual, multifaceted summarization evaluation. It consists of 96K summaries with human
ratings along 6 quality dimensions: comprehensibility, repetition, grammar, attribution, main idea(s), and conciseness,
covering 6 languages, 9 systems and 4 datasets.
"""

_HOMEPAGE = "https://github.com/google-research-datasets/seahorse"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = "https://storage.googleapis.com/seahorse-public/seahorse_data.zip"

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


# The original dataset only contaions gem_id, we need to retrieve the article following https://github.com/google-research-datasets/seahorse?tab=readme-ov-file#retrieving-articles-from-gem
def get_wikilingual_data(lang, split):
    ds = datasets.load_dataset("gem", name=f"wiki_lingua_{lang}", split=split)
    df = ds.to_pandas()
    return dict(zip(*[df[col] for col in ["gem_id", "source"]]))


def get_xlsum_data(lang, split):
    df = datasets.load_dataset("GEM/xlsum", lang)
    return {item["gem_id"]: item["text"] for item in df[split]}


# Both train and validation splits in seahorse are taken from the validation split from the original dataset
_WIKILINGUAL_DATA = {split: get_wikilingual_data("vietnamese_vi", split) for split in ["test", "validation"]}
_XLSUM_DATA = {split: get_xlsum_data("vietnamese", split) for split in ["test", "validation"]}


def get_article(gem_id, split):
    if "wiki_lingua" in gem_id:
        data = _WIKILINGUAL_DATA
    elif "xlsum" in gem_id:
        data = _XLSUM_DATA
    else:
        raise AssertionError("gem_id should either from wiki_lingua or xlsum.")
    return data[split if split == "test" else "validation"][gem_id]


class SeahorseDataset(datasets.GeneratorBasedBuilder):
    """Seahorse is a dataset for multilingual, multifaceted summarization evaluation."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "gem_id": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                    "model": datasets.Value("string"),
                    "question1": datasets.Value("string"),
                    "question2": datasets.Value("string"),
                    "question3": datasets.Value("string"),
                    "question4": datasets.Value("string"),
                    "question5": datasets.Value("string"),
                    "question6": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": f"{data_dir}/seahorse_data/train.tsv",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": f"{data_dir}/seahorse_data/validation.tsv",
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": f"{data_dir}/seahorse_data/test.tsv",
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> tuple[int, dict]:
        df = pd.read_csv(filepath, sep="\t")
        mask = df["worker_lang"] == "vi"
        df_vi = df[mask]
        if self.config.schema == "source":
            for i, row in df_vi.iterrows():
                yield i, {
                    "gem_id": row["gem_id"],
                    "summary": row["summary"],
                    "model": row["model"],
                    "question1": row["question1"],
                    "question2": row["question2"],
                    "question3": row["question3"],
                    "question4": row["question4"],
                    "question5": row["question5"],
                    "question6": row["question6"],
                }

        elif self.config.schema == "seacrowd_t2t":
            for i, row in df_vi.iterrows():
                yield i, {
                    "id": str(i),
                    "text_1": get_article(row["gem_id"], split),
                    "text_2": row["summary"],
                    "text_1_name": "article",
                    "text_2_name": "summary",
                }
