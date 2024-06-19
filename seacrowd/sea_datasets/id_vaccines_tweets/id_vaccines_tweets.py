import datasets
import pandas

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                      DEFAULT_SOURCE_VIEW_NAME, Tasks, Licenses)

_DATASETNAME = "id_vaccines_tweets"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@article{febriyanti2021analisis,
  title={ANALISIS SENTIMEN MASYARAKAT INDONESIA TERHADAP PELAKSANAAN VAKSIN COVID'19},
  author={Febriyanti, Syintya and Nursidah, Dea Ratu and Gustiara, Dela and Yulianti, Rika},
  journal={Khazanah: Jurnal Mahasiswa},
  volume={13},
  number={2},
  year={2021}
}
"""

_DESCRIPTION = """\
Dataset containing tweets about COVID-19 vaccines with manually labelled information about whether they are a
subjective tweet and their sentiment polarity. Tweets are from 20-27 June 2021 and 15-22 July 2021.
"""

_HOMEPAGE = "https://github.com/rayendito/id-vaccines-tweets"

_LICENSE = Licenses.UNKNOWN.value

_URL = "https://raw.githubusercontent.com/rayendito/id-vaccines-tweets/main/id_vaccines_tweets.csv"

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class IdVaccinesTweetsDataset(datasets.GeneratorBasedBuilder):
    """This is a seacrowd dataloader for id_vaccines_tweets, for every example in the dataset, it contains a subjective
    tweet and their sentiment polarity.  Tweets are from 20-27 June 2021 and 15-22 July 2021."""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=_DESCRIPTION,
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=_DESCRIPTION,
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "idx": datasets.Value("string"),
                    "form_text": datasets.Value("string"),
                    "norm_text": datasets.Value("string"),
                    "subjective": datasets.Value("float"),
                    "sentiment": datasets.Value("float"),
                }
            )
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features([-1.0, 0.0, 1.0])
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """ "return splitGenerators"""
        downloaded_files = dl_manager.download_and_extract(_URL)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files})]

    def _generate_examples(self, filepath):
        data_lines = pandas.read_csv(filepath, skip_blank_lines=True)
        keys = data_lines.keys()
        indexes = data_lines[keys[0]][1:]
        norms = data_lines[keys[1]][1:]
        formals = data_lines[keys[2]][1:]
        subjs = data_lines[keys[3]][1:]
        posnegs = data_lines[keys[4]][1:]
        if self.config.schema == "source":
            for idx, (ind, norm, form, subj, posneg) in enumerate(zip(indexes, norms, formals, subjs, posnegs)):
                yield idx, {
                    "idx": str(ind),
                    "form_text": form,
                    "norm_text": norm,
                    "subjective": float(subj),
                    "sentiment": float(posneg),
                }
        if self.config.schema == "seacrowd_text":
            for idx, (ind, norm, posneg) in enumerate(zip(indexes, norms, posnegs)):
                yield idx, {"id": str(ind), "text": norm, "label": float(posneg)}
