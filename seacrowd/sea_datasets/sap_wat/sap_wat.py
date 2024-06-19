from pathlib import Path
from typing import List

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_DATASETNAME = "sap_wat"

_LANGUAGES = ["eng", "ind", "zlm", "tha", "vie"] 

_CITATION = """\
@inproceedings{buschbeck-exel-2020-parallel,
    title = "A Parallel Evaluation Data Set of Software Documentation with Document Structure Annotation",
    author = "Buschbeck, Bianka  and
      Exel, Miriam",
    editor = "Nakazawa, Toshiaki  and
      Nakayama, Hideki  and
      Ding, Chenchen  and
      Dabre, Raj  and
      Kunchukuttan, Anoop  and
      Pa, Win Pa  and
      Bojar, Ond{\v{r}}ej  and
      Parida, Shantipriya  and
      Goto, Isao  and
      Mino, Hidaya  and
      Manabe, Hiroshi  and
      Sudoh, Katsuhito  and
      Kurohashi, Sadao  and
      Bhattacharyya, Pushpak",
    booktitle = "Proceedings of the 7th Workshop on Asian Translation",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.wat-1.20",
    pages = "160--169",
    abstract = "This paper accompanies the software documentation data set for machine translation, a parallel 
    evaluation data set of data originating from the SAP Help Portal, that we released to the machine translation 
    community for research purposes. It offers the possibility to tune and evaluate machine translation systems 
    in the domain of corporate software documentation and contributes to the availability of a wider range of 
    evaluation scenarios. The data set comprises of the language pairs English to Hindi, Indonesian, Malay and 
    Thai, and thus also increases the test coverage for the many low-resource language pairs. Unlike most evaluation 
    data sets that consist of plain parallel text, the segments in this data set come with additional metadata that 
    describes structural information of the document context. We provide insights into the origin and creation, the 
    particularities and characteristics of the data set as well as machine translation results.",
}

"""

_DESCRIPTION = """The data set originates from the SAP Help Portal that contains documentation for SAP products and user 
assistance for product-related questions. The data has been processed in a way that makes it suitable as development and 
test data for machine translation purposes. The current language scope is English to Hindi, Indonesian, Japanese, Korean, 
Malay, Thai, Vietnamese, Simplified Chinese and Traditional Chinese. For each language pair about 4k segments are available, 
split into development and test data. The segments are provided in their document context and are annotated with additional 
metadata from the document."""

_HOMEPAGE = "https://github.com/SAP/software-documentation-data-set-for-machine-translation"

_LICENSE = Licenses.CC_BY_NC_4_0.value

_URLs = {
    _DATASETNAME: "https://raw.githubusercontent.com/SAP/software-documentation-data-set-for-machine-translation/master/{split}_data/en{lang}/software_documentation.{split}.en{lang}.{appx}"
}

_SUPPORTED_TASKS = [
    Tasks.MACHINE_TRANSLATION
]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_SUBSET = ["id", "ms", "th", "vi"]

_LOCAL = False

class SapWatDataset(datasets.GeneratorBasedBuilder):
    """SAP WAT is a software documentation dataset for machine translation. The current language scope is English to Hindi, 
    Indonesian, Japanese, Korean, Malay, Thai, Vietnamese, Simplified Chinese and Traditional Chinese. Here, we only consider 
    EN-ID, EN-TH, EN-MS, EN-VI"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_en_{lang}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"SAP WAT source schema for EN-{lang.upper()}",
            schema="source",
            subset_id=f"{_DATASETNAME}_en_{lang}",
        ) 
        for lang in _SUBSET] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_en_{lang}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"SAP WAT SEACrowd schema for EN-{lang.upper()}",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_en_{lang}",
        ) 
        for lang in _SUBSET
    ]

    DEFAULT_CONFIG_NAME = "sap_wat_en_id_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string")
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

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        lang = self.config.name.split("_")[3]

        splits = {datasets.Split.VALIDATION: "dev", datasets.Split.TEST: "test"}
        data_urls = {
            split: _URLs[_DATASETNAME].format(split=splits[split], lang=lang, appx=lang) for split in splits
        }
        dl_paths = dl_manager.download(data_urls)

        en_data_urls = {
            split: _URLs[_DATASETNAME].format(split=splits[split], lang=lang, appx="en") for split in splits
        }
        en_dl_paths = dl_manager.download(en_data_urls)
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": dl_paths[split], "en_filepath": en_dl_paths[split]},
            )
            for split in splits
        ]

    def _generate_examples(self, filepath: Path, en_filepath: Path):
        with open(en_filepath, "r") as f:
            lines_1 = f.readlines()
        with open(filepath, "r") as f:
            lines_2 = f.readlines()

        if self.config.schema == "source":
            for _id, (line_1, line_2) in enumerate(zip(lines_1, lines_2)):
                ex = {
                    "id": _id,
                    "text": line_1.strip(),
                    "label": line_2.strip()
                }                
                yield _id, ex

        elif self.config.schema == "seacrowd_t2t":
            lang = self.config.name.split("_")[3]
            lang_name = _LANGUAGES[_SUBSET.index(lang)+1]

            for _id, (line_1, line_2) in enumerate(zip(lines_1, lines_2)):
                ex = {
                    "id": _id,
                    "text_1": line_1.strip(),
                    "text_2": line_2.strip(),
                    "text_1_name": 'eng',
                    "text_2_name": lang_name,
                }
                yield _id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")