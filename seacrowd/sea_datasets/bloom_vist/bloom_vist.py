"""
SEA Crowd Data Loader for Bloom VIST.
"""
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = r"""
@inproceedings{leong-etal-2022-bloom,
    title = "Bloom Library: Multimodal Datasets in 300+ Languages for a Variety of Downstream Tasks",
    author = "Leong, Colin  and
      Nemecek, Joshua  and
      Mansdorfer, Jacob  and
      Filighera, Anna  and
      Owodunni, Abraham  and
      Whitenack, Daniel",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.590",
    doi = "10.18653/v1/2022.emnlp-main.590",
    pages = "8608--8621",
}
"""

logger = datasets.logging.get_logger(__name__)

# this config is created for SEACrowd Dataloader
_LANG_CONFIG = {
    "abc": "Ambala Ayta",
    "ahk": "Akha",
    "bfn": "Bunak",
    "bjn": "Banjar",
    "bkx": "Baikeno",
    "brb": "Brao",
    "brv": "Western Bru",
    "bya": "Batak",
    "bzi": "Bisu",
    "ceb": "Cebuano",
    "cgc": "Kagayanen",
    "cmo": "Central Mnong",
    "ddg": "Fataluku",
    "dmg": "Upper Kinabatangan",
    "dnw": "Western Dani",
    "dtp": "Kadazan Dusun",
    "enc": "En",
    "fil": "Filipino",
    "hil": "Hiligaynon",
    "hro": "Haroi",
    "idt": "Idaté",
    "ilo": "Ilocano",
    "ind": "Indonesian",
    "jra": "Jarai",
    "kak": "Kalanguya",
    "khb": "Lü",
    "khm": "Khmer",
    "kqr": "Kimaragang",
    "krr": "Krung",
    "ksw": "S’gaw Karen",
    "lhu": "Lahu",
    "lsi": "Lacid",
    "lwl": "Eastern Lawa",
    "mdr": "Mandar",
    "mgm": "Mambae",
    "mhx": "Lhao Vo",
    "mkz": "Makasae",
    "mry": "Mandaya",
    "msb": "Masbatenyo",
    "mya": "Burmese",
    "nod": "Northern Thai",
    "nxa": "Nauete",
    "nxl": "South Nuaulu",
    "pag": "Pangasinan",
    "pce": "Ruching Palaung",
    "pea": "Peranakan Indonesian",
    "pmf": "Pamona",
    "psp": "Filipino Sign Language",
    "sea": "Semai",
    "sgd": "Surigaonon",
    "sml": "Central Sama",
    "snl": "Sangil",
    "tdt": "Tetun Dili",
    "tet": "Tetun",
    "tha": "Thai",
    "tkd": "Tukudede",
    "tpu": "Tampuan",
    "war": "Waray-Waray",
    "wms": "Wambon",
    "yet": "Yetfa",
    "yin": "Riang Lai",
    "zlm": "Malay",
}

_LOCAL = False
_LANGUAGES = list(_LANG_CONFIG.keys())


_DATASETNAME = "bloom_vist"
_DESCRIPTION = r"""
BLOOM VIST is a visual storytelling of books that consists of 62 languages indigenous to SEA.
This dataset is owned by Bloom, a free, open-source software developed by SIL International and associated with Bloom Library, app, and services.
This dataset is released with the LICENSE family of Creative Commons (although each story datapoints has its licensing in more detail,
e.g cc-by, cc-by-nc, cc-by-nd, cc-by-sa, cc-by-nc-nd, cc-by-nc-sa).
Before using this dataloader, please accept the acknowledgement at https://huggingface.co/datasets/sil-ai/bloom-vist and use huggingface-cli login for authentication.
"""

_HOMEPAGE = "https://huggingface.co/datasets/sil-ai/bloom-vist"
_LICENSE = Licenses.CC.value

_URL = "https://huggingface.co/datasets/sil-ai/bloom-vist"
_HF_REMOTE_REF = "/".join(_URL.split("/")[-2:])

_SUPPORTED_TASKS = [Tasks.IMAGE_CAPTIONING]
_SOURCE_VERSION = "0.1.0"
_SEACROWD_VERSION = "2024.06.20"

CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]


def conform_init_config():
    """Assertion Function for Instantiated Configs"""
    if len(_LANGUAGES) == 0:
        raise AssertionError("No Languages detected from config!")
    if len(CONFIG_SUFFIXES_FOR_TASK) != len(_SUPPORTED_TASKS):
        raise AssertionError("Config prefixes don't matched in terms of `len` with `_SUPPORTED_TASKS`!")
    if len(CONFIG_SUFFIXES_FOR_TASK) == 0:
        raise AssertionError("Config prefixes and `_SUPPORTED_TASKS` have `len` of 0!")


conform_init_config()


def construct_configs_on_langs(languages: list = None) -> List[SEACrowdConfig]:
    """
    The function `construct_configs` constructs a list of SEACrowdConfig objects based on the provided
    languages or a default language, and returns the list.

    input:
        languages (list, default None): The `languages` parameter is a list that specifies the languages for which the
        configurations need to be constructed. If no languages are provided (value=None), the first value in language config
        will be used.
    output:
        a list of `SEACrowdConfig` objects based on instantiated init variables
    """

    # set output var
    config_list = []

    # construct zipped arg for config instantiation
    TASKS_AND_CONFIG_SUFFIX_PAIRS = list(zip(_SUPPORTED_TASKS, CONFIG_SUFFIXES_FOR_TASK))

    # implement source schema
    version, config_name_prefix = _SOURCE_VERSION, "source"
    config_list += [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{_LANG}_{config_name_prefix}",
            version=datasets.Version(version),
            description=f"{_DATASETNAME} {config_name_prefix} schema for language code {_LANG}",
            schema=f"{config_name_prefix}",
            subset_id=_LANG,
        )
        for _LANG in languages
    ]

    # implement SEACrowd schema
    version, config_name_prefix = _SEACROWD_VERSION, "seacrowd"
    for task_obj, config_name_suffix in TASKS_AND_CONFIG_SUFFIX_PAIRS:
        config_list += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{_LANG}_{config_name_prefix}_{config_name_suffix}",
                version=datasets.Version(version),
                description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name} and language code {_LANG}",
                schema=f"{config_name_prefix}_{config_name_suffix}",
                subset_id=_LANG,
            )
            for _LANG in languages
        ]
    return config_list


class BloomVISTDataset(datasets.GeneratorBasedBuilder):
    """Bloom VIST dataset, subsetted from https://huggingface.co/datasets/sil-ai/bloom-vist"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = construct_configs_on_langs(_LANGUAGES)

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # source schema
        if _config_schema_name == "source":
            features = datasets.Features(
                {
                    "title": datasets.Value("string"),
                    "license": datasets.Value("string"),
                    "album_id": datasets.Value("string"),
                    "story": datasets.Sequence(
                        feature={"image_id": datasets.Value("string"), "image_url": datasets.Value("string"), "story_index": datasets.Value("int32"), "story_id": datasets.Value("string"), "text": datasets.Value("string")}, length=-1, id=None
                    ),
                }
            )

        # image-text schema
        elif _config_schema_name == "seacrowd_imtext":
            features = schemas.image_text_features()

        else:
            raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        hf_dset_dict = datasets.load_dataset(_HF_REMOTE_REF, self.config.subset_id)

        return [datasets.SplitGenerator(name=datasets.Split(dset_key), gen_kwargs={"hf_dset": dset}) for dset_key, dset in hf_dset_dict.items() if dset.num_rows > 0]

    def _generate_examples(self, hf_dset) -> Tuple[int, Dict]:
        _config_schema_name = self.config.schema

        _idx = 0
        for datapoints in hf_dset:
            # for source schema, the `_idx` will be taken from "album_id" value
            if _config_schema_name == "source":
                yield datapoints["album_id"], {colname: datapoints[colname] for colname in self.info.features}

            # for seacrowd schema, the `_idx` will be created manually
            # since one album_id has multiple pairs of image-text
            elif _config_schema_name == "seacrowd_imtext":
                # check the len of the features in sequenced columns
                # since in source hf there's no validation on data integrity
                _len_vars = []
                _ftrs_in_seq = ("image_id", "image_url", "story_index", "story_id", "text")
                story_data = datapoints["story"]
                for ftr in _ftrs_in_seq:
                    _len_vars.append(len(story_data[ftr]))

                # skip story w/ mismatched infos
                if max(_len_vars) != min(_len_vars):
                    continue

                for num_data in range(max(_len_vars)):
                    yield _idx, {"id": _idx, "image_paths": [story_data["image_url"][num_data]], "texts": story_data["text"][num_data], "metadata": {"context": datapoints["title"], "labels": []}}
                    _idx += 1

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")
