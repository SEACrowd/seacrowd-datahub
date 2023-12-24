"""
SEA Crowd Data Loader for Bloom Captioning.
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
    abstract = "We present Bloom Library, a linguistically diverse set of multimodal and multilingual datasets for language modeling, image captioning, visual storytelling, and speech synthesis/recognition. These datasets represent either the most, or among the most, multilingual datasets for each of the included downstream tasks. In total, the initial release of the Bloom Library datasets covers 363 languages across 32 language families. We train downstream task models for various languages represented in the data, showing the viability of the data for future work in low-resource, multimodal NLP and establishing the first known baselines for these downstream tasks in certain languages (e.g., Bisu [bzi], with an estimated population of 700 users). Some of these first-of-their-kind baselines are comparable to state-of-the-art performance for higher-resourced languages. The Bloom Library datasets are released under Creative Commons licenses on the Hugging Face datasets hub to catalyze more linguistically diverse research in the included downstream tasks.",
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
    "dtr": "Lotud",
    "enc": "En",
    "fil": "Filipino",
    "gal": "Galolen",
    "hil": "Hiligaynon",
    "hre": "Hre",
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
    "llg": "Lole",
    "lsi": "Lacid",
    "lwl": "Eastern Lawa",
    "mdr": "Mandar",
    "mgm": "Mambae",
    "mhx": "Lhao Vo",
    "mkz": "Makasae",
    "mnw": "Mon",
    "mqj": "Mamasa",
    "mry": "Mandaya",
    "msb": "Masbatenyo",
    "mya": "Burmese",
    "nod": "Northern Thai",
    "nst": "Tangshang Naga",
    "nxa": "Nauete",
    "nxl": "South Nuaulu",
    "pag": "Pangasinan",
    "pce": "Ruching Palaung",
    "pdu": "Kayan",
    "pea": "Peranakan Indonesian",
    "pmf": "Pamona",
    "sea": "Semai",
    "sgd": "Surigaonon",
    "shn": "Shan",
    "sml": "Central Sama",
    "snl": "Sangil",
    "tdt": "Tetun Dili",
    "tet": "Tetun",
    "tha": "Thai",
    "tkd": "Tukudede",
    "tnt": "Tontemboan",
    "tom": "Tombulu",
    "tpu": "Tampuan",
    "vie": "Vietnamese",
    "war": "Waray-Waray",
    "wms": "Wambon",
    "wnk": "Wanukaka",
    "xmm": "Manado Malay",
    "yet": "Yetfa",
    "zlm": "Malay",
}

_LOCAL = False
_LANGUAGES = list(_LANG_CONFIG.keys())


_DATASETNAME = "bloom_captioning"
_DESCRIPTION = r"""
    This is a Bloom Library dataset developed for the image captioning task.
    It covers 74 languages indigenous to SEA overall, amounting to total data of 21K.
    This dataset belongs to a CC license, where its datapoints has specific license attached to it.
"""

_HOMEPAGE = "https://huggingface.co/datasets/sil-ai/bloom-captioning"
_LICENSE = Licenses.CC.value

_URL = "https://huggingface.co/datasets/sil-ai/bloom-captioning"
_HF_REMOTE_REF = "/".join(_URL.split("/")[-2:])

_SUPPORTED_TASKS = [Tasks.IMAGE_CAPTIONING]
_SOURCE_VERSION = "0.1.0"
_SEACROWD_VERSION = "1.0.0"

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


class BloomCaptioningDataset(datasets.GeneratorBasedBuilder):
    """Bloom Captioning dataset, subsetted from https://huggingface.co/datasets/sil-ai/bloom-captioning"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = construct_configs_on_langs(_LANGUAGES)

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # source schema
        if _config_schema_name == "source":
            features = datasets.Features(
                {
                    "image_id": datasets.Value("string"),
                    "image_url": datasets.Value("string"),
                    "caption": datasets.Value("string"),
                    "story_id": datasets.Value("string"),
                    "album_id": datasets.Value("string"),
                    "license": datasets.Value("string"),
                    "original_bloom_language_tag": datasets.Value("string"),
                    "index_in_story": datasets.Value("uint16"),
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

        return [datasets.SplitGenerator(name=datasets.Split(dset_key), gen_kwargs={"hf_dset": dset}) for dset_key, dset in hf_dset_dict.items()]

    def _generate_examples(self, hf_dset) -> Tuple[int, Dict]:
        _config_schema_name = self.config.schema

        _idx = 0
        for datapoints in hf_dset:
            # for both schema, the `_idx` will be taken from "image_id" value
            if _config_schema_name == "source":
                yield datapoints["image_id"], {colname: datapoints[colname] for colname in self.info.features}

            elif _config_schema_name == "seacrowd_imtext":
                yield datapoints["image_id"], {"id": datapoints["image_id"], "image_paths": [datapoints["image_url"]], "texts": datapoints["caption"], "metadata": {"context": "", "labels": []}}
                _idx += 1

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")
