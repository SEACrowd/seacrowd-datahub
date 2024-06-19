"""
SEA Crowd Data Loader for Bloom LM.
"""
from typing import Dict, Iterator, List, Tuple

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
    "kvt": "Lahta",
    "lao": "Lao",
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
    "psp_ceb": "Filipino Sign Language",
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
    "yin": "Riang Lai",
    "zlm": "Malay",
}

_LOCAL = False
_LANGUAGES = list(_LANG_CONFIG.keys())

_DATASETNAME = "bloom_lm"
_DESCRIPTION = r"""
This is a Bloom Library dataset developed for the self-supervised language modeling task.
It covers 74 languages indigenous to SEA overall, amounting to total data of 21K.
This dataset belongs to a CC license, where its datapoints has specific license attached to it.
Before using this dataloader, please accept the acknowledgement at https://huggingface.co/datasets/sil-ai/bloom-lm and use huggingface-cli login for authentication.
"""

_HOMEPAGE = "https://huggingface.co/datasets/sil-ai/bloom-lm"
_LICENSE = Licenses.CC.value

_URL = "https://huggingface.co/datasets/sil-ai/bloom-lm"
_HF_REMOTE_REF = "/".join(_URL.split("/")[-2:])

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "0.1.0"
_SEACROWD_VERSION = "2024.06.20"

CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]


def construct_configs_on_langs() -> List[SEACrowdConfig]:
    """
    The function `construct_configs` constructs a list of SEACrowdConfig objects based on `_LANGUAGES` var, and returns the list.

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
            # since the actual subset_id in source for "psp_ceb" is "psp", we are defining the subset_id as following for loading to source HF
            subset_id=_LANG if _LANG != "psp_ceb" else "psp",
        )
        for _LANG in _LANGUAGES
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
                # since the actual subset_id in source for "psp_ceb" is "psp", we are defining the subset_id as following for loading to source HF
                subset_id=_LANG if _LANG != "psp_ceb" else "psp",
            )
            for _LANG in _LANGUAGES
        ]
    return config_list


class BloomLMDataset(datasets.GeneratorBasedBuilder):
    """Bloom LM dataset, subsetted from https://huggingface.co/datasets/sil-ai/bloom-lm"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = construct_configs_on_langs()

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # source schema
        if _config_schema_name == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "license": datasets.Value("string"),
                    "copyright": datasets.Value("string"),
                    "pageCount": datasets.Value("int32"),
                    "bookInstanceId": datasets.Value("string"),
                    "bookLineage": datasets.Value("string"),
                }
            )

        # ssp schema
        elif _config_schema_name == "seacrowd_ssp":
            features = schemas.ssp_features

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

    def _generate_examples(self, hf_dset) -> Iterator[Tuple[int, Dict]]:
        _config_schema_name = self.config.schema

        _idx = 0
        for datapoints in hf_dset:
            # the `_idx` will be generated manually since no `id` present in the dataset fulfill the purpose as primary key
            if _config_schema_name == "source":
                yield _idx, {colname: datapoints[colname] for colname in self.info.features}

            elif _config_schema_name == "seacrowd_ssp":
                yield _idx, {"id": _idx, "text": datapoints["text"]}

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

            _idx += 1
