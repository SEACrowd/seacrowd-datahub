"""
SEA Crowd Data Loader for Audio Keyword Spotting.
"""
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

# since the dataset doesn't have any citation and it was derived using someone else's work, this citation variable will cite source work instead (total of 3, ML Spoken Words 1 and Trabina 2)
_CITATION = r"""
@inproceedings{mazumder2021multilingual,
    title={Multilingual Spoken Words Corpus},
    author={Mazumder, Mark and Chitlangia, Sharad and Banbury, Colby and Kang, Yiping and Ciro, Juan Manuel and Achorn, Keith and Galvez, Daniel and Sabini, Mark and Mattson, Peter and Kanter, David and others},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021}
}
@inproceedings{wu-etal-2018-creating,
    title = "Creating a Translation Matrix of the {B}ible{'}s Names Across 591 Languages",
    author = "Wu, Winston  and
      Vyas, Nidhi  and
      Yarowsky, David",
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Hasida, Koiti  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios  and
      Tokunaga, Takenobu",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1263",
}
@inproceedings{wu-yarowsky-2018-comparative,
    title = "A Comparative Study of Extremely Low-Resource Transliteration of the World{'}s Languages",
    author = "Wu, Winston  and
      Yarowsky, David",
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Hasida, Koiti  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios  and
      Tokunaga, Takenobu",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1150",
}
"""

logger = datasets.logging.get_logger(__name__)

_LOCAL = False
_LANGUAGES = ["ind"]

_DATASETNAME = "audio_keyword_spotting"
_DESCRIPTION = r"This dataset is a ASR for short text & voices, focusing in identifying common words (or keywords) with entities of Person name and Place Name found in Bible, as found in trabina (https://github.com/wswu/trabina)."

_HOMEPAGE = "https://huggingface.co/datasets/sil-ai/audio-keyword-spotting"
_LICENSE = Licenses.CC_BY_4_0.value

_URL = "https://huggingface.co/datasets/sil-ai/audio-keyword-spotting"
_HF_REMOTE_REF = "/".join(_URL.split("/")[-2:])

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SOURCE_VERSION = "0.0.1"
_SEACROWD_VERSION = "2024.06.20"

CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]


def construct_configs() -> List[SEACrowdConfig]:
    """
    The function `construct_configs` constructs a list of SEACrowdConfig objects and returns the config list.

    input:
        None
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
            name=f"{_DATASETNAME}_{config_name_prefix}",
            version=datasets.Version(version),
            description=f"{_DATASETNAME} {config_name_prefix} schema",
            schema=f"{config_name_prefix}",
            subset_id=config_name_prefix,
        )
    ]

    # implement SEACrowd schema
    version, config_name_prefix = _SEACROWD_VERSION, "seacrowd"
    for task_obj, config_name_suffix in TASKS_AND_CONFIG_SUFFIX_PAIRS:
        config_list += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{config_name_prefix}_{config_name_suffix}",
                version=datasets.Version(version),
                description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name}",
                schema=f"{config_name_prefix}_{config_name_suffix}",
                subset_id=config_name_prefix,
            )
        ]
    return config_list


class AudioKeywordSpottingDataset(datasets.GeneratorBasedBuilder):
    """AudioKeywordSpotting dataset, subsetted from https://huggingface.co/datasets/sil-ai/audio-keyword-spotting"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = construct_configs()
    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # source schema
        if _config_schema_name == "source":
            _GENDERS = ["MALE", "FEMALE", "OTHER", "NAN"]
            features = datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "is_valid": datasets.Value("bool"),
                    "language": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "gender": datasets.ClassLabel(names=_GENDERS),
                    "keyword": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                }
            )

        # speech-text schema
        elif _config_schema_name == "seacrowd_sptext":
            features = schemas.speech_text_features

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
        hf_dset_dict = datasets.load_dataset(_HF_REMOTE_REF, "ind")

        return [datasets.SplitGenerator(name=datasets.Split(dset_key), gen_kwargs={"hf_dset": dset}) for dset_key, dset in hf_dset_dict.items() if dset.num_rows > 0]

    def _generate_examples(self, hf_dset) -> Tuple[int, Dict]:
        _config_schema_name = self.config.schema

        _idx = 0
        for datapoints in hf_dset:
            # since no _idx is available to be used, we're creating it manually for both schema
            if _config_schema_name == "source":
                yield _idx, {colname: datapoints[colname] for colname in self.info.features}

            elif _config_schema_name == "seacrowd_sptext":
                yield _idx, {"id": _idx, "path": datapoints["file"], "audio": datapoints["audio"], "text": datapoints["keyword"], "speaker_id": datapoints["speaker_id"], "metadata": {"speaker_age": None, "speaker_gender": datapoints["gender"]}}

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

            _idx += 1
