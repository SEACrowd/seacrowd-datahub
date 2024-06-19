import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{koto-etal-2023-large,
    title = "Large Language Models Only Pass Primary School Exams in {I}ndonesia: A Comprehensive Test on {I}ndo{MMLU}",
    author = "Koto, Fajri  and
      Aisyah, Nurul  and
      Li, Haonan  and
      Baldwin, Timothy",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.760",
    doi = "10.18653/v1/2023.emnlp-main.760",
    pages = "12359--12374",
}
"""

_DATASETNAME = "indommlu"

_DESCRIPTION = """
IndoMMLU is the first multi-task language understanding benchmark for Indonesian culture and languages, which consists
of questions from primary school to university entrance exams in Indonesia. By employing professional teachers, we
obtain 14,906 questions across 63 tasks and education levels, with 46% of the questions focusing on assessing
proficiency in the Indonesian language and knowledge of nine local languages and cultures in Indonesia.
"""

_HOMEPAGE = "https://huggingface.co/datasets/indolem/IndoMMLU"

_LANGUAGES = ["ind", "ban", "mad", "nij", "sun", "jav", "mak", "bjn", "abl"]

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_LOCAL = False

_URLS = {_DATASETNAME: {"test": "https://huggingface.co/datasets/indolem/IndoMMLU/resolve/main/IndoMMLU.csv"}}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


lang2subject = {"ind": "Bahasa Indonesia", "ban": "Bahasa Bali", "mad": "Bahasa Madura", "nij": "Bahasa Dayak Ngaju", "sun": "Bahasa Sunda", "jav": "Bahasa Jawa", "mak": "Bahasa Makassar", "bjn": "Bahasa Banjar", "abl": "Bahasa Lampung"}

subject2english = {
    "Sejarah": "History",
    "Geografi": "Geography",
    "Bahasa Lampung": "Lampungic",
    "IPS": "Social science",
    "Bahasa Bali": "Balinese",
    "Bahasa Makassar": "Makassarese",
    "Bahasa Banjar": "Banjarese",
    "Kimia": "Chemistry",
    "Biologi": "Biology",
    "IPA": "Science",
    "Agama Kristen": "Christian religion",
    "Kesenian": "Art",
    "Agama Islam": "Islam religion",
    "Agama Hindu": "Hindu religion",
    "Bahasa Madura": "Madurese",
    "Penjaskes": "Sport",
    "Bahasa Indonesia": "Indonesian language",
    "Fisika": "Physics",
    "Budaya Alam Minangkabau": "Minangkabau culture",
    "Bahasa Dayak Ngaju": "Dayak language",
    "Sosiologi": "Sociology",
    "Ekonomi": "Economy",
    "Bahasa Sunda": "Sundanese",
    "Bahasa Jawa": "Javanese",
    "PPKN": "Civic education",
}

subject2group = {
    "Sejarah": "Humanities",
    "Geografi": "Social science",
    "Bahasa Lampung": "Local languages and cultures",
    "IPS": "Social science",
    "Bahasa Bali": "Local languages and cultures",
    "Bahasa Makassar": "Local languages and cultures",
    "Bahasa Banjar": "Local languages and cultures",
    "Kimia": "STEM",
    "Biologi": "STEM",
    "IPA": "STEM",
    "Agama Kristen": "Humanities",
    "Kesenian": "Humanities",
    "Agama Islam": "Humanities",
    "Agama Hindu": "Humanities",
    "Bahasa Madura": "Local languages and cultures",
    "Penjaskes": "Humanities",
    "Bahasa Indonesia": "Indonesian language",
    "Fisika": "STEM",
    "Budaya Alam Minangkabau": "Local languages and cultures",
    "Bahasa Dayak Ngaju": "Local languages and cultures",
    "Sosiologi": "Social science",
    "Ekonomi": "Social science",
    "Bahasa Sunda": "Local languages and cultures",
    "Bahasa Jawa": "Local languages and cultures",
    "PPKN": "Social science",
}

special_case = ["SD-SMP-SMA", "SD-SMP"]
level_mapper = {
    "SMA": "SMA",  # SMA --> high school level"
    "Seleksi PTN": "University entrance test",
    "SD": "SD",  # SD --> elementary school level
    "SMP": "SMP",  # SMP --> junior high school level
    "Kelas I SD": "SD",
    "Kelas X SMA": "SMA",
    "Kelas XI SMA": "SMA",
    "Kelas XII SMA": "SMA",
    "V SD": "SD",
    "VI SD": "SD",
    "VII SMP": "SMP",
    "VIII SMP ": "SMP",
    "IX SMP": "SMP",
    "Kelas III SD": "SD",
    "Kelas IV SD": "SD",
    "Kelas II SD": "SD",
}


def fix_level(level, kelas):
    # Fixing Level
    if level in special_case:
        kelas = float(kelas)
        if kelas >= 1 and kelas <= 6:
            level = "SD"
        elif kelas >= 7 and kelas <= 9:
            level = "SMP"
        elif kelas >= 10:
            level = "SMA"
        else:
            print(level)
    fixed_level = level_mapper[level]

    # Fixing class
    kelas = str(kelas)
    if kelas.strip() in ["PTN", "2023-10-12 00:00:00"]:
        fixed_kelas = 13
    elif kelas == "4,5,6":
        fixed_kelas = 6
    else:
        fixed_kelas = int(float(kelas.strip()))

    # sanity check over the level and kelas
    return fixed_level, fixed_kelas


def pass_schema_filter(schema, row):
    if schema == "source":
        return True
    lang = schema.split("_")[1]
    if lang not in _LANGUAGES:  # seacrowd_qa
        return True
    if lang == "ind":  # contains "Bahasa Indonesia" and all other non-language subjects
        return (lang2subject[lang] == row["subject"]) or (row["subject"] not in lang2subject.values())
    return lang2subject[lang] == row["subject"]


class IndoMMLUDataset(datasets.GeneratorBasedBuilder):
    """IndoMMLU is the first multitask language understanding benchmark for Indonesian culture and languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_qa",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_qa",
            subset_id=_DATASETNAME,
        ),
    ]
    for lang in _LANGUAGES:
        lang_config = SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_seacrowd_qa",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} {lang} SEACrowd schema",
            schema=f"seacrowd_qa",
            subset_id=_DATASETNAME,
        )
        BUILDER_CONFIGS.append(lang_config)

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "subject": datasets.Value("string"),
                    "group": datasets.Value("string"),
                    "level": datasets.Value("string"),
                    "class": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "options": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "is_for_fewshot": datasets.Value("string"),
                }
            )

        else:
            features = schemas.qa_features
            features["meta"] = {
                "subject": datasets.Value("string"),
                "group": datasets.Value("string"),
                "level": datasets.Value("string"),
                "class": datasets.Value("string"),
                "is_for_fewshot": datasets.Value("string"),
            }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir, "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        data = csv.DictReader(open(filepath[split], newline=""))
        print(self.config.schema)
        for i, row in enumerate(data):
            if pass_schema_filter(self.config.schema, row):
                fixed_level, fixed_kelas = fix_level(row["level"], row["kelas"])
                # The choices are in the format of ["A. xxx", "B. xxx", ...], but answer is only with ["A"], replacing both with only the answer content
                choices = row["jawaban"].split("\n")
                answer_choice = row["kunci"]
                # Find the corresponding choice in the choices.
                # Skip the 2 datapoint (i = 4223, 14150) with invalid answer_choice.
                corresponding_choice = next((choice for choice in choices if choice.startswith(answer_choice)), None)
                if corresponding_choice is None:
                    continue
                else:
                    if self.config.schema == "source":
                        yield i, {
                            "subject": subject2english[row["subject"]],
                            "group": subject2group[row["subject"]],
                            "level": fixed_level,
                            "class": fixed_kelas,
                            "question": row["soal"],
                            "options": [opt[2:].strip() for opt in choices],  # remove A., B., ... in the options,
                            "answer": corresponding_choice[2:].strip(),  # remove A., B., ... in the answer
                            "is_for_fewshot": row["is_for_fewshot"],
                        }
                    else:
                        yield i, {
                            "id": str(i),
                            "question_id": str(i),
                            "document_id": str(i),
                            "question": row["soal"],
                            "type": "multiple_choice",
                            "choices": [opt[2:].strip() for opt in choices],  # remove A., B., ... in the options
                            "context": "",
                            "answer": [corresponding_choice[2:].strip()],  # remove A., B., ... in the answer,
                            "meta": {"subject": subject2english[row["subject"]], "group": subject2group[row["subject"]], "level": fixed_level, "class": fixed_kelas, "is_for_fewshot": row["is_for_fewshot"]},
                        }
