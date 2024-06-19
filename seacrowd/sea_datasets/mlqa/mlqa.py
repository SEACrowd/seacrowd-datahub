import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = r"""\
@article{lewis2019mlqa,
    author={Lewis, Patrick and O\{g}uz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
    title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
    journal={arXiv preprint arXiv:1910.07475},
    year={2019}
}
"""

_DATASETNAME = "mlqa"

_DESCRIPTION = """\
MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
4 different languages on average.
"""

_HOMEPAGE = "https://github.com/facebookresearch/MLQA"
_LICENSE = Licenses.CC_BY_SA_3_0.value
_LANGUAGES = ["vie"]
_URL = "https://dl.fbaipublicfiles.com/MLQA/"
_DEV_TEST_URL = "MLQA_V1.zip"
_TRANSLATE_TEST_URL = "mlqa-translate-test.tar.gz"
_TRANSLATE_TRAIN_URL = "mlqa-translate-train.tar.gz"
_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_LOCAL = False


class MLQADataset(datasets.GeneratorBasedBuilder):
    """
    MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
    MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
    German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
    4 different languages on average.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    subsets = [
        "mlqa-translate-test.vi",
        "mlqa-translate-train.vi",
        "mlqa.vi.ar",
        "mlqa.vi.de",
        "mlqa.vi.zh",
        "mlqa.vi.en",
        "mlqa.vi.es",
        "mlqa.vi.hi",
        "mlqa.vi.vi",
        "mlqa.ar.vi",
        "mlqa.de.vi",
        "mlqa.zh.vi",
        "mlqa.en.vi",
        "mlqa.es.vi",
        "mlqa.hi.vi",
    ]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="{sub}_source".format(sub=subset),
            version=datasets.Version(_SOURCE_VERSION),
            description="{sub} source schema".format(sub=subset),
            schema="source",
            subset_id="{sub}".format(sub=subset),
        )
        for subset in subsets
    ] + [
        SEACrowdConfig(
            name="{sub}_seacrowd_qa".format(sub=subset),
            version=datasets.Version(_SEACROWD_VERSION),
            description="{sub} SEACrowd schema".format(sub=subset),
            schema="seacrowd_qa",
            subset_id="{sub}".format(sub=subset),
        )
        for subset in subsets
    ]

    DEFAULT_CONFIG_NAME = "mlqa.vi.vi_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {"context": datasets.Value("string"), "question": datasets.Value("string"), "answers": datasets.Features({"answer_start": [datasets.Value("int64")], "text": [datasets.Value("string")]}), "id": datasets.Value("string")}
            )
        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        name_split = self.config.name.split("_")
        url = ""
        data_path = ""

        if name_split[0].startswith("mlqa-translate-train"):
            config_name, lang = name_split[0].split(".")
            url = _URL + _TRANSLATE_TRAIN_URL
            data_path = dl_manager.download(url)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # Whatever you put in gen_kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": f"{config_name}/{lang}_squad-translate-train-train-v1.1.json",
                        "files": dl_manager.iter_archive(data_path),
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": f"{config_name}/{lang}_squad-translate-train-dev-v1.1.json",
                        "files": dl_manager.iter_archive(data_path),
                        "split": "test",
                    },
                ),
            ]

        elif name_split[0].startswith("mlqa-translate-test"):
            config_name, lang = name_split[0].split(".")
            url = _URL + _TRANSLATE_TEST_URL
            data_path = dl_manager.download(url)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": f"{config_name}/translate-test-context-{lang}-question-{lang}.json",
                        "files": dl_manager.iter_archive(data_path),
                        "split": "test",
                    },
                ),
            ]

        elif name_split[0].startswith("mlqa."):
            url = _URL + _DEV_TEST_URL
            data_path = dl_manager.download_and_extract(url)
            ctx_lang, qst_lang = name_split[0].split(".")[1:]
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            os.path.join(data_path, "MLQA_V1/dev"),
                            f"dev-context-{ctx_lang}-question-{qst_lang}.json",
                        ),
                        "split": "dev",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            os.path.join(data_path, "MLQA_V1/test"),
                            f"test-context-{ctx_lang}-question-{qst_lang}.json",
                        ),
                        "split": "test",
                    },
                ),
            ]
        elif name_split[0] == "mlqa":
            url = _URL + _DEV_TEST_URL
            data_path = dl_manager.download_and_extract(url)
            ctx_lang = qst_lang = "vi"
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            os.path.join(data_path, "MLQA_V1/dev"),
                            f"dev-context-{ctx_lang}-question-{qst_lang}.json",
                        ),
                        "split": "dev",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            os.path.join(data_path, "MLQA_V1/test"),
                            f"test-context-{ctx_lang}-question-{qst_lang}.json",
                        ),
                        "split": "test",
                    },
                ),
            ]

    def _generate_examples(self, filepath: Path, split: str, files=None) -> Tuple[int, Dict]:
        is_config_ok = True
        if self.config.name.startswith("mlqa-translate"):
            for path, f in files:
                if path == filepath:
                    data = json.loads(f.read().decode("utf-8"))
                    break

        elif self.config.schema == "source" or self.config.schema == "seacrowd_qa":
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        else:
            is_config_ok = False
            raise ValueError(f"Invalid config: {self.config.name}")

        if is_config_ok:
            count = 0
            for examples in data["data"]:
                for example in examples["paragraphs"]:
                    context = example["context"]
                    for qa in example["qas"]:
                        question = qa["question"]
                        id_ = qa["id"]
                        answers = qa["answers"]
                        answers_start = [answer["answer_start"] for answer in answers]
                        answers_text = [answer["text"] for answer in answers]

                        if self.config.schema == "source":
                            yield count, {
                                "context": context,
                                "question": question,
                                "answers": {"answer_start": answers_start, "text": answers_text},
                                "id": id_,
                            }
                            count += 1

                        elif self.config.schema == "seacrowd_qa":
                            yield count, {"question_id": id_, "context": context, "question": question, "answer": {"answer_start": answers_start[0], "text": answers_text[0]}, "id": id_, "choices": [], "type": "extractive", "document_id": count, "meta":{}}
                            count += 1
