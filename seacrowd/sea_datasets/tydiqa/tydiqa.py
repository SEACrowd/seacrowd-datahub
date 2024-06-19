import json

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = r"""\
@article{clark-etal-2020-tydi,
    title = "{T}y{D}i {QA}: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages",
    author = "Clark, Jonathan H.  and
      Choi, Eunsol  and
      Collins, Michael  and
      Garrette, Dan  and
      Kwiatkowski, Tom  and
      Nikolaev, Vitaly  and
      Palomaki, Jennimaria",
    editor = "Johnson, Mark  and
      Roark, Brian  and
      Nenkova, Ani",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "8",
    year = "2020",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2020.tacl-1.30",
    doi = "10.1162/tacl_a_00317",
    pages = "454--470",
    abstract = "Confidently making progress on multilingual modeling requires challenging, trustworthy evaluations.
    We present TyDi QA{---}a question answering dataset covering 11 typologically diverse languages with 204K
    question-answer pairs. The languages of TyDi QA are diverse with regard to their typology{---}the set of
    linguistic features each language expresses{---}such that we expect models performing well on this set to
    generalize across a large number of the world{'}s languages. We present a quantitative analysis of the data
    quality and example-level qualitative linguistic analyses of observed language phenomena that would not be found
    in English-only corpora. To provide a realistic information-seeking task and avoid priming effects, questions are
    written by people who want to know the answer, but don{'}t know the answer yet, and the data is collected directly
    in each language without the use of translation.",
}

@inproceedings{cahyawijaya-etal-2021-indonlg,
    title = "{I}ndo{NLG}: Benchmark and Resources for Evaluating {I}ndonesian Natural Language Generation",
    author = "Cahyawijaya, Samuel  and
      Winata, Genta Indra  and
      Wilie, Bryan  and
      Vincentio, Karissa  and
      Li, Xiaohong  and
      Kuncoro, Adhiguna  and
      Ruder, Sebastian  and
      Lim, Zhi Yuan  and
      Bahar, Syafri  and
      Khodra, Masayu  and
      Purwarianti, Ayu  and
      Fung, Pascale",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.699",
    doi = "10.18653/v1/2021.emnlp-main.699",
    pages = "8875--8898"
}
"""

_DATASETNAME = "tydiqa"

_DESCRIPTION = """\
    TyDi QA is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs.
    The languages of TyDi QA are diverse with regard to their typology -- the set of linguistic features that each language
    expresses -- such that we expect models performing well on this set to generalize across a large number of the languages
    in the world. It contains language phenomena that would not be found in English-only corpora. To provide a realistic
    information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but
    don’t know the answer yet, (unlike SQuAD and its descendents) and the data is collected directly in each language
    without the use of translation (unlike MLQA and XQuAD).
    """

_HOMEPAGE = "https://github.com/google-research-datasets/tydiqa"
_LICENSE = Licenses.APACHE_2_0.value
_HF_URL = "https://huggingface.co/datasets/tydiqa"
_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_LANGUAGES = ["ind", "tha"]
_LOCAL = False
_SOURCE_VERSION = "1.0.0"
_SOURCE_VERSION_P = "1.0.0"
_SOURCE_VERSION_S = "1.1.0"
_SEACROWD_VERSION = "2024.06.20"

_URL = "https://storage.googleapis.com/tydiqa/"
_PRIMARY_URLS = {
    "train": _URL + "v1.0/tydiqa-v1.0-train.jsonl.gz",
    "dev": _URL + "v1.0/tydiqa-v1.0-dev.jsonl.gz",
}
_SECONDARY_URLS = {
    "train": _URL + "v1.1/tydiqa-goldp-v1.1-train.json",
    "dev": _URL + "v1.1/tydiqa-goldp-v1.1-dev.json",
}

_SELECTP_DESP = """Passage selection task (SelectP): Given a list of the passages in the article, return either (a) the index of
          the passage that answers the question or (b) NULL if no such passage exists.
          """
_MINSPAN_DESP = """Minimal answer span task (MinSpan): Given the full text of an article, return one of (a) the start and end
          byte indices of the minimal span that completely answers the question; (b) YES or NO if the question requires
          a yes/no answer and we can draw a conclusion from the passage; (c) NULL if it is not possible to produce a
          minimal answer for this question."""
_GOLDP_DESP = """Gold passage task (GoldP): Given a passage that is guaranteed to contain the
          answer, predict the single contiguous span of characters that answers the question. This is more similar to
          existing reading comprehension datasets (as opposed to the information-seeking task outlined above).
          """
_ID_DESP = """{I}ndo{NLG}: Benchmark and Resources for Evaluating {I}ndonesian Natural Language Generation, is a benchmark
          for evaluating Indonesian natural language generation (NLG) systems. The question-answer pairs are collected
          for each language without using translation services. It uses the Indonesian data from the secondary Gold
          passage task of the TyDiQA dataset.  As the original dataset only provides training and validation sets,
          TydiQA-ID randomly split off 15% of the training data and use it as the test set.
          """


def config_constructor(subset_id, schema, desc, version):
    return SEACrowdConfig(name=f"{_DATASETNAME}_{subset_id}_{schema}", description=desc, version=datasets.Version(version), schema=schema, subset_id=subset_id)


class TydiqaDataset(datasets.GeneratorBasedBuilder):
    """
    This is a main class of SEACrowd dataloader for TyDi QA, which is a question answering dataset covering 11 typologically
    diverse languages with 204K question-answer pairs. The languages of TyDi QA are diverse with regard to their typology.
    Here we also specially provide the split on the primary and secondary task for SEA language like indonesian and thai.
    """

    BUILDER_CONFIGS = [
        # source schema
        # selectp source schema
        config_constructor(subset_id="selectp", schema="source", desc=_SELECTP_DESP, version=_SOURCE_VERSION_P),
        config_constructor(subset_id="selectp_ind", schema="source", desc=_SELECTP_DESP, version=_SOURCE_VERSION_P),
        config_constructor(subset_id="selectp_tha", schema="source", desc=_SELECTP_DESP, version=_SOURCE_VERSION_P),
        # minspan source schema
        config_constructor(subset_id="minspan", schema="source", desc=_MINSPAN_DESP, version=_SOURCE_VERSION_P),
        config_constructor(subset_id="minspan_ind", schema="source", desc=_MINSPAN_DESP, version=_SOURCE_VERSION_P),
        config_constructor(subset_id="minspan_tha", schema="source", desc=_MINSPAN_DESP, version=_SOURCE_VERSION_P),
        # goldp source schema
        config_constructor(subset_id="goldp", schema="source", desc=_GOLDP_DESP, version=_SOURCE_VERSION_S),
        config_constructor(subset_id="goldp_ind", schema="source", desc=_GOLDP_DESP, version=_SOURCE_VERSION_S),
        # tydiqa_id source schema
        config_constructor(subset_id="id", schema="source", desc=_ID_DESP, version=_SOURCE_VERSION_P),
        # seacrowd schema
        # selectp seacrowd schema
        config_constructor(subset_id="selectp", schema="seacrowd_qa", desc=_SELECTP_DESP, version=_SEACROWD_VERSION),
        config_constructor(subset_id="selectp_ind", schema="seacrowd_qa", desc=_SELECTP_DESP, version=_SEACROWD_VERSION),
        config_constructor(subset_id="selectp_tha", schema="seacrowd_qa", desc=_SELECTP_DESP, version=_SEACROWD_VERSION),
        # minspan seacrowd schema
        config_constructor(subset_id="minspan", schema="seacrowd_qa", desc=_MINSPAN_DESP, version=_SEACROWD_VERSION),
        config_constructor(subset_id="minspan_ind", schema="seacrowd_qa", desc=_MINSPAN_DESP, version=_SEACROWD_VERSION),
        config_constructor(subset_id="minspan_tha", schema="seacrowd_qa", desc=_MINSPAN_DESP, version=_SEACROWD_VERSION),
        # goldp seacrowd schema
        config_constructor(subset_id="goldp", schema="seacrowd_qa", desc=_GOLDP_DESP, version=_SEACROWD_VERSION),
        config_constructor(subset_id="goldp_ind", schema="seacrowd_qa", desc=_GOLDP_DESP, version=_SEACROWD_VERSION),
        # tydiqa_id seacrowd schema
        config_constructor(subset_id="id", schema="seacrowd_qa", desc=_ID_DESP, version=_SEACROWD_VERSION),
    ]
    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_id_source"

    def _info(self):
        if ("selectp" in self.config.name) or ("minspan" in self.config.name):
            if "source" in self.config.name:
                features = datasets.Features(
                    {
                        "passage_answer_candidates": datasets.features.Sequence(
                            {
                                "plaintext_start_byte": datasets.Value("int32"),
                                "plaintext_end_byte": datasets.Value("int32"),
                            }
                        ),
                        "question_text": datasets.Value("string"),
                        "document_title": datasets.Value("string"),
                        "language": datasets.Value("string"),
                        "annotations": datasets.features.Sequence(
                            {
                                "passage_answer_candidate_index": datasets.Value("int32"),
                                "minimal_answers_start_byte": datasets.Value("int32"),
                                "minimal_answers_end_byte": datasets.Value("int32"),
                                "yes_no_answer": datasets.Value("string"),
                            }
                        ),
                        "document_plaintext": datasets.Value("string"),
                        "document_url": datasets.Value("string"),
                    }
                )
            elif "seacrowd" in self.config.name:
                features = schemas.qa_features
                features["meta"] = {
                    "passage_answer_candidates": datasets.features.Sequence(
                        {
                            "plaintext_start_byte": datasets.Value("int32"),
                            "plaintext_end_byte": datasets.Value("int32"),
                        }
                    ),
                    "annotations": datasets.features.Sequence(
                        {
                            "passage_answer_candidate_index": datasets.Value("int32"),
                            "minimal_answers_start_byte": datasets.Value("int32"),
                            "minimal_answers_end_byte": datasets.Value("int32"),
                            "yes_no_answer": datasets.Value("string"),
                        }
                    ),
                    "language": datasets.Value("string"),
                }

        elif ("goldp" in self.config.name) or ("tydiqa_id" in self.config.name):
            if "source" in self.config.name:
                features = datasets.Features(
                    {
                        "id": datasets.Value("string"),
                        "title": datasets.Value("string"),
                        "context": datasets.Value("string"),
                        "question": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "answer_start": datasets.Value("int32"),
                            }
                        ),
                    }
                )
            elif "seacrowd" in self.config.name:
                features = schemas.qa_features
                features["meta"] = {
                    "answer_start": datasets.Sequence(datasets.Value("int32")),
                }
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        primary_downloaded = dl_manager.download_and_extract(_PRIMARY_URLS)
        secondary_downloaded = dl_manager.download_and_extract(_SECONDARY_URLS)

        if ("selectp" in self.config.name) or ("minspan" in self.config.name):
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": primary_downloaded["train"]},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": primary_downloaded["dev"]},
                ),
            ]

        elif "goldp" in self.config.name:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": secondary_downloaded["train"]},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": secondary_downloaded["dev"]},
                ),
            ]
        elif "tydiqa_id" in self.config.name:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": secondary_downloaded["train"], "split": "train"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": secondary_downloaded["train"], "split": "test"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": secondary_downloaded["dev"], "split": "validation"},
                ),
            ]

    def _generate_examples(self, filepath, split=None):
        """Yields examples."""

        if ("selectp" in self.config.name) or ("minspan" in self.config.name):
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row)
                    passages = data["passage_answer_candidates"]
                    end_byte = [passage["plaintext_end_byte"] for passage in passages]
                    start_byte = [passage["plaintext_start_byte"] for passage in passages]
                    title = data["document_title"]
                    lang = data["language"]
                    question = data["question_text"]
                    annotations = data["annotations"]
                    yes_no_answers = [annotation["yes_no_answer"] for annotation in annotations]
                    min_answers_end_byte = [annotation["minimal_answer"]["plaintext_end_byte"] for annotation in annotations]
                    min_answers_start_byte = [annotation["minimal_answer"]["plaintext_start_byte"] for annotation in annotations]
                    passage_cand_answers = [annotation["passage_answer"]["candidate_index"] for annotation in annotations]
                    doc = data["document_plaintext"]
                    url = data["document_url"]
                    if (self.config.name == "tydiqa_selectp_source") or (self.config.name == "tydiqa_minspan_source"):
                        yield id_, primary_source_helper(id_, start_byte, end_byte, question, title, lang, passage_cand_answers, min_answers_start_byte, min_answers_end_byte, yes_no_answers, doc, url)
                    elif (self.config.name == "tydiqa_selectp_ind_source") or (self.config.name == "tydiqa_minspan_ind_source"):
                        if lang == "indonesian":
                            yield id_, primary_source_helper(id_, start_byte, end_byte, question, title, lang, passage_cand_answers, min_answers_start_byte, min_answers_end_byte, yes_no_answers, doc, url)
                    elif (self.config.name == "tydiqa_selectp_tha_source") or (self.config.name == "tydiqa_minspan_tha_source"):
                        if lang == "thai":
                            yield id_, primary_source_helper(id_, start_byte, end_byte, question, title, lang, passage_cand_answers, min_answers_start_byte, min_answers_end_byte, yes_no_answers, doc, url)
                    # seacrowd
                    elif (self.config.name == "tydiqa_selectp_seacrowd_qa") or (self.config.name == "tydiqa_minspan_seacrowd_qa"):
                        yield id_, primary_seacrowd_helper(id_, title, question, doc, start_byte, end_byte, passage_cand_answers, min_answers_start_byte, min_answers_end_byte, yes_no_answers, lang)
                    elif (self.config.name == "tydiqa_selectp_ind_seacrowd_qa") or (self.config.name == "tydiqa_minspan_ind_seacrowd_qa"):
                        if lang == "indonesian":
                            yield id_, primary_seacrowd_helper(id_, title, question, doc, start_byte, end_byte, passage_cand_answers, min_answers_start_byte, min_answers_end_byte, yes_no_answers, lang)
                    elif (self.config.name == "tydiqa_selectp_tha_seacrowd_qa") or (self.config.name == "tydiqa_minspan_tha_seacrowd_qa"):
                        if lang == "thai":
                            yield id_, primary_seacrowd_helper(id_, title, question, doc, start_byte, end_byte, passage_cand_answers, min_answers_start_byte, min_answers_end_byte, yes_no_answers, lang)
                    else:
                        raise ValueError(f"No configs to match {self.config.name} in primary_task")

        elif ("goldp" in self.config.name) or ("tydiqa_id" in self.config.name):
            with (open(filepath, encoding="utf-8") as f):
                data = json.load(f)
                tydiqa_id_num = 0
                for article in data["data"]:
                    title = article.get("title", "").strip()
                    for paragraph in article["paragraphs"]:
                        context = paragraph["context"].strip()
                        for qa in paragraph["qas"]:
                            question = qa["question"].strip()
                            id_ = qa["id"]
                            answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                            answers = [answer["text"].strip() for answer in qa["answers"]]
                            if self.config.name == "tydiqa_goldp_source":
                                yield id_, second_source_helper(id_, title, context, question, answer_starts, answers)

                            elif self.config.name == "tydiqa_goldp_ind_source":
                                if id_.startswith("indonesian"):
                                    yield id_, second_source_helper(id_, title, context, question, answer_starts, answers)
                            elif self.config.name == "tydiqa_id_source":
                                if id_.startswith("indonesian"):
                                    tydiqa_id_num += 1
                                    if split == "train" and tydiqa_id_num >= 856:
                                        yield id_, second_source_helper(id_, title, context, question, answer_starts, answers)
                                    if split == "test" and tydiqa_id_num < 856:
                                        yield id_, second_source_helper(id_, title, context, question, answer_starts, answers)
                                    if split == "validation":
                                        yield id_, second_source_helper(id_, title, context, question, answer_starts, answers)

                            elif self.config.name == "tydiqa_goldp_seacrowd_qa":
                                yield id_, second_seacrowd_helper(id_, question, context, answers, answer_starts)
                            elif self.config.name == "tydiqa_goldp_ind_seacrowd_qa":
                                if id_.startswith("indonesian"):
                                    yield id_, second_seacrowd_helper(id_, question, context, answers, answer_starts)
                            elif self.config.name == "tydiqa_id_seacrowd_qa":
                                if id_.startswith("indonesian"):
                                    tydiqa_id_num += 1
                                    if split == "train" and tydiqa_id_num >= 856:
                                        yield id_, second_seacrowd_helper(id_, question, context, answers, answer_starts)
                                    if split == "test" and tydiqa_id_num < 856:
                                        yield id_, second_seacrowd_helper(id_, question, context, answers, answer_starts)
                                    if split == "validation":
                                        yield id_, second_seacrowd_helper(id_, question, context, answers, answer_starts)
                            else:
                                raise ValueError(f"No configs to match {self.config.name} in secondary_task")


def primary_source_helper(id_, start_byte, end_byte, question, title, lang, passage_cand_answers, min_answers_start_byte, min_answers_end_byte, yes_no_answers, doc, url):
    return {
        "passage_answer_candidates": {
            "plaintext_start_byte": start_byte,
            "plaintext_end_byte": end_byte,
        },
        "question_text": question,
        "document_title": title,
        "language": lang,
        "annotations": {
            "passage_answer_candidate_index": passage_cand_answers,
            "minimal_answers_start_byte": min_answers_start_byte,
            "minimal_answers_end_byte": min_answers_end_byte,
            "yes_no_answer": yes_no_answers,
        },
        "document_plaintext": doc,
        "document_url": url,
    }


def primary_seacrowd_helper(id_, title, question, doc, start_byte, end_byte, passage_cand_answers, min_answers_start_byte, min_answers_end_byte, yes_no_answers, lang):
    return {
        "id": str(id_),
        "question_id": title,
        "document_id": title,
        "question": question,
        "type": "multiple_choice",
        "choices": [""],
        "context": doc,
        "answer": [""],
        "meta": {
            "passage_answer_candidates": {
                "plaintext_start_byte": start_byte,
                "plaintext_end_byte": end_byte,
            },
            "annotations": {
                "passage_answer_candidate_index": passage_cand_answers,
                "minimal_answers_start_byte": min_answers_start_byte,
                "minimal_answers_end_byte": min_answers_end_byte,
                "yes_no_answer": yes_no_answers,
            },
            "language": lang,
        },
    }


def second_source_helper(id_, title, context, question, answer_starts, answers):
    return {
        "title": title,
        "context": context,
        "question": question,
        "id": id_,
        "answers": {
            "answer_start": answer_starts,
            "text": answers,
        },
    }


def second_seacrowd_helper(id_, question, context, answers, answer_starts):
    return {
        "id": id_,
        "question_id": id_,
        "document_id": id_,
        "question": question,
        "type": "abstractive",
        "choices": [],
        "context": context,
        "answer": answers,
        "meta": {"answer_start": answer_starts},
    }
