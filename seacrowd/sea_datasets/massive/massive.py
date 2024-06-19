import json
from typing import List

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{fitzgerald2022massive,
      title={MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages},
      author={Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron
      Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter
      Leeuwis and Gokhan Tur and Prem Natarajan},
      year={2022},
      eprint={2204.08582},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@inproceedings{bastianelli-etal-2020-slurp,
    title = "{SLURP}: A Spoken Language Understanding Resource Package",
    author = "Bastianelli, Emanuele  and
      Vanzo, Andrea  and
      Swietojanski, Pawel  and
      Rieser, Verena",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.588",
    doi = "10.18653/v1/2020.emnlp-main.588",
    pages = "7252--7262",
    abstract = "Spoken Language Understanding infers semantic meaning directly from audio data, and thus promises to
    reduce error propagation and misunderstandings in end-user applications. However, publicly available SLU resources are limited.
    In this paper, we release SLURP, a new SLU package containing the following: (1) A new challenging dataset in English spanning
    18 domains, which is substantially bigger and linguistically more diverse than existing datasets; (2) Competitive baselines
    based on state-of-the-art NLU and ASR systems; (3) A new transparent metric for entity labelling which enables a detailed error
    analysis for identifying potential areas of improvement. SLURP is available at https://github.com/pswietojanski/slurp."
}
"""
_DATASETNAME = "massive"
_DESCRIPTION = """\
MASSIVE dataset—Multilingual Amazon Slu resource package (SLURP) for Slot-filling, Intent classification, and
Virtual assistant Evaluation. MASSIVE contains 1M realistic, parallel, labeled virtual assistant utterances
spanning 18 domains, 60 intents, and 55 slots. MASSIVE was created by tasking professional translators to
localize the English-only SLURP dataset into 50 typologically diverse languages, including 8 native languages
and 2 other languages mostly spoken in Southeast Asia.
"""
_HOMEPAGE = "https://github.com/alexa/massive"
_LICENSE = Licenses.CC_BY_4_0.value
_LOCAL = False
_LANGUAGES = ["ind", "jav", "khm", "zlm", "mya", "tha", "tgl", "vie"]

_URLS = {
    _DATASETNAME: "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
}
_SUPPORTED_TASKS = [Tasks.INTENT_CLASSIFICATION, Tasks.SLOT_FILLING]
_SOURCE_VERSION = "1.1.0"
_SEACROWD_VERSION = "2024.06.20"

# ind, jav, khm, zlm, mya, tha, tgl, vie, cmn, tam
_LANGS = [
    "af-ZA",
    "am-ET",
    "ar-SA",
    "az-AZ",
    "bn-BD",
    "cy-GB",
    "da-DK",
    "de-DE",
    "el-GR",
    "en-US",
    "es-ES",
    "fa-IR",
    "fi-FI",
    "fr-FR",
    "he-IL",
    "hi-IN",
    "hu-HU",
    "hy-AM",
    "id-ID",  # ind
    "is-IS",
    "it-IT",
    "ja-JP",
    "jv-ID",  # jav
    "ka-GE",
    "km-KH",  # khm
    "kn-IN",
    "ko-KR",
    "lv-LV",
    "ml-IN",
    "mn-MN",
    "ms-MY",  # zlm
    "my-MM",  # mya
    "nb-NO",
    "nl-NL",
    "pl-PL",
    "pt-PT",
    "ro-RO",
    "ru-RU",
    "sl-SL",
    "sq-AL",
    "sv-SE",
    "sw-KE",
    "ta-IN",
    "te-IN",
    "th-TH",  # tha
    "tl-PH",  # tgl
    "tr-TR",
    "ur-PK",
    "vi-VN",  # vie
    "zh-CN",  # cmn
    "zh-TW",
]
_SUBSETS = ["id-ID", "jv-ID", "km-KH", "ms-MY", "my-MM", "th-TH", "tl-PH", "vi-VN"]

_SCENARIOS = ["calendar", "recommendation", "social", "general", "news", "cooking", "iot", "email", "weather", "alarm", "transport", "lists", "takeaway", "play", "audio", "music", "qa", "datetime"]

_INTENTS = [
    "audio_volume_other",
    "play_music",
    "iot_hue_lighton",
    "general_greet",
    "calendar_set",
    "audio_volume_down",
    "social_query",
    "audio_volume_mute",
    "iot_wemo_on",
    "iot_hue_lightup",
    "audio_volume_up",
    "iot_coffee",
    "takeaway_query",
    "qa_maths",
    "play_game",
    "cooking_query",
    "iot_hue_lightdim",
    "iot_wemo_off",
    "music_settings",
    "weather_query",
    "news_query",
    "alarm_remove",
    "social_post",
    "recommendation_events",
    "transport_taxi",
    "takeaway_order",
    "music_query",
    "calendar_query",
    "lists_query",
    "qa_currency",
    "recommendation_movies",
    "general_joke",
    "recommendation_locations",
    "email_querycontact",
    "lists_remove",
    "play_audiobook",
    "email_addcontact",
    "lists_createoradd",
    "play_radio",
    "qa_stock",
    "alarm_query",
    "email_sendemail",
    "general_quirky",
    "music_likeness",
    "cooking_recipe",
    "email_query",
    "datetime_query",
    "transport_traffic",
    "play_podcasts",
    "iot_hue_lightchange",
    "calendar_remove",
    "transport_query",
    "transport_ticket",
    "qa_factoid",
    "iot_cleaning",
    "alarm_set",
    "datetime_convert",
    "iot_hue_lightoff",
    "qa_definition",
    "music_dislikeness",
]

_TAGS = [
    "O",
    "B-food_type",
    "B-movie_type",
    "B-person",
    "B-change_amount",
    "I-relation",
    "I-game_name",
    "B-date",
    "B-movie_name",
    "I-person",
    "I-place_name",
    "I-podcast_descriptor",
    "I-audiobook_name",
    "B-email_folder",
    "B-coffee_type",
    "B-app_name",
    "I-time",
    "I-coffee_type",
    "B-transport_agency",
    "B-podcast_descriptor",
    "I-playlist_name",
    "B-media_type",
    "B-song_name",
    "I-music_descriptor",
    "I-song_name",
    "B-event_name",
    "I-timeofday",
    "B-alarm_type",
    "B-cooking_type",
    "I-business_name",
    "I-color_type",
    "B-podcast_name",
    "I-personal_info",
    "B-weather_descriptor",
    "I-list_name",
    "B-transport_descriptor",
    "I-game_type",
    "I-date",
    "B-place_name",
    "B-color_type",
    "B-game_name",
    "I-artist_name",
    "I-drink_type",
    "B-business_name",
    "B-timeofday",
    "B-sport_type",
    "I-player_setting",
    "I-transport_agency",
    "B-game_type",
    "B-player_setting",
    "I-music_album",
    "I-event_name",
    "I-general_frequency",
    "I-podcast_name",
    "I-cooking_type",
    "I-radio_name",
    "I-joke_type",
    "I-meal_type",
    "I-transport_type",
    "B-joke_type",
    "B-time",
    "B-order_type",
    "B-business_type",
    "B-general_frequency",
    "I-food_type",
    "I-time_zone",
    "B-currency_name",
    "B-time_zone",
    "B-ingredient",
    "B-house_place",
    "B-audiobook_name",
    "I-ingredient",
    "I-media_type",
    "I-news_topic",
    "B-music_genre",
    "I-definition_word",
    "B-list_name",
    "B-playlist_name",
    "B-email_address",
    "I-currency_name",
    "I-movie_name",
    "I-device_type",
    "I-weather_descriptor",
    "B-audiobook_author",
    "I-audiobook_author",
    "I-app_name",
    "I-order_type",
    "I-transport_name",
    "B-radio_name",
    "I-business_type",
    "B-definition_word",
    "B-artist_name",
    "I-movie_type",
    "B-transport_name",
    "I-email_folder",
    "B-music_album",
    "I-house_place",
    "I-music_genre",
    "B-drink_type",
    "I-alarm_type",
    "B-music_descriptor",
    "B-news_topic",
    "B-meal_type",
    "I-transport_descriptor",
    "I-email_address",
    "I-change_amount",
    "B-device_type",
    "B-transport_type",
    "B-relation",
    "I-sport_type",
    "B-personal_info",
]


class MASSIVEDataset(datasets.GeneratorBasedBuilder):
    """MASSIVE datasets contains datasets to detect the intent from the text and fill the dialogue slots"""

    BUILDER_CONFIGS = (
        [
            SEACrowdConfig(
                name=f"massive_{subset}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"MASSIVE source schema for {subset}",
                schema="source",
                subset_id="massive_" + subset,
            )
            for subset in _SUBSETS
        ]
        + [
            SEACrowdConfig(
                name=f"massive_{subset}_seacrowd_text",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"MASSIVE Nusantara intent classification schema for {subset}",
                schema="seacrowd_text",
                subset_id="massive_intent_" + subset,
            )
            for subset in _SUBSETS
        ]
        + [
            SEACrowdConfig(
                name=f"massive_{subset}_seacrowd_seq_label",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"MASSIVE Nusantara slot filling schema for {subset}",
                schema="seacrowd_seq_label",
                subset_id="massive_slot_filling_" + subset,
            )
            for subset in _SUBSETS
        ]
        + [
            SEACrowdConfig(
                name="massive_source",
                version=datasets.Version(_SOURCE_VERSION),
                description="MASSIVE source schema",
                schema="source",
                subset_id="massive",
            ),
            SEACrowdConfig(
                name="massive_seacrowd_text",
                version=datasets.Version(_SEACROWD_VERSION),
                description="MASSIVE Nusantara intent classification schema",
                schema="seacrowd_text",
                subset_id="massive_intent",
            ),
            SEACrowdConfig(
                name="massive_seacrowd_seq_label",
                version=datasets.Version(_SEACROWD_VERSION),
                description="MASSIVE Nusantara slot filling schema",
                schema="seacrowd_seq_label",
                subset_id="massive_slot_filling",
            ),
        ]
    )

    DEFAULT_CONFIG_NAME = "massive_id-ID_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "locale": datasets.Value("string"),
                    "partition": datasets.Value("string"),
                    "scenario": datasets.features.ClassLabel(names=_SCENARIOS),
                    "intent": datasets.features.ClassLabel(names=_INTENTS),
                    "utt": datasets.Value("string"),
                    "annot_utt": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=_TAGS)),
                    "worker_id": datasets.Value("string"),
                    "slot_method": datasets.Sequence(
                        {
                            "slot": datasets.Value("string"),
                            "method": datasets.Value("string"),
                        }
                    ),
                    "judgments": datasets.Sequence(
                        {
                            "worker_id": datasets.Value("string"),
                            "intent_score": datasets.Value("int8"),  # [0, 1, 2]
                            "slots_score": datasets.Value("int8"),  # [0, 1, 2]
                            "grammar_score": datasets.Value("int8"),  # [0, 1, 2, 3, 4]
                            "spelling_score": datasets.Value("int8"),  # [0, 1, 2]
                            "language_identification": datasets.Value("string"),
                        }
                    ),
                }
            )
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(label_names=_INTENTS)
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(label_names=_TAGS)
        else:
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        archive = dl_manager.download(_URLS[_DATASETNAME])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": dl_manager.iter_archive(archive),
                    "split": "train",
                    "lang": self.config.name,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "files": dl_manager.iter_archive(archive),
                    "split": "dev",
                    "lang": self.config.name,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "files": dl_manager.iter_archive(archive),
                    "split": "test",
                    "lang": self.config.name,
                },
            ),
        ]

    def _get_bio_format(self, text):
        """This function is modified from https://huggingface.co/datasets/qanastek/MASSIVE/blob/main/MASSIVE.py"""
        tags, tokens = [], []

        bio_mode = False
        cpt_bio = 0
        current_tag = None

        split_iter = iter(text.split(" "))

        for s in split_iter:
            if s.startswith("["):
                current_tag = s.strip("[")
                bio_mode = True
                cpt_bio += 1
                next(split_iter)
                continue

            elif s.endswith("]"):
                bio_mode = False
                if cpt_bio == 1:
                    prefix = "B-"
                else:
                    prefix = "I-"
                token = prefix + current_tag
                word = s.strip("]")
                current_tag = None
                cpt_bio = 0

            else:
                if bio_mode:
                    if cpt_bio == 1:
                        prefix = "B-"
                    else:
                        prefix = "I-"
                    token = prefix + current_tag
                    word = s
                    cpt_bio += 1
                else:
                    token = "O"
                    word = s

            tags.append(token)
            tokens.append(word)

        return tokens, tags

    def _generate_examples(self, files: list, split: str, lang: str):
        _id = 0

        lang = lang.replace("massive_", "").replace("source", "").replace("seacrowd_text", "").replace("seacrowd_seq_label", "")

        if not lang:
            lang = _LANGS.copy()
        else:
            lang = [lang[:-1]]

        # logger.info("Generating examples from = %s", ", ".join(lang))

        for path, f in files:
            curr_lang = path.split(f"{_SOURCE_VERSION[:-2]}/data/")[-1].split(".jsonl")[0]

            if not lang:
                break
            elif curr_lang in lang:
                lang.remove(curr_lang)
            else:
                continue

            # Read the file
            lines = f.read().decode(encoding="utf-8").split("\n")

            for line in lines:
                data = json.loads(line)

                if data["partition"] != split:
                    continue

                # Slot method
                if "slot_method" in data:
                    slot_method = [
                        {
                            "slot": s["slot"],
                            "method": s["method"],
                        }
                        for s in data["slot_method"]
                    ]
                else:
                    slot_method = []

                # Judgments
                if "judgments" in data:
                    judgments = [
                        {
                            "worker_id": j["worker_id"],
                            "intent_score": j["intent_score"],
                            "slots_score": j["slots_score"],
                            "grammar_score": j["grammar_score"],
                            "spelling_score": j["spelling_score"],
                            "language_identification": j["language_identification"] if "language_identification" in j else "target",
                        }
                        for j in data["judgments"]
                    ]
                else:
                    judgments = []

                if self.config.schema == "source":
                    tokens, tags = self._get_bio_format(data["annot_utt"])

                    yield _id, {
                        "id": str(_id) + "_" + data["id"],
                        "locale": data["locale"],
                        "partition": data["partition"],
                        "scenario": data["scenario"],
                        "intent": data["intent"],
                        "utt": data["utt"],
                        "annot_utt": data["annot_utt"],
                        "tokens": tokens,
                        "ner_tags": tags,
                        "worker_id": data["worker_id"],
                        "slot_method": slot_method,
                        "judgments": judgments,
                    }

                elif self.config.schema == "seacrowd_seq_label":
                    tokens, tags = self._get_bio_format(data["annot_utt"])

                    yield _id, {
                        "id": str(_id) + "_" + data["id"],
                        "tokens": tokens,
                        "labels": tags,
                    }

                elif self.config.schema == "seacrowd_text":
                    yield _id, {
                        "id": str(_id) + "_" + data["id"],
                        "text": data["utt"],
                        "label": data["intent"],
                    }

                else:
                    raise ValueError(f"Invalid config: {self.config.name}")

                _id += 1
