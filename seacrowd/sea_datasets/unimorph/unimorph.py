from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@misc{batsuren2022unimorph,
      title={UniMorph 4.0: Universal Morphology},
      author={
        Khuyagbaatar Batsuren and Omer Goldman and Salam Khalifa and Nizar
        Habash and Witold Kieraś and Gábor Bella and Brian Leonard and Garrett
        Nicolai and Kyle Gorman and Yustinus Ghanggo Ate and Maria Ryskina and
        Sabrina J. Mielke and Elena Budianskaya and Charbel El-Khaissi and Tiago
        Pimentel and Michael Gasser and William Lane and Mohit Raj and Matt
        Coler and Jaime Rafael Montoya Samame and Delio Siticonatzi Camaiteri
        and Benoît Sagot and Esaú Zumaeta Rojas and Didier López Francis and
        Arturo Oncevay and Juan López Bautista and Gema Celeste Silva Villegas
        and Lucas Torroba Hennigen and Adam Ek and David Guriel and Peter Dirix
        and Jean-Philippe Bernardy and Andrey Scherbakov and Aziyana Bayyr-ool
        and Antonios Anastasopoulos and Roberto Zariquiey and Karina Sheifer and
        Sofya Ganieva and Hilaria Cruz and Ritván Karahóǧa and Stella
        Markantonatou and George Pavlidis and Matvey Plugaryov and Elena
        Klyachko and Ali Salehi and Candy Angulo and Jatayu Baxi and Andrew
        Krizhanovsky and Natalia Krizhanovskaya and Elizabeth Salesky and Clara
        Vania and Sardana Ivanova and Jennifer White and Rowan Hall Maudslay and
        Josef Valvoda and Ran Zmigrod and Paula Czarnowska and Irene Nikkarinen
        and Aelita Salchak and Brijesh Bhatt and Christopher Straughn and Zoey
        Liu and Jonathan North Washington and Yuval Pinter and Duygu Ataman and
        Marcin Wolinski and Totok Suhardijanto and Anna Yablonskaya and Niklas
        Stoehr and Hossep Dolatian and Zahroh Nuriah and Shyam Ratan and Francis
        M. Tyers and Edoardo M. Ponti and Grant Aiton and Aryaman Arora and
        Richard J. Hatcher and Ritesh Kumar and Jeremiah Young and Daria
        Rodionova and Anastasia Yemelina and Taras Andrushko and Igor Marchenko
        and Polina Mashkovtseva and Alexandra Serova and Emily Prud'hommeaux and
        Maria Nepomniashchaya and Fausto Giunchiglia and Eleanor Chodroff and
        Mans Hulden and Miikka Silfverberg and Arya D. McCarthy and David
        Yarowsky and Ryan Cotterell and Reut Tsarfaty and Ekaterina Vylomova},
      year={2022},
      eprint={2205.03608},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_LOCAL = False
_LANGUAGES = ["ind", "kod", "ceb", "hil", "tgl"]
_DATASETNAME = "unimorph"
_DESCRIPTION = """\
The Universal Morphology (UniMorph) project is a collaborative effort providing
broad-coverage instantiated normalized morphological inflection tables for
undreds of diverse world languages. The project comprises two major thrusts: a
language-independent feature schema for rich morphological annotation, and a
type-level resource of annotated data in diverse languages realizing that
schema. 5 Austronesian languages spoken in Southeast Asia, consisting 2
Malayo-Polynesian languages and 3 Greater Central Philippine languages, become
the part of UniMorph 4.0 release.
"""

_HOMEPAGE = "https://unimorph.github.io"
_LICENSE = Licenses.CC_BY_SA_3_0.value
_URL = "https://raw.githubusercontent.com/unimorph/"

_SUPPORTED_TASKS = [Tasks.MORPHOLOGICAL_INFLECTION]
_SOURCE_VERSION = "4.0.0"
_SEACROWD_VERSION = "2024.06.20"


class UnimorphDataset(datasets.GeneratorBasedBuilder):
    """Unimorh 4.0 dataset by Batsuren et al., (2022)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "pairs_multi"

    dataset_names = sorted([f"{_DATASETNAME}_{lang}" for lang in _LANGUAGES])
    BUILDER_CONFIGS = []
    for name in dataset_names:
        source_config = SEACrowdConfig(
            name=f"{name}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=name,
        )
        BUILDER_CONFIGS.append(source_config)
        seacrowd_config = SEACrowdConfig(
            name=f"{name}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=name,
        )
        BUILDER_CONFIGS.append(seacrowd_config)

    # Add configuration that allows loading all datasets at once.
    BUILDER_CONFIGS.extend(
        [
            # unimorph_source
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} source schema (all)",
                schema="source",
                subset_id=_DATASETNAME,
            ),
            # unimorph_seacrowd_pairs
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} SEACrowd schema (all)",
                schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
                subset_id=_DATASETNAME,
            ),
        ]
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"
    # https://huggingface.co/datasets/universal_morphologies/blob/main/universal_morphologies.py
    CLASS_CATEGORIES = {
        "Aktionsart": ["STAT", "DYN", "TEL", "ATEL", "PCT", "DUR", "ACH", "ACCMP", "SEMEL", "ACTY"],
        "Animacy": ["ANIM", "INAN", "HUM", "NHUM"],
        "Argument_Marking": [
            "ARGNO1S",
            "ARGNO2S",
            "ARGNO3S",
            "ARGNO1P",
            "ARGNO2P",
            "ARGNO3P",
            "ARGAC1S",
            "ARGAC2S",
            "ARGAC3S",
            "ARGAC1P",
            "ARGAC2P",
            "ARGAC3P",
            "ARGAB1S",
            "ARGAB2S",
            "ARGAB3S",
            "ARGAB1P",
            "ARGAB2P",
            "ARGAB3P",
            "ARGER1S",
            "ARGER2S",
            "ARGER3S",
            "ARGER1P",
            "ARGER2P",
            "ARGER3P",
            "ARGDA1S",
            "ARGDA2S",
            "ARGDA3S",
            "ARGDA1P",
            "ARGDA2P",
            "ARGDA3P",
            "ARGBE1S",
            "ARGBE2S",
            "ARGBE3S",
            "ARGBE1P",
            "ARGBE2P",
            "ARGBE3P",
        ],
        "Aspect": ["IPFV", "PFV", "PRF", "PROG", "PROSP", "ITER", "HAB"],
        "Case": [
            "NOM",
            "ACC",
            "ERG",
            "ABS",
            "NOMS",
            "DAT",
            "BEN",
            "PRP",
            "GEN",
            "REL",
            "PRT",
            "INS",
            "COM",
            "VOC",
            "COMPV",
            "EQTV",
            "PRIV",
            "PROPR",
            "AVR",
            "FRML",
            "TRANS",
            "BYWAY",
            "INTER",
            "AT",
            "POST",
            "IN",
            "CIRC",
            "ANTE",
            "APUD",
            "ON",
            "ONHR",
            "ONVR",
            "SUB",
            "REM",
            "PROXM",
            "ESS",
            "ALL",
            "ABL",
            "APPRX",
            "TERM",
        ],
        "Comparison": ["CMPR", "SPRL", "AB", "RL", "EQT"],
        "Definiteness": ["DEF", "INDF", "SPEC", "NSPEC"],
        "Deixis": ["PROX", "MED", "REMT", "REF1", "REF2", "NOREF", "PHOR", "VIS", "NVIS", "ABV", "EVEN", "BEL"],
        "Evidentiality": ["FH", "DRCT", "SEN", "VISU", "NVSEN", "AUD", "NFH", "QUOT", "RPRT", "HRSY", "INFER", "ASSUM"],
        "Finiteness": ["FIN", "NFIN"],
        "Gender": [
            "MASC",
            "FEM",
            "NEUT",
            "NAKH1",
            "NAKH2",
            "NAKH3",
            "NAKH4",
            "NAKH5",
            "NAKH6",
            "NAKH7",
            "NAKH8",
            "BANTU1",
            "BANTU2",
            "BANTU3",
            "BANTU4",
            "BANTU5",
            "BANTU6",
            "BANTU7",
            "BANTU8",
            "BANTU9",
            "BANTU10",
            "BANTU11",
            "BANTU12",
            "BANTU13",
            "BANTU14",
            "BANTU15",
            "BANTU16",
            "BANTU17",
            "BANTU18",
            "BANTU19",
            "BANTU20",
            "BANTU21",
            "BANTU22",
            "BANTU23",
        ],
        "Information_Structure": ["TOP", "FOC"],
        "Interrogativity": ["DECL", "INT"],
        "Language_Specific": [
            "LGSPEC1",
            "LGSPEC2",
            "LGSPEC3",
            "LGSPEC4",
            "LGSPEC5",
            "LGSPEC6",
            "LGSPEC7",
            "LGSPEC8",
            "LGSPEC9",
            "LGSPEC10",
        ],
        "Mood": [
            "IND",
            "SBJV",
            "REAL",
            "IRR",
            "AUPRP",
            "AUNPRP",
            "IMP",
            "COND",
            "PURP",
            "INTEN",
            "POT",
            "LKLY",
            "ADM",
            "OBLIG",
            "DEB",
            "PERM",
            "DED",
            "SIM",
            "OPT",
        ],
        "Number": ["SG", "PL", "GRPL", "DU", "TRI", "PAUC", "GRPAUC", "INVN"],
        "Part_Of_Speech": [
            "N",
            "PROPN",
            "ADJ",
            "PRO",
            "CLF",
            "ART",
            "DET",
            "V",
            "ADV",
            "AUX",
            "V.PTCP",
            "V.MSDR",
            "V.CVB",
            "ADP",
            "COMP",
            "CONJ",
            "NUM",
            "PART",
            "INTJ",
        ],
        "Person": ["0", "1", "2", "3", "4", "INCL", "EXCL", "PRX", "OBV"],
        "Polarity": ["POS", "NEG"],
        "Politeness": [
            "INFM",
            "FORM",
            "ELEV",
            "HUMB",
            "POL",
            "AVOID",
            "LOW",
            "HIGH",
            "STELEV",
            "STSUPR",
            "LIT",
            "FOREG",
            "COL",
        ],
        "Possession": [
            "ALN",
            "NALN",
            "PSS1S",
            "PSS2S",
            "PSS2SF",
            "PSS2SM",
            "PSS2SINFM",
            "PSS2SFORM",
            "PSS3S",
            "PSS3SF",
            "PSS3SM",
            "PSS1D",
            "PSS1DI",
            "PSS1DE",
            "PSS2D",
            "PSS2DM",
            "PSS2DF",
            "PSS3D",
            "PSS3DF",
            "PSS3DM",
            "PSS1P",
            "PSS1PI",
            "PSS1PE",
            "PSS2P",
            "PSS2PF",
            "PSS2PM",
            "PSS3PF",
            "PSS3PM",
        ],
        "Switch_Reference": ["SS", "SSADV", "DS", "DSADV", "OR", "SIMMA", "SEQMA", "LOG"],
        "Tense": ["PRS", "PST", "FUT", "IMMED", "HOD", "1DAY", "RCT", "RMT"],
        "Valency": ["IMPRS", "INTR", "TR", "DITR", "REFL", "RECP", "CAUS", "APPL"],
        "Voice": ["ACT", "MID", "PASS", "ANTIP", "DIR", "INV", "AGFOC", "PFOC", "LFOC", "BFOC", "ACFOC", "IFOC", "CFOC"],
    }

    TAG_TO_CAT = dict([(tag, cat) for cat, tags in CLASS_CATEGORIES.items() for tag in tags])
    CLASS_LABELS = [feat for _, category in CLASS_CATEGORIES.items() for feat in category]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "lemma": datasets.Value("string"),
                    "forms": datasets.Sequence(
                        dict(
                            [("word", datasets.Value("string"))]
                            + [(cat, datasets.Sequence(datasets.ClassLabel(names=tasks))) for cat, tasks in self.CLASS_CATEGORIES.items()]
                            + [("Other", datasets.Sequence(datasets.Value("string")))]  # for misspecified tags
                        )
                    ),
                }
            )

        if self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            all_features = [feat for _, category in self.CLASS_CATEGORIES.items() for feat in category]
            features = schemas.pairs_multi_features(label_names=self.CLASS_LABELS)

        return datasets.DatasetInfo(description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION)

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Return SplitGenerators."""
        source_data = []

        lang = self.config.name.split("_")[1]
        if lang in _LANGUAGES:
            # Load data per language
            source_data.append(dl_manager.download_and_extract(_URL + f"{lang}/main/{lang}"))
        else:
            # Load examples from all languages at once.
            for lang in _LANGUAGES:
                source_data.append(dl_manager.download_and_extract(_URL + f"{lang}/main/{lang}"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": source_data,
                },
            )
        ]

    def _generate_examples(self, filepaths: List[Path]) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""

        all_forms: Dict[str, List[Dict[str, Any]]] = {}
        for source_file in filepaths:
            with open(source_file, encoding="utf-8") as file:
                for row in file:
                    if row.strip() == "" or row.strip().startswith("#"):
                        continue
                    lemma, word, tags = row.strip().split("\t")
                    all_forms[lemma] = all_forms.get(lemma, [])
                    tag_list = tags.replace("NDEF", "INDF").split(";")
                    form = dict([("word", word), ("Other", [])] + [(cat, []) for cat, tasks in self.CLASS_CATEGORIES.items()])
                    for tag_pre in tag_list:
                        tag = tag_pre.split("+")
                        if tag[0] in self.TAG_TO_CAT:
                            form[self.TAG_TO_CAT[tag[0]]] = tag
                        else:
                            form["Other"] += tag
                    all_forms[lemma] += [form]

        if self.config.schema == "source":
            for id_, (lemma, forms) in enumerate(all_forms.items()):
                res = {"lemma": lemma, "forms": {}}
                for k in ["word", "Other"] + list(self.CLASS_CATEGORIES.keys()):
                    res["forms"][k] = [form[k] for form in forms]
                yield id_, res

        if self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            idx = 0
            for lemma, forms in all_forms.items():
                for form in forms:
                    inflection = form.pop("word")
                    feats = [feat[0] for feat in list(form.values()) if feat and feat[0] in self.CLASS_LABELS]
                    example = {
                        "id": idx,
                        "text_1": lemma,
                        "text_2": inflection,
                        "label": feats,
                    }
                    idx += 1
                    yield idx, example
