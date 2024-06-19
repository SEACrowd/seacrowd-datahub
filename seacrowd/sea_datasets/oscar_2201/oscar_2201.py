import re
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@inproceedings{abadji2022cleaner,
    author    = {Julien Abadji and
                Pedro Javier Ortiz Su{\'{a}}rez and
                Laurent Romary and
                Beno{\^{\i}}t Sagot},
    title     = {Towards a Cleaner Document-Oriented Multilingual Crawled Corpus},
    booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference,
                {LREC} 2022, Marseille, France, 20-25 June 2022},
    pages     = {4344--4355},
    publisher = {European Language Resources Association},
    year      = {2022},
    url       = {https://aclanthology.org/2022.lrec-1.463},
}

@inproceedings{abadji2021ungoliant,
    author    = {Julien Abadji and
                Pedro Javier Ortiz Su{\'a}rez and
                Laurent Romary and
                Beno{\^i}t Sagot},
    title     = {Ungoliant: An optimized pipeline for the generation of a very large-scale multilingual web corpus},
    series    = {Proceedings of the Workshop on Challenges in the Management of Large Corpora
                (CMLC-9) 2021. Limerick, 12 July 2021 (Online-Event)},
    editor    = {Harald L{\"u}ngen and
                Marc Kupietz and
                Piotr Bański and
                Adrien Barbaresi and
                Simon Clematide and
                Ines Pisetta},
    publisher = {Leibniz-Institut f{\"u}r Deutsche Sprache},
    address   = {Mannheim},
    doi       = {10.14618/ids-pub-10468},
    url       = {https://nbn-resolving.org/urn:nbn:de:bsz:mh39-104688},
    pages     = {1 -- 9},
    year      = {2021},
    abstract  = {Since the introduction of large language models in Natural Language
    Processing, large raw corpora have played a crucial role in Computational Linguistics.
    However, most of these large raw corpora are either available only for English or not
    available to the general public due to copyright issues. Nevertheless, there are some
    examples of freely available multilingual corpora for training Deep Learning NLP
    models, such as the OSCAR and Paracrawl corpora. However, they have quality issues,
    especially for low-resource languages. Moreover, recreating or updating these corpora
    is very complex. In this work, we try to reproduce and improve the goclassy pipeline
    used to create the OSCAR corpus. We propose a new pipeline that is faster, modular,
    parameterizable, and well documented. We use it to create a corpus similar to OSCAR
    but larger and based on recent data. Also, unlike OSCAR, the metadata information is
    at the document level. We release our pipeline under an open source license and
    publish the corpus under a research-only license.},
    language  = {en}
}

@article{kreutzer2022quality,
    title     = {Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets},
    author    = {Kreutzer, Julia  and
                Caswell, Isaac  and
                Wang, Lisa  and
                Wahab, Ahsan  and
                van Esch, Daan  and
                Ulzii-Orshikh, Nasanbayar  and
                Tapo, Allahsera  and
                Subramani, Nishant  and
                Sokolov, Artem  and
                Sikasote, Claytone  and
                Setyawan, Monang  and
                Sarin, Supheakmungkol  and
                Samb, Sokhar  and
                Sagot, Beno{\^\i}t  and
                Rivera, Clara  and
                Rios, Annette  and
                Papadimitriou, Isabel  and
                Osei, Salomey  and
                Suarez, Pedro Ortiz  and
                Orife, Iroro  and
                Ogueji, Kelechi  and
                Rubungo, Andre Niyongabo  and
                Nguyen, Toan Q.  and
                M{\"u}ller, Mathias  and
                M{\"u}ller, Andr{\'e}  and
                Muhammad, Shamsuddeen Hassan  and
                Muhammad, Nanda  and
                Mnyakeni, Ayanda  and
                Mirzakhalov, Jamshidbek  and
                Matangira, Tapiwanashe  and
                Leong, Colin  and
                Lawson, Nze  and
                Kudugunta, Sneha  and
                Jernite, Yacine  and
                Jenny, Mathias  and
                Firat, Orhan  and
                Dossou, Bonaventure F. P.  and
                Dlamini, Sakhile  and
                de Silva, Nisansa  and
                {\c{C}}abuk Ball{\i}, Sakine  and
                Biderman, Stella  and
                Battisti, Alessia  and
                Baruwa, Ahmed  and
                Bapna, Ankur  and
                Baljekar, Pallavi  and
                Azime, Israel Abebe  and
                Awokoya, Ayodele  and
                Ataman, Duygu  and
                Ahia, Orevaoghene  and
                Ahia, Oghenefego  and
                Agrawal, Sweta  and
                Adeyemi, Mofetoluwa},
    editor    = {Roark, Brian  and
                Nenkova, Ani},
    journal   = {Transactions of the Association for Computational Linguistics},
    volume    = {10},
    year      = {2022},
    address   = {Cambridge, MA},
    publisher = {MIT Press},
    url       = {https://aclanthology.org/2022.tacl-1.4},
    doi       = {10.1162/tacl_a_00447},
    pages     = {50--72},
    abstract  = {With the success of large-scale pre-training and multilingual modeling in
    Natural Language Processing (NLP), recent years have seen a proliferation of large,
    Web-mined text datasets covering hundreds of languages. We manually audit the quality
    of 205 language-specific corpora released with five major public datasets (CCAligned,
    ParaCrawl, WikiMatrix, OSCAR, mC4). Lower-resource corpora have systematic issues: At
    least 15 corpora have no usable text, and a significant fraction contains less than
    50{\%} sentences of acceptable quality. In addition, many are mislabeled or use
    nonstandard/ambiguous language codes. We demonstrate that these issues are easy to
    detect even for non-proficient speakers, and supplement the human audit with automatic
    analyses. Finally, we recommend techniques to evaluate and improve multilingual
    corpora and discuss potential risks that come with low-quality data releases.},
}

@inproceedings{ortizsuarez2020monolingual,
    title     = {A Monolingual Approach to Contextualized Word Embeddings for Mid-Resource Languages},
    author    = {Ortiz Su{'a}rez, Pedro Javier  and
                Romary, Laurent  and
                Sagot, Benoit},
    booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    month     = {jul},
    year      = {2020},
    address   = {Online},
    publisher = {Association for Computational Linguistics},
    url       = {https://www.aclweb.org/anthology/2020.acl-main.156},
    pages     = {1703--1714},
    abstract  = {We use the multilingual OSCAR corpus, extracted from Common Crawl via
    language classification, filtering and cleaning, to train monolingual contextualized
    word embeddings (ELMo) for five mid-resource languages. We then compare the
    performance of OSCAR-based and Wikipedia-based ELMo embeddings for these languages on
    the part-of-speech tagging and parsing tasks. We show that, despite the noise in the
    Common-Crawl-based OSCAR data, embeddings trained on OSCAR perform much better than
    monolingual embeddings trained on Wikipedia. They actually equal or improve the
    current state of the art in tagging and parsing for all five languages. In particular,
    they also improve over multilingual Wikipedia-based contextual embeddings
    (multilingual BERT), which almost always constitutes the previous state of the art,
    thereby showing that the benefit of a larger, more diverse corpus surpasses the
    cross-lingual benefit of multilingual embedding architectures.},
}

@inproceedings{ortizsuarez2019asynchronous,
    author    = {Pedro Javier {Ortiz Su{'a}rez} and
                Benoit Sagot and
                Laurent Romary},
    title     = {Asynchronous pipelines for processing huge corpora on medium to low resource infrastructures},
    series    = {Proceedings of the Workshop on Challenges in the Management of Large Corpora
                (CMLC-7) 2019. Cardiff, 22nd July 2019},
    editor    = {Piotr Bański and
                Adrien Barbaresi and
                Hanno Biber and
                Evelyn Breiteneder and
                Simon Clematide and
                Marc Kupietz and
                Harald L{"u}ngen and
                Caroline Iliadi},
    publisher = {Leibniz-Institut f{"u}r Deutsche Sprache},
    address   = {Mannheim},
    doi       = {10.14618/ids-pub-9021},
    url       = {http://nbn-resolving.de/urn:nbn:de:bsz:mh39-90215},
    pages     = {9 -- 16},
    year      = {2019},
    abstract  = {Common Crawl is a considerably large, heterogeneous multilingual corpus
    comprised of crawled documents from the internet, surpassing 20TB of data and
    distributed as a set of more than 50 thousand plain text files where each contains
    many documents written in a wide variety of languages. Even though each document has a
    metadata block associated to it, this data lacks any information about the language in
    which each document is written, making it extremely difficult to use Common Crawl for
    monolingual applications. We propose a general, highly parallel, multithreaded
    pipeline to clean and classify Common Crawl by language; we specifically design it so
    that it runs efficiently on medium to low resource infrastructures where I/O speeds
    are the main constraint. We develop the pipeline so that it can be easily reapplied to
    any kind of heterogeneous corpus and so that it can be parameterised to a wide range
    of infrastructures. We also distribute a 6.3TB version of Common Crawl, filtered,
    classified by language, shuffled at line level in order to avoid copyright issues, and
    ready to be used for NLP applications.},
    language  = {en}
}
"""

_DATASETNAME = "oscar_2201"
_DESCRIPTION = """\
OSCAR or Open Super-large Crawled Aggregated coRpus is a huge multilingual corpus
obtained by language classification and filtering of the Common Crawl corpus using
the ungoliant architecture. Data is distributed by language in both original and
deduplicated form.
"""

_HOMEPAGE = "https://huggingface.co/datasets/oscar-corpus/OSCAR-2201"
_LICENSE = Licenses.CC0_1_0.value
_BASE_URL = "https://huggingface.co/datasets/oscar-corpus/OSCAR-2201/resolve/main/compressed/{lang}_meta/"

_LOCAL = False
_LANGUAGES = ["war", "ceb", "min", "vie", "ilo", "tgl", "lao", "khm", "mya", "jav", "ind", "tha", "sun", "zlm"]

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "2022.1.0"
_SEACROWD_VERSION = "2024.06.20"


class Oscar2201Dataset(datasets.GeneratorBasedBuilder):
    """OSCAR subset for SEA languages, version 2201."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "ssp"
    SUBSETS = ["war", "ceb", "min", "vi", "ta", "ilo", "tl", "lo", "km", "my", "jv", "id", "th", "su", "ms"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} {subset} source schema",
            schema="source",
            subset_id=subset,
        ) for subset in SUBSETS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_seacrowd_ssp",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} {subset} SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id=subset,
        )
        for subset in SUBSETS
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_jv_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "text": datasets.Value("string"),
                    "meta": {
                        "warc_headers": {
                            "warc-record-id": datasets.Value("string"),
                            "warc-date": datasets.Value("string"),
                            "content-type": datasets.Value("string"),
                            "content-length": datasets.Value("int32"),
                            "warc-type": datasets.Value("string"),
                            "warc-identified-content-language": datasets.Value("string"),
                            "warc-refers-to": datasets.Value("string"),
                            "warc-target-uri": datasets.Value("string"),
                            "warc-block-digest": datasets.Value("string"),
                        },
                        "identification": {
                            "label": datasets.Value("string"),
                            "prob": datasets.Value("float"),
                        },
                        "annotations": datasets.Sequence(datasets.Value("string")),
                        "line_identifications": [
                            {
                                "label": datasets.Value("string"),
                                "prob": datasets.Value("float"),
                            }
                        ],
                    },
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.ssp_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        base_path = _BASE_URL.format(lang=self.config.name.split("_")[2])

        checksum_url = urljoin(base_path, "checksum.sha256")
        checksum_path = Path(dl_manager.download(checksum_url))
        with open(checksum_path, encoding="utf-8") as f:
            filenames = [line.split()[1] for line in f if line]
            filenames = sorted(filenames, key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else x)
            data_urls = [urljoin(base_path, filename) for filename in filenames]

        data_paths = list(map(Path, dl_manager.download([url for url in data_urls if url.endswith(".jsonl.gz")])))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": data_paths,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepaths: [Path], split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        key = 0
        for filepath in filepaths:
            with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    if self.config.schema == "source":
                        meta = dict()
                        meta["warc_headers"] = doc["warc_headers"]
                        meta["warc_headers"]["warc-identified-content-language"] = doc["warc_headers"].get("warc-identified-content-language")
                        meta["identification"] = doc["metadata"]["identification"]
                        meta["annotations"] = doc["metadata"]["annotation"]
                        meta["line_identifications"] = doc["metadata"]["sentence_identifications"]
                        yield key, {"id": key, "text": doc["content"], "meta": meta}
                    elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                        yield key, {"id": str(key), "text": doc["content"]}
                    key += 1