import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{goldhahn2012leipzig,
  author    = {Goldhahn, Dirk and Eckart, Thomas and Quasthoff, Uwe and others},
  title     = {Building large monolingual dictionaries at the leipzig corpora collection: From 100 to 200 languages},
  journal   = {LREC},
  volume    = {29},
  year      = {2012},
  url       = {http://www.lrec-conf.org/proceedings/lrec2012/pdf/327_Paper.pdf},
  doi       = {},
  biburl    = {},
  bibsource = {google scholar}
}
"""

_DATASETNAME = "leipzig_corpora"

_DESCRIPTION = """\
This is a collection of corpora in different languages, all built by randomly selecting sentences from web and newspaper sources.
Each language has its own directory containing .txt files that list the words and sentences in the corpus, map words or sentences
to their sources, and show the cooccurrence of words. The 2017 Community version of the collection contains text material crawled
from different websites and contains data for 20 SEA languages.
"""

_HOMEPAGE = "https://wortschatz.uni-leipzig.de/en/download"

_LANGUAGES = ["ban", "bjn", "bew", "bcl", "mya", "ceb", "hil", "ind", "khm", "lao", "zsm", "min", "pam", "pag", "ksw", "tgl", "tha", "vie", "war", "jav", "mad"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    "ban": ["ban_community_2017.tar.gz", "ban_community_2021.tar.gz", "ban-id_web_2013_10K.tar.gz", "ban-id_web_2013_30K.tar.gz", "ban_wikipedia_2021_10K.tar.gz"],
    "bjn": ["bjn_community_2017.tar.gz", "bjn-id_web_2015_10K.tar.gz", "bjn_wikipedia_2021_10K.tar.gz"],
    "bew": ["bew_community_2017.tar.gz"],
    "bcl": ["bcl_community_2017.tar.gz", "bcl_wikipedia_2011_10K.tar.gz", "bcl_wikipedia_2014_10K.tar.gz"],
    "mya": ["mya_community_2017.tar.gz", "mya_community_2022.tar.gz"],
    "ceb": ["ceb_community_2017.tar.gz", "ceb_newscrawl_2011_10K.tar.gz", "ceb_wikipedia_2011_10K.tar.gz", "ceb_wikipedia_2014_300K.tar.gz", "ceb_wikipedia_2016_1M.tar.gz", "ceb_wikipedia_2021_1M.tar.gz"],
    "hil": ["hil_community_2017.tar.gz"],
    "ind": [
        "ind_mixed_2012_1M.tar.gz",
        "ind_mixed_2013_1M.tar.gz",
        "ind_mixed-tufs4_2012_1M.tar.gz",
        "ind_news_2008_300K.tar.gz",
        "ind_news_2009_300K.tar.gz",
        "ind_news_2010_300K.tar.gz",
        "ind_news_2011_300K.tar.gz",
        "ind_news_2012_300K.tar.gz",
        "ind_news_2019_1M.tar.gz",
        "ind_news_2020_1M.tar.gz",
        "ind_news_2022_1M.tar.gz",
        "ind_mixed-tufs4_2012_1M.tar.gz",
        "ind_news_2008_300K.tar.gz",
        "ind_news_2009_300K.tar.gz",
        "ind_news_2010_300K.tar.gz",
        "ind_news_2011_300K.tar.gz",
        "ind_news_2012_300K.tar.gz",
        "ind_news_2019_1M.tar.gz",
        "ind_news_2020_1M.tar.gz",
        "ind_news_2022_1M.tar.gz",
        "ind_news-tufs10_2011_300K.tar.gz",
        "ind_news-tufs11_2012_300K.tar.gz",
        "ind_news-tufs7_2008_300K.tar.gz",
        "ind_news-tufs8_2009_300K.tar.gz",
        "ind_news-tufs9_2010_300K.tar.gz",
        "ind_newscrawl_2011_1M.tar.gz",
        "ind_newscrawl_2012_1M.tar.gz",
        "ind_newscrawl_2015_300K.tar.gz",
        "ind_newscrawl_2016_1M.tar.gz",
        "ind_newscrawl-tufs5_2011_3M.tar.gz",
        "ind_newscrawl-tufs6_2012_3M.tar.gz",
        "ind_web_2011_300K.tar.gz",
        "ind_web_2012_1M.tar.gz",
        "ind-id_web_2013_1M.tar.gz",
        "ind-bn_web_2015_10K.tar.gz",
        "ind-in_web_2015_1M.tar.gz",
        "ind-id_web_2017_1M.tar.gz",
        "ind-com_web_2018_1M.tar.gz",
        "ind-id_web-public_2017_1M.tar.gz",
        "ind_web-tufs12_2011_300K.tar.gz",
        "ind_web-tufs13_2012_3M.tar.gz",
        "ind_web-tufs2_2013_1M.tar.gz",
        "ind_web-tufs3_2015_3M.tar.gz",
        "ind_wikipedia_2010_300K.tar.gz",
        "ind_wikipedia_2014_1M.tar.gz",
        "ind_wikipedia_2016_1M.tar.gz",
        "ind_wikipedia_2021_1M.tar.gz",
        "ind_wikipedia-tufs14_2016_1M.tar.gz",
        "ind_wikipedia-tufs16_2016_30K.tar.gz",
    ],
    "khm": ["ckb_community_2017.tar.gz", "ckb_wikipedia_2016_30K.tar.gz", "ckb_wikipedia_2021_100K.tar.gz"],
    "lao": ["lao_community_2017.tar.gz", "lao_community_2021.tar.gz"],
    "zsm": ["zsm_mixed-tufs4_2012_300K.tar.gz", "zsm_newscrawl-tufs15_2011_100K.tar.gz", "zsm_web-tufs1_2015_10K.tar.gz", "zsm_web-tufs13_2012_300K.tar.gz", "zsm_web-tufs3_2015_10K.tar.gz", "zsm_wikipedia-tufs16_2016_300K.tar.gz"],
    "min": ["min_community_2017.tar.gz", "min-id_web_2013_10K.tar.gz", "min_wikipedia_2014_100K.tar.gz", "min_wikipedia_2016_100K.tar.gz", "min_wikipedia_2021_100K.tar.gz"],
    "pam": ["pam_community_2017.tar.gz", "pam_wikipedia_2010_10K.tar.gz", "pam_wikipedia_2011_10K.tar.gz", "pam_wikipedia_2014_10K.tar.gz", "pam_wikipedia_2016_10K.tar.gz"],
    "pag": ["pag_community_2017.tar.gz"],
    "ksw": ["ksw_community_2017.tar.gz"],
    "tgl": ["tgl_community_2017.tar.gz", "tgl_news_2020_30K.tar.gz", "tgl_newscrwal_2011_300K.tar.gz", "tgl_wikipedia_2014_100K.tar.gz", "tgl_wikipedia_2016_100K.tar.gz", "tgl_wikipedia_2021_100K.tar.gz"],
    "tha": [
        "tha_community_2017.tar.gz",
        "tha_community_2021.tar.gz",
        "tha_news_2020_30K.tar.gz",
        "tha_newscrawl_2011_100K.tar.gz",
        "tha-th_web_2015_100K.tar.gz",
        "tha-th_web_2016_300K.tar.gz",
        "tha-th_web_2018_1M.tar.gz",
        "tha_wikipedia_2016_10K.tar.gz",
        "tha_wikipedia_2021_10K.tar.gz",
    ],
    "vie": [
        "vie_mixed_2014_1M.tar.gz",
        "vie_news_2019_300K.tar.gz",
        "vie_news_2020_1M.tar.gz",
        "vie_news_2022_1M.tar.gz",
        "vie_newscrwal_2011_1M.tar.gz",
        "vie-kh_web_2013_10K.tar.gz",
        "vie-vn_web_2015_1M.tar.gz",
        "vie_wikipedia_2016_1M.tar.gz",
        "vie_wikipedia_2021_1M.tar.gz",
    ],
    "war": ["war_community_2017.tar.gz", "war_wikipedia_2014_300K.tar.gz", "war_wikipedia_2016_300K.tar.gz", "war_wikipedia_2021_10K.tar.gz"],
    "jav": [
        "jav_community_2017.tar.gz",
        "jav-id_web_2013_30K.tar.gz",
        "jav-id_web_2015_30K.tar.gz",
        "jav_wikipedia_2010_10K.tar.gz",
        "jav_wikipedia_2011_30K.tar.gz",
        "jav_wikipedia_2016_100K.tar.gz",
        "jav-bms_wikipedia_2016_10K.tar.gz",
        "jav_wikipedia_2021_100K.tar.gz",
        "jav-bms_wikipedia_2021_10K.tar.gz",
    ],
    "mad": ["mad_community_2017.tar.gz", "mad-id_web_2013_10K.tar.gz"],
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class NewDataset(datasets.GeneratorBasedBuilder):
    """This is a collection of corpora in different languages, all built by randomly selecting sentences from web and newspaper sources."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_ssp",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = "leipzig_corpora_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "language": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_ssp":
            features = schemas.ssp_features
            features["language"] = datasets.Value("string")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        all_sentence_patha = {}
        for lang in _LANGUAGES:
            urls = _URLS[lang]
            data_dir = dl_manager.download_and_extract(["https://downloads.wortschatz-leipzig.de/corpora/" + url for url in urls])
            all_sentence_patha[lang] = self._get_path(data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": all_sentence_patha,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            i = 0
            for lang, path in filepath.items():
                for p in path:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            ex = {
                                "language": lang,
                                "text": line.strip().split("\t")[-1],
                            }
                            yield i, ex
                            i += 1

        elif self.config.schema == "seacrowd_ssp":
            i = 0
            for lang, path in filepath.items():
                for p in path:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            ex = {
                                "id": str(i),
                                "language": lang,
                                "text": line.strip(),
                            }
                            yield i, ex
                            i += 1

    def _get_path(self, filepath):
        """Reads data from the file and returns a list of examples."""
        results = []
        for path in filepath:
            dir = os.listdir(path)
            if len(dir) == 1:
                final_path = os.path.join(path, dir[0])
                dir = os.listdir(final_path)
                for file in dir:
                    if "sentence" in file:
                        results.append(os.path.join(final_path, file))
            else:
                for file in dir:
                    if "sentence" in file:
                        results.append(os.path.join(path, file))
        return results
