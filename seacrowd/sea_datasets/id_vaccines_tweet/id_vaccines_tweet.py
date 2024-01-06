import datasets
import pandas

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                      DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "id_vaccines"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@article{Febriyanti_Nursidah_Gustiara_Yulianti_2022, title={ANALISIS SENTIMEN MASYARAKAT INDONESIA TERHADAP PELAKSANAAN
VAKSIN COVID’19}, volume={13}, url={https://journal.uii.ac.id/khazanah/article/view/21204},
DOI={10.20885/khazanah.vol13.iss2.art4}, abstractNote={&amp;lt;p&amp;gt;Adanya pandemi Covid’19 di Indonesia
memberikan dampak besar ke berbagai sektor&amp;lt;br /&amp;gt;kehidupan. Hal ini menyebabkan pemerintah memberikan
respon cepat untuk mengatasi&amp;lt;br /&amp;gt;kasus ini dengan pembuatan berbagai kebijakan, salah satunya adalah
kebijakan baru&amp;lt;br /&amp;gt;mengenai vaksinasi yang diwajibkan untuk semua masyarakat di Indonesia.
Kebijakan&amp;lt;br /&amp;gt;tersebut menimbulkan berbagai macam respon dari masyarakat yang kebanyakan&amp;lt;
br /&amp;gt;disalurkan melalui media sosial. Adanya respon dari masyarakat tersebut dapat menjadi&amp;lt;br
/&amp;gt;salah satu acuan untuk melakukan evaluasi terhadap pelaksanaan vaksinasi di Indonesia.&amp;lt;br /&amp;
gt;Maka dari itu, dengan memanfaatkan data dari media sosial twitter, penelitian ini&amp;lt;br /&amp;gt;bertujuan
 untuk menganalisis respon masyarakat terhadap pelaksanaan vaksinasi&amp;lt;br /&amp;gt;dengan cara mengklasifikasikan
 respon tersebut ke dalam respon positif, negatif, dan&amp;lt;br /&amp;gt;netral. Hasil analisis diperoleh bahwa
 respon masyarakat selama 3 bulan, yaitu pada&amp;lt;br /&amp;gt;bulan Mei dan Juni masih terdapat respon netral
 sedangkan pada bulan Juli keseluruhan&amp;lt;br /&amp;gt;kata memiliki respon negatif seiring dengan kenaikan
 kasus Covid’19.&amp;lt;/p&amp;gt;
&amp;lt;p&amp;gt;&amp;lt;br /&amp;gt;&amp;lt;strong&amp;gt;Kata Kunci:&amp;lt;/strong&amp;gt; covid’19, vaksinasi,
pelaksanaan vaksin, kartu/sertifikat vaksin, sentimen&amp;lt;br /&amp;gt;analisis&amp;lt;/p&amp;gt;}, number={2},
journal={Khazanah: Jurnal Mahasiswa}, author={Febriyanti, Syintya and Nursidah, Dea Ratu and Gustiara, Dela and
Yulianti, Rika}, year={2022}, month={May} }
"""

_DESCRIPTION = """\
Dataset containing tweets about COVID-19 vaccines with manually labelled information about whether they are a
subjective tweet and their sentiment polarity. Tweets are from 20-27 June 2021 and 15-22 July 2021.
"""

_HOMEPAGE = "	https://github.com/rayendito/id-vaccines-tweets"

_LICENSE = "UNKNOWN"

_URL = "https://raw.githubusercontent.com/rayendito/id-vaccines-tweets/main/id_vaccines_tweets.csv"

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "0.0.1"


class id_vaccines_tweetDataset(datasets.GeneratorBasedBuilder):
    """Dataset containing tweets about COVID-19 vaccines with manually labelled information about whether they are
    a subjective tweet and their sentiment polarity. Tweets are from 20-27 June 2021 and 15-22 July 2021."""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="id_vaccines_tweet_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Dataset containing tweets about COVID-19 vaccines with manually labelled information about " "whether they are a subjective tweet and their sentiment polarity. " "Tweets are from 20-27 June 2021 and 15-22 July 2021.",
            schema="source",
            subset_id="id_vaccines_tweet",
        ),
        SEACrowdConfig(
            name="id_vaccines_tweet_seacrowd_text",
            version=datasets.Version(_SOURCE_VERSION),
            description="Dataset containing tweets about COVID-19 vaccines with manually labelled information about " "whether they are a subjective tweet and their sentiment polarity. " "Tweets are from 20-27 June 2021 and 15-22 July 2021.",
            schema="seacrowd.text",
            subset_id="id_vaccines_tweet",
        ),
    ]

    DEFAULT_CONFIG_NAME = "id_vaccines_tweet_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "idx": datasets.Value("string"),
                    "form_text": datasets.Value("string"),
                    "norm_text": datasets.Value("string"),
                    "subjective": datasets.Value("float"),
                    "sentiment": datasets.Value("float"),
                }
            )
        elif self.config.schema == "seacrowd.text":
            features = schemas.text_features([-1.0, 0.0, 1.0])
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """ "return splitGenerators"""
        downloaded_files = dl_manager.download_and_extract(_URL)
        print(f"Saved to {downloaded_files}")
        print(f"This dataset is {self.config.name.replace('_source', '')}")

        return [datasets.SplitGenerator(name="datasets.ALL", gen_kwargs={"filepath": downloaded_files})]

    def _generate_examples(self, filepath):
        data_lines = pandas.read_csv(filepath, skip_blank_lines=True)
        keys = data_lines.keys()
        indexes = data_lines[keys[0]][1:]
        norms = data_lines[keys[1]][1:]
        formals = data_lines[keys[2]][1:]
        subjs = data_lines[keys[3]][1:]
        posnegs = data_lines[keys[4]][1:]
        if self.config.schema == "source":
            for idx, (ind, norm, form, subj, posneg) in enumerate(zip(indexes, norms, formals, subjs, posnegs)):
                yield idx, {
                    "idx": str(ind),
                    "form_text": form,
                    "norm_text": norm,
                    "subjective": float(subj),
                    "sentiment": float(posneg),
                }
        if self.config.schema == "seacrowd.text":
            for idx, (ind, norm, posneg) in enumerate(zip(indexes, norms, posnegs)):
                yield idx, {"id": str(ind), "text": norm, "label": float(posneg)}
