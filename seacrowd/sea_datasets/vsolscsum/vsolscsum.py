# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The Vietnamese dataset for social context summarization \
    The dataset contains 141 open-domain articles along with \
    3,760 sentences, 2,448 extracted standard sentences and \
    comments as standard summaries and 6,926 comments in 12 \
    events. This dataset was manually annotated by human. \
    Note that the extracted standard summaries also include comments.\
    The label of a sentence or comment was generated based on the \
    voting among social annotators. For example, given a sentence, \
    each annotator makes a binary decision in order to indicate \
    that whether this sentence is a summary candidate (YES) or not \
    (NO). If three annotators agree yes, this sentences is labeled by 3. \
    Therefore, the label of each sentence or comment ranges from 1 to 5\
    (1: very poor, 2: poor, 3: fair, 4: good; 5: perfect). The standard \
    summary sentences are those which receive at least three agreements \
    from annotators. The inter-agreement calculated by Cohen's Kappa \
    after validation among annotators is 0.685.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@inproceedings{nguyen-etal-2016-vsolscsum,
    title = "{VS}o{LSCS}um: Building a {V}ietnamese Sentence-Comment Dataset for Social Context Summarization",
    author = "Nguyen, Minh-Tien  and
      Lai, Dac Viet  and
      Do, Phong-Khac  and
      Tran, Duc-Vu  and
      Nguyen, Minh-Le",
    editor = "Hasida, Koiti  and
      Wong, Kam-Fai  and
      Calzorari, Nicoletta  and
      Choi, Key-Sun",
    booktitle = "Proceedings of the 12th Workshop on {A}sian Language Resources ({ALR}12)",
    month = dec,
    year = "2016",
    address = "Osaka, Japan",
    publisher = "The COLING 2016 Organizing Committee",
    url = "https://aclanthology.org/W16-5405",
    pages = "38--48",
    abstract = "This paper presents VSoLSCSum, a Vietnamese linked sentence-comment dataset, which was manually created to treat the lack of standard corpora for social context summarization in Vietnamese. The dataset was collected through the keywords of 141 Web documents in 12 special events, which were mentioned on Vietnamese Web pages. Social users were asked to involve in creating standard summaries and the label of each sentence or comment. The inter-agreement calculated by Cohen{'}s Kappa among raters after validating is 0.685. To illustrate the potential use of our dataset, a learning to rank method was trained by using a set of local and social features. Experimental results indicate that the summary model trained on our dataset outperforms state-of-the-art baselines in both ROUGE-1 and ROUGE-2 in social context summarization.",
}
"""

_DATASETNAME = "vsolscsum"

_DESCRIPTION = """
The Vietnamese dataset for social context summarization \
    The dataset contains 141 open-domain articles along with \
    3,760 sentences, 2,448 extracted standard sentences and \
    comments as standard summaries and 6,926 comments in 12 \
    events. This dataset was manually annotated by human. \
    Note that the extracted standard summaries also include comments.\
    The label of a sentence or comment was generated based on the \
    voting among social annotators. For example, given a sentence, \
    each annotator makes a binary decision in order to indicate \
    that whether this sentence is a summary candidate (YES) or not \
    (NO). If three annotators agree yes, this sentences is labeled by 3. \
    Therefore, the label of each sentence or comment ranges from 1 to 5\
    (1: very poor, 2: poor, 3: fair, 4: good; 5: perfect). The standard \
    summary sentences are those which receive at least three agreements \
    from annotators. The inter-agreement calculated by Cohen's Kappa \
    after validation among annotators is 0.685.
"""

_HOMEPAGE = "https://github.com/nguyenlab/VSoLSCSum-Dataset"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/nguyenlab/VSoLSCSum-Dataset/master/VSoSLCSum.xml",
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class VSolSCSumDataset(datasets.GeneratorBasedBuilder):
    """
    The Vietnamese dataset for social context summarization \
    The dataset contains 141 open-domain articles along with \
    3,760 sentences, 2,448 extracted standard sentences and \
    comments as standard summaries and 6,926 comments in 12 \
    events. This dataset was manually annotated by human. \
    Note that the extracted standard summaries also include comments.\
    The label of a sentence or comment was generated based on the \
    voting among social annotators. For example, given a sentence, \
    each annotator makes a binary decision in order to indicate \
    that whether this sentence is a summary candidate (YES) or not \
    (NO). If three annotators agree yes, this sentences is labeled by 3. \
    Therefore, the label of each sentence or comment ranges from 1 to 5\
    (1: very poor, 2: poor, 3: fair, 4: good; 5: perfect). The standard \
    summary sentences are those which receive at least three agreements \
    from annotators. The inter-agreement calculated by Cohen's Kappa \
    after validation among annotators is 0.685.
    """

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
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "post_id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                    "document_and_comment": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_path = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r", encoding="utf-8") as file:
            xml_content = file.read()

        root = ET.fromstring(xml_content)

        def extract_data_from_xml(root):
            data = []

            for post in root.findall(".//post"):
                post_id = post.get("id")
                title = post.find("title").text
                summary_sentences = [sentence.find("content").text for sentence in post.find(".//summary").find("sentences").findall("sentence")]
                document_sentences = [sentence.find("content").text for sentence in post.find(".//document").find("sentences").findall("sentence")]
                comment_sentences = [sentence.find("content").text for sentence in post.find(".//comments").find(".//comment").find("sentences").findall("sentence")]

                summary_text = " ".join(summary_sentences)
                document_text = " ".join(document_sentences)
                comment_text = " ".join(comment_sentences)

                data.append(
                    {
                        "post_id": post_id,
                        "title": title,
                        "summary": summary_text,
                        "document_and_comment": f"{document_text} | {comment_text}",
                    }
                )

            return data

        extracted_data = extract_data_from_xml(root)
        df = pd.DataFrame(extracted_data)

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == "seacrowd_t2t":

                example = {
                    "id": str(row["post_id"]),
                    "text_1": str(row["summary"]),
                    "text_2": str(row["document_and_comment"]),
                    "text_1_name": "summary",
                    "text_2_name": "document_and_comment",
                }

            yield index, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py

if __name__ == "__main__":
    datasets.load_dataset(__file__)
