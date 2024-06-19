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


from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils import schemas
from seacrowd.utils.constants import Tasks, Licenses, TASK_TO_SCHEMA


_CITATION = """\
@article{shiohara2021two,
  title={Two Brunei Malay Texts: A Story of the Maiden Stem and Two Episodes in the History of Weston and Bukau},
  author={Shiohara, Asako and Fitri, Mohd Izzuddin},
  journal={アジア・アフリカの言語と言語学 (Asian and African languages and linguistics)},
  number={15},
  pages={171--190},
  year={2021},
  publisher={アジア・アフリカ言語文化研究所}
}
"""


_DATASETNAME = "melayu_brunei"

_DESCRIPTION = """\
This article gives two texts of Brunei Malay (ISO 639-3: kxd) collected in the town of Weston in Sabah State of Malaysia. 
The texts exhibit linguistic features that are similar to those of Brunei Malay spoken in Brunei Darussalam; 
it has a vowel inventory of only three vowels /a, i, u/, use of the pronoun kitani for the first person plural inclusive
and the use of the base-stem transitive form in patientive voice clauses. One of the texts tells a folk story about 
Batang Dayang and other text includes two episodes: Javanese runaways arriving in Weston and the origin of the name 
Bukau, a town near Weston.
"""

_HOMEPAGE = "https://github.com/matbahasa/Melayu_Brunei"

_LANGUAGES = ['kxd']

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
   'Folklor2-1-01': 'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-1-01.txt',
   'Folklor2-1-02':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-1-02.txt',
   'Folklor2-1-03':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-1-03.txt',
   'Folklor2-1-04':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-1-04.txt',
   'Folklor2-1-05':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-1-05.txt',
   'Folklor2-1-06':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-1-06.txt',
   'Folklor2-1-07':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-1-07.txt',
   'Folklor2-1-08':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-1-08.txt',
   'Folklor2-2-01':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-01.txt',
   'Folklor2-2-02':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-02.txt',
   'Folklor2-2-03':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-03.txt',
   'Folklor2-2-04':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-04.txt',
   'Folklor2-2-05':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-05.txt',
   'Folklor2-2-06':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-06.txt',
   'Folklor2-2-07':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-07.txt',
   'Folklor2-2-08':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-08.txt',
   'Folklor2-2-09':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-09.txt',
   'Folklor2-2-10':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-10.txt',
   'Folklor2-2-11':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-11.txt',
   'Folklor2-2-12':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-12.txt',
   'Folklor2-2-13':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-13.txt',
   'Folklor2-2-14':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-14.txt',
   'Folklor2-2-15':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-15.txt',
   'Folklor2-2-16':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-2-16.txt',
   'Folklor2-3-01':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-3-01.txt',
   'Folklor2-3-02':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-3-02.txt',
   'Folklor2-3-03':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-3-03.txt',
   'Folklor2-3-06':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-3-06.txt',
   'Folklor2-3-07':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-3-07.txt',
   'Folklor2-3-08':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-3-08.txt',
   'Folklor2-3-09':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-3-09.txt',
   'Folklor2-3-10':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-3-10.txt',
   'Folklor2-4-00':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-00.txt',
   'Folklor2-4-01':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-01.txt',
   'Folklor2-4-02':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-02.txt',
   'Folklor2-4-03':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-03.txt',
   'Folklor2-4-04':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-04.txt',
   'Folklor2-4-05':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-05.txt',
   'Folklor2-4-06':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-06.txt',
   'Folklor2-4-07':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-07.txt',
   'Folklor2-4-08':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-08.txt',
   'Folklor2-4-09':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-4-09.txt',
   'Folklor2-5-01':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-5-01.txt',
   'Folklor2-5-02':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-5-02.txt',
   'Folklor2-5-03':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor2-5-03.txt',
   'Folklor3-0-01':'https://raw.githubusercontent.com/matbahasa/Melayu_Brunei/master/Folklor/Folklor3-0-01.txt',
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]


_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class MelayuBruneiDataset(datasets.GeneratorBasedBuilder):
    """This article gives two texts of Brunei Malay (ISO 639-3: kxd) collected in the town of
    Weston in Sabah State of Malaysia. The texts exhibit linguistic features that are similar to those of
    Brunei Malay spoken in Brunei Darussalam; it has a vowel inventory of only three vowels /a, i, u/,
    use of the pronoun kitani for the first person plural inclusive and the use of the base-stem transitive
    form in patientive voice clauses. One of the texts tells a folk story about Batang Dayang and other text
    includes two episodes: Javanese runaways arriving in Weston and the origin of the name Bukau, a town near Weston.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.self_supervised_pretraining.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = [_URLS[key] for key in _URLS.keys()]
        data_path = dl_manager.download_and_extract(urls)


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path[0],
                    "other_path": data_path[1:]
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, other_path: List) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        filepaths = [filepath] + other_path
        data = []
        for filepath in filepaths:
            with open(filepath, "r") as f:
                data.append(" ".join([line.rstrip() for line in f.readlines()]))

        for id, text in enumerate(data):
            yield id, {"id": id, "text": text}