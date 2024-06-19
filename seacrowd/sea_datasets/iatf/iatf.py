from pathlib import Path
from typing import Dict, List, Tuple

import datasets

try:
    import pyreadr
except:
    print("Install the `pyreadr` package to use.")

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (TASK_TO_SCHEMA, Licenses, Tasks)

_DATASETNAME = "iatf"

_CITATION = """\
@misc{
   iatf,
   title={Inter-Agency Task Force for the Management of Emerging Infectious Diseases (IATF) COVID-19 Resolutions},
   url={https://como-ph.github.io/post/creating-text-data-from-iatf-resolutions/},
   author={Chris Mercado, John Robert Medina, Ernest Guevarra}
}
"""

_DESCRIPTION = """\
To assess possible impact of various COVID-19 prediction models on Philippine government response, text from various resolutions issued by
the Inter-agency Task Force for the Management of Emerging Infectious Diseases (IATF) has been collected using data mining approaches implemented in R.
"""

_HOMEPAGE = "https://github.com/como-ph/covidphtext/tree/master/data"

_LICENSE = Licenses.GPL_3_0.value

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SEACROWD_SCHEMA_NAME = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()
_LANGUAGES = ["fil"]
_LOCAL = False
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_URL_BASE = "https://github.com/como-ph/covidphtext/raw/master/data/"
_URLS = [
   "iatfGuidelineOmnibus.rda",
   "iatfResolution01.rda",
   "iatfResolution02.rda",
   "iatfResolution03.rda",
   "iatfResolution04.rda",
   "iatfResolution05.rda",
   "iatfResolution06.rda",
   "iatfResolution07.rda",
   "iatfResolution08.rda",
   "iatfResolution09.rda",
   "iatfResolution10.rda",
   "iatfResolution11.rda",
   "iatfResolution12.rda",
   "iatfResolution13.rda",
   "iatfResolution14.rda",
   "iatfResolution15.rda",
   "iatfResolution16.rda",
   "iatfResolution17.rda",
   "iatfResolution18.rda",
   "iatfResolution19.rda",
   "iatfResolution20.rda",
   "iatfResolution21.rda",
   "iatfResolution22.rda",
   "iatfResolution23.rda",
   "iatfResolution24.rda",
   "iatfResolution25.rda",
   "iatfResolution26.rda",
   "iatfResolution27.rda",
   "iatfResolution28.rda",
   "iatfResolution29.rda",
   "iatfResolution30.rda",
   "iatfResolution30A.rda",
   "iatfResolution31.rda",
   "iatfResolution32.rda",
   "iatfResolution33.rda",
   "iatfResolution34.rda",
   "iatfResolution35.rda",
   "iatfResolution36.rda",
   "iatfResolution37.rda",
   "iatfResolution38.rda",
   "iatfResolution39.rda",
   "iatfResolution40.rda",
   "iatfResolution41.rda",
   "iatfResolution42.rda",
   "iatfResolution43.rda",
   "iatfResolution44.rda",
   "iatfResolution45.rda",
   "iatfResolution46.rda",
   "iatfResolution46A.rda",
   "iatfResolution47.rda",
   "iatfResolution48.rda",
   "iatfResolution49.rda",
   "iatfResolution50.rda",
   "iatfResolution50A.rda",
   "iatfResolution51.rda",
   "iatfResolution52.rda",
   "iatfResolution53.rda",
   "iatfResolution54.rda",
   "iatfResolution55.rda",
   "iatfResolution55A.rda",
   "iatfResolution56.rda",
   "iatfResolution57.rda",
   "iatfResolution58.rda",
   "iatfResolution59.rda",
   "iatfResolution60.rda",
   "iatfResolution60A.rda",
]


class IATFDataset(datasets.GeneratorBasedBuilder):
    """Inter-agency Task Force for the Management of Emerging Infectious Diseases Dataset"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{_SEACROWD_SCHEMA_NAME}",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} seacrowd schema",
            schema=f"seacrowd_{_SEACROWD_SCHEMA_NAME}",
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
        elif self.config.schema == f"seacrowd_{_SEACROWD_SCHEMA_NAME}":
            features = schemas.self_supervised_pretraining.features
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
        filepaths = [Path(dl_manager.download(_URL_BASE + url)) for url in _URLS]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": filepaths},
            ),
        ]

    def _generate_examples(self, filepaths: List[Path]) -> Tuple[int, Dict]:
        counter = 0
        for path in filepaths:
            data = pyreadr.read_r(path)
            text = " ".join([str(x) for x in data[list(data.keys())[0]]["text"].values])
            if self.config.schema == "source":
                yield (
                    counter,
                    {
                        "id": str(counter),
                        "text": text.strip(),
                    },
                )
            elif self.config.schema == f"seacrowd_{_SEACROWD_SCHEMA_NAME}":
                yield (
                    counter,
                    {
                        "id": str(counter),
                        "text": text.strip(),
                    },
                )

            counter += 1
