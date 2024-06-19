import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import audiosegment
except:
    print("Please install audiosegment to use the `national_speech_corpus_sg_imda` dataloader.")
import datasets
import pandas as pd

try:
    import textgrid
except:
    print("Please install textgrid to use the `national_speech_corpus_sg_imda` dataloader.")

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{koh19_interspeech,
  author={Jia Xin Koh and Aqilah Mislan and Kevin Khoo and Brian Ang and Wilson Ang and Charmaine Ng and Ying-Ying Tan},
  title={{Building the Singapore English National Speech Corpus}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={321--325},
  doi={10.21437/Interspeech.2019-1525},
  issn={2308-457X}
}
"""

_DATASETNAME = "national_speech_corpus_sg_imda"

_DESCRIPTION = """\
The National Speech Corpus (NSC) is the first large-scale Singapore English corpus spearheaded by the Info-communications and Media Development Authority (IMDA) of Singapore.
It aims to become an important source of open speech data for automatic speech recognition (ASR) research and speech-related applications.
The NSC improves speech engines’ accuracy of recognition and transcription for locally accented English.
The NSC is also able to contribute to speech synthesis technology where an AI voice can be produced that is more familiar to Singaporeans, with local terms pronounced more accurately.
"""

_HOMEPAGE = "https://www.imda.gov.sg/how-we-can-help/national-speech-corpus"

_LANGUAGES = ["eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = f"{Licenses.OTHERS.value} | Singapore Open Data Licence V1.0"

_LOCAL = True

_URLS = {}

# paths of all file locations, presented in a list to support different operating systems
_PATHS = {
    "read_balanced": {
        "metadata": ["PART1", "DOC", "Speaker Information (Part 1).XLSX"],
        "data": {
            "standing_mic": {
                "audio": ["PART1", "DATA", "CHANNEL0", "WAVE"],
                "text": ["PART1", "DATA", "CHANNEL0", "SCRIPT"],
            },
            "boundary_mic": {
                "audio": ["PART1", "DATA", "CHANNEL1", "WAVE"],
                "text": ["PART1", "DATA", "CHANNEL1", "SCRIPT"],
            },
            "phone": {
                "audio": ["PART1", "DATA", "CHANNEL2", "WAVE"],
                "text": ["PART1", "DATA", "CHANNEL2", "SCRIPT"],
            },
        },
    },
    "read_pertinent": {
        "metadata": ["PART2", "DOC", "Speaker Information (Part 2).XLSX"],
        "data": {
            "standing_mic": {
                "audio": ["PART2", "DATA", "CHANNEL0", "WAVE"],
                "text": ["PART2", "DATA", "CHANNEL0", "SCRIPT"],
            },
            "boundary_mic": {
                "audio": ["PART2", "DATA", "CHANNEL1", "WAVE"],
                "text": ["PART2", "DATA", "CHANNEL1", "SCRIPT"],
            },
            "phone": {
                "audio": ["PART2", "DATA", "CHANNEL2", "WAVE"],
                "text": ["PART2", "DATA", "CHANNEL2", "SCRIPT"],
            },
        },
    },
    "conversational_f2f": {
        "metadata": ["PART3", "Documents", "Speakers (Part 3).XLSX"],
        "data": {
            "close_mic": {
                "audio": ["PART3", "Audio Same CloseMic"],
                "text": ["PART3", "Scripts Same"],
            },
            "boundary_mic": {
                "audio": ["PART3", "Audio Same BoundaryMic"],
                "text": ["PART3", "Scripts Same"],
            },
        },
    },
    "conversational_telephone": {
        "metadata": ["PART3", "Documents", "Speakers (Part 3).XLSX"],
        "data": {
            "ivr": {
                "audio": ["PART3", "Audio Separate IVR"],
                "text": ["PART3", "Scripts Separate"],
            },
            "standing_mic": {
                "audio": ["PART3", "Audio Separate StandingMic"],
                "text": ["PART3", "Scripts Separate"],
            },
        },
    },
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "2.0.8"  # should be 2.08 but HuggingFace does not allow

_SEACROWD_VERSION = "2024.06.20"


class NationalSpeechCorpusSgIMDADataset(datasets.GeneratorBasedBuilder):
    """The National Speech Corpus (NSC), the first large-scale Singapore English corpus spearheaded by the Info-communications and Media Development Authority (IMDA) of Singapore."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="national_speech_corpus_sg_imda_source",
            version=SOURCE_VERSION,
            description="national_speech_corpus_sg_imda source schema",
            schema="source",
            subset_id="national_speech_corpus_sg_imda",
        ),
        SEACrowdConfig(
            name="national_speech_corpus_sg_imda_seacrowd_sptext",
            version=SEACROWD_VERSION,
            description="national_speech_corpus_sg_imda SEACrowd schema",
            schema="seacrowd_sptext",
            subset_id="national_speech_corpus_sg_imda",
        ),
    ]

    DEFAULT_CONFIG_NAME = "national_speech_corpus_sg_imda_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_sptext":
            features = schemas.speech_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError(f"This is a local dataset. Please download the data from {_HOMEPAGE} and pass the path as the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        splits = []
        for split_name, data in _PATHS.items():
            metadata_path = os.path.join(*data["metadata"])
            for mic_type, data_path in data["data"].items():
                splits.append(
                    datasets.SplitGenerator(
                        name=f"{split_name}_{mic_type}",
                        gen_kwargs={"filepath": data_dir, "audio_dir": os.path.join(*data_path["audio"]), "text_dir": os.path.join(*data_path["text"]), "split": split_name, "mic_type": mic_type, "metadata_path": metadata_path},
                    )
                )
        return splits

    def _read_part1_part2_text(self, text_path):
        text_data = {}
        with open(text_path, encoding="utf-8-sig") as f:
            for line in f:
                comp = line.split("\t")
                text_id = comp[0].strip()
                if len(text_id) > 1:
                    text_data[text_id] = comp[1].strip()
        return text_data

    def _generate_examples(self, filepath: Path, audio_dir: Path, text_dir: Path, split: str, mic_type: str, metadata_path: str, tmp_cache="~/.cache/nsc") -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        # get speaker info from Excel file
        all_speaker_info = {}
        if split.startswith("read"):
            excel_file = pd.read_excel(open(os.path.join(filepath, metadata_path), "rb"), sheet_name="Speakers", dtype="object")
            column_name = "SCD/PART" + "1" if split == "read_balanced" else 2
            for idx, row in excel_file.iterrows():
                all_speaker_info[str(row[column_name])] = {"gender": row["SEX"]}
        elif split == "conversational_f2f":
            excel_file = pd.read_excel(open(os.path.join(filepath, metadata_path), "rb"), sheet_name="Same Room", dtype="object")
            for idx, row in excel_file.iterrows():
                all_speaker_info[row["SCD"]] = {"age": row["AGE"], "gender": row["SEX"]}
        elif split == "conversational_telephone":
            excel_file = pd.read_excel(open(os.path.join(filepath, metadata_path), "rb"), sheet_name="Separate Room", dtype="object")
            for idx, row in excel_file.iterrows():
                all_speaker_info[f"{row['Conference ID']}_{row['Speaker ID']}"] = {"age": row["Age"], "gender": row["Gender"]}

        for rel_text_path in os.listdir(os.path.join(filepath, text_dir)):
            text_path = os.path.join(filepath, text_dir, rel_text_path)
            text_name, _ext = os.path.splitext(rel_text_path)
            if not os.path.isfile(text_path):
                continue
            if split.startswith("conversational"):
                if split == "conversational_telephone":
                    # conf_{confid}_{confid}_{speaker_id} -> {confid}_{speaker_id}
                    speaker_id = text_name.split("_", 2)[-1]
                    pass
                else:
                    speaker_id = text_name
                speaker_info = all_speaker_info.get(speaker_id, {})
                speaker_info["id"] = speaker_id

                if mic_type in ["close_mic", "standing_mic"]:
                    audio_filename = text_name
                    audio_filepath = audio_filename + ".wav"
                    audio_path = os.path.join(audio_dir, audio_filepath)
                elif mic_type == "boundary_mic":
                    audio_filename = text_name.split("-")[0]
                    audio_filepath = audio_filename + ".wav"
                    audio_path = os.path.join(audio_dir, audio_filepath)
                elif mic_type == "ivr":
                    audio_subdir, audio_filename = text_name.rsplit("_", 1)
                    audio_filepath = audio_filename + ".wav"
                    audio_path = os.path.join(audio_dir, audio_subdir, audio_filepath)

                audio_file = audiosegment.from_file(os.path.join(filepath, audio_path)).resample(sample_rate_Hz=16000)
                text_spans = textgrid.TextGrid.fromFile(text_path)
                audio_name, _ = os.path.splitext(os.path.basename(audio_dir))
                for text_span in text_spans[0]:
                    start, end, text = text_span.minTime, text_span.maxTime, text_span.mark
                    key = f"{audio_name}_{start}_{end}"

                    start_sec, end_sec = int(start * 1000), int(end * 1000)
                    segment = audio_file[start_sec:end_sec]
                    export_dir = os.path.join(tmp_cache, "segmented", audio_name)
                    os.makedirs(export_dir, exist_ok=True)
                    segement_filename = os.path.join(export_dir, f"{audio_filename}-{round(start, 0)}-{round(end, 0)}.wav")
                    segment.export(segement_filename, format="wav")

                    example = {
                        "id": key,
                        "speaker_id": speaker_info["id"],
                        "path": segement_filename,
                        "audio": segement_filename,
                        "text": text,
                    }
                    if self.config.schema == "seacrowd_sptext":
                        example["metadata"] = {
                            "speaker_gender": speaker_info.get("gender", None),  # not all speaker details are available
                            "speaker_age": None,  # speaker age only available in age groups, but SEACrowd schema requires int64
                        }
                    yield key, example

            else:
                text_data = self._read_part1_part2_text(text_path)
                audio_name, session = text_name[1:-1], text_name[-1]
                speaker_id = audio_name
                speaker_info = all_speaker_info.get(speaker_id, {})
                speaker_info["id"] = speaker_id
                for text_id, text in text_data.items():
                    with zipfile.ZipFile(os.path.join(filepath, audio_dir, f"SPEAKER{audio_name}.zip")) as zip_file:
                        zip_path = os.path.join(f"SPEAKER{audio_name}", f"SESSION{session}", f"{text_id}.WAV")
                        extract_path = os.path.join(tmp_cache, audio_dir)
                        os.makedirs(extract_path, exist_ok=True)
                        audio_path = zip_file.extract(zip_path, path=extract_path)

                        key = text_id
                        example = {
                            "id": key,
                            "speaker_id": speaker_info["id"],
                            "path": audio_path,
                            "audio": audio_path,
                            "text": text,
                        }
                        if self.config.schema == "seacrowd_sptext":
                            example["metadata"] = {
                                "speaker_gender": speaker_info.get("gender", None),
                                "speaker_age": None,  # speaker age only available in age groups, but SEACrowd schema requires int64
                            }
                        yield key, example
