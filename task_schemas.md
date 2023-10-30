# Nusantara Schema Documentation
We have defined a set of lightweight, task-specific schema to help simplify programmatic access to common `nusantara-nlp` datasets. This schema should be implemented for each dataset in addition to a schema that preserves the original dataset format.

### Example Schema and Associated Tasks

- [Knowledge Base (KB)](#knowledge-base)
  - Named entity recognition (NER)
  - Named entity disambiguation/normalization/linking (NED)
  - Event extraction (EE)
  - Relation extraction (RE)
  - Coreference resolution (COREF)
- [Question Answering (QA)](#question-answering)
  - Question answering (QA)
- [Textual Entailment (TE)](#textual-entailment)
  - Textual entailment (TE)
- [Text Pairs (PAIRS)](#text-pairs)
  - Semantic Similarity (STS)
- [Text to Text (T2T)](#text-to-text)
  - Paraphasing (PARA)
  - Translation (TRANSL)
  - Summarization (SUM)
- [Text (TEXT)](#text)
  - Text classification (TXTCLASS)


## Knowledge Base

[Schema Template](seacrowd/utils/schemas/kb.py)

This is a simple container format with minimal nesting that supports a range of common knowledge base construction / information extraction tasks.

- Named entity recognition (NER)
- Named entity disambiguation/normalization/linking (NED)
- Event extraction (EE)
- Relation extraction (RE)
- Coreference resolution (COREF)

```
{
    "id": "ABCDEFG",
    "document_id": "XXXXXX",
    "passages": [...],
    "entities": [...],
    "events": [...],
    "coreferences": [...],
    "relations": [...]
}
```



**Schema Notes**

- `id` fields appear at the top (i.e. document) level and in every sub-component (`passages`, `entities`, `events`, `coreferences`, `relations`). They can be set in any fashion that makes every `id` field in a dataset unique (including `id` fields in different splits like train/validation/test).
- `document_id` should be a dataset provided document id. If not provided in the dataset, it can be set equal to the top level `id`.
- `offsets` contain character offsets into the string that would be created from `" ".join([passage["text"] for passage in passages])`
- `offsets` and `text` are always lists to support discontinous spans. For continuous spans, they will have the form `offsets=[(lo,hi)], text=["text span"]`. For discontinuous spans, they will have the form `offsets=[(lo1,hi1), (lo2,hi2), ...], text=["text span 1", "text span 2", ...]`
- `normalized` sub-component may contain 1 or more normalized links to database entity identifiers.
- `passages` captures document structure such as named sections.
- `entities`,`events`,`coreferences`,`relations` may be empty fields depending on the dataset and specific task.

## Text
- [Schema Template](seacrowd/utils/schemas/text.py)
- Examples: [SmSA](seacrowd/sea_datasets/smsa/smsa.py)

```
{
    "id": "0",
    "text": "meski masa kampanye sudah selesai , bukan berati habis pula upaya mengerek tingkat kedipilihan elektabilitas .",
    "labels": [
        "neutral"
    ]
}
```

## Sequence Labeling
- [Schema Template](seacrowd/utils/schemas/seq_label.py)
- Examples: [BaPOS](seacrowd/sea_datasets/bapos/bapos.py)

```
{
    {
    "id": "0",
    "tokens": [
        "Seorang",
        "penduduk",
        "yang",
        "tinggal",
        "dekat",
        "tempat",
        "kejadian",
        "mengatakan",
        ",",
        "dia",
        "mendengar",
        "suara",
        "tabrakan",
        "yang",
        "keras",
        "dan",
        "melihat",
        "mobil",
        "ambulan",
        "membawa",
        "orang-orang",
        "yang",
        "berlumuran",
        "darah",
        "."
    ],
    "labels": [
        "B-NND",
        "B-NN",
        "B-SC",
        "B-VB",
        "B-JJ",
        "B-NN",
        "B-NN",
        "B-VB",
        "B-Z",
        "B-PRP",
        "B-VB",
        "B-NN",
        "B-NN",
        "B-SC",
        "B-JJ",
        "B-CC",
        "B-VB",
        "B-NN",
        "B-NN",
        "B-VB",
        "B-NN",
        "B-SC",
        "B-VB",
        "B-NN",
        "B-Z"
    ]
}
```

## Text Pairs
- [Schema Template](seacrowd/utils/schemas/pairs.py)
- Examples: [MQP](https://github.com/bigscience-workshop/biomedical/blob/main/examples/mqp.py)

```
{
	"id": "0",
	"document_id": "NULL",
	"text_1": "Am I over weight (192.9) for my age (39)?",
	"text_2": "I am a 39 y/o male currently weighing about 193 lbs. Do you think I am overweight?",
	"label": 1,
}
```

## Question Answering
- [Schema Template](seacrowd/utils/schemas/qa.py)
- Examples: [TyDiQA-ID](seacrowd/sea_datasets/tydiqa_id/tydiqa_id.py)

```
{
	"id": "0",
	"document_id": "24267510",
	"question_id": "55031181e9bde69634000014",
	"question": "Is RANKL secreted from the cells?",
	"type": "yesno",
	"choices": [],
	"context": "Osteoprotegerin (OPG) is a soluble secreted factor that acts as a decoy receptor for receptor activator of NF-\u03baB ligand (RANKL)",
	"answer": ["yes"],
}
```

## Text to Text

- [Schema Template](seacrowd/utils/schemas/text_to_text.py)
- Examples: [ParaMed](https://github.com/bigscience-workshop/biomedical/blob/main/examples/paramed.py)

```
{
	"id": "0",
	"text_1": "Pleasing God doesn"t mean that we must busy ourselves with a new set of "spiritual" activities\n",
	"text_2": "Menyenangkan Allah tidaklah berarti bahwa kita harus menyibukkan diri sendiri dengan berbagai aktivitas rohani\n",
	"text_1_name": "eng",
	"text_2_name": "ind"
}
```

## Self-supervised pretraining
- [Schema Template](seacrowd/utils/schemas/self_supervised_pretraining.py)
- Examples: [CC100](seacrowd/sea_datasets/cc100/cc100.py)

```
{
    "id": "0",
    "text": "Placeholder text. Will change to a real example soon."
}
```

## Speech recognition
- [Schema Template](seacrowd/utils/schemas/speech_text.py)
- Examples: [TITML-IDN](seacrowd/sea_datasets/titml_idn/titml_idn.py)

```
{
    {"id": "01-001",
    "path": ".cache/huggingface/datasets/downloads/extracted/ecbf4ad46b3db9b85aa9108272c39dc75a268b4c0b92f2827866ef17dea97585/01/01-001.wav",
    "audio": {
        "path": ".cache/huggingface/datasets/downloads/extracted/ecbf4ad46b3db9b85aa9108272c39dc75a268b4c0b92f2827866ef17dea97585/01/01-001.wav",
        "array": array([-0.0005188 , -0.00018311, -0.00021362, ..., -0.00018311, -0.00033569, -0.00015259], dtype=float32),
        "sampling_rate": 16000
    },
    "text": "hai selamat pagi apa kabar",
    "speaker": "01",
    "metadata": {"speaker_age": 25, "speaker_gender": "female"}}
}
```
