"""
Question Answering Schema
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question_id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "type": datasets.Value("string"),
        "choices": datasets.Sequence(datasets.Value("string")),
        "context": datasets.Value("string"),
        "answer": datasets.Sequence(datasets.Value("string")),
        # fill meta field in `_generate_examples` as empty dict if no info can be added in here
        # the schema aren't specified either to allow some flexibility
        "meta": {}
    }
)
