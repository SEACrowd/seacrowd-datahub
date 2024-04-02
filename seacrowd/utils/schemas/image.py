"""
General Image Classification Schema

The field "metadata" is not specified to allow some flexibility.
On how to use "metadata", choose one:
1. defining as empty dict if you don't think it's usable in
    `_generate_examples`, or
2. defining meta as dict of key with intended colname meta and its val with
    dataset.Features class in `_info` Dataloader method then populate it with the
    values in `_general_examples` Dataloader method
"""

import datasets


def features(label_names=["Yes", "No"]):
    return datasets.Features(
        {
            "id": datasets.Value("string"),
            "labels": datasets.ClassLabel(names=label_names),
            "image_path": datasets.Value("string"),
            "metadata": {},
        }
    )


def multi_features(label_names=["Yes", "No"]):
    return datasets.Features(
        {
            "id": datasets.Value("string"),
            "labels": datasets.Sequence(datasets.ClassLabel(names=label_names)),
            "image_path": datasets.Value("string"),
            "metadata": {},
        }
    )
