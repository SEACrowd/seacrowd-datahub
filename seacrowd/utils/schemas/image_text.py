"""
General ImageText Classification Schema
"""
import datasets

def features(label_names = ["Yes", "No"]):
    return datasets.Features(
        {
            "id": datasets.Value("string"),
            "image_paths": datasets.Sequence(datasets.Value("string")),
            "texts": datasets.Value("string"),
            "metadata": {
                "context": datasets.Value("string"),
                "labels": datasets.Sequence(datasets.ClassLabel(names=label_names)),
                
                # the schema of 'meta' aren't specified either to allow some flexibility
                "meta": {}

                # notes on how to use this field of 'meta'
                # you can choose two of options:
                # 1. defining as empty dict if you don't think it's usable in `_generate_examples`, or
                # 2. defining meta as dict of key with intended colname meta and its val with dataset.Features class
                #    in `_info` Dataloader method then populate it with the values in `_general_examples` Dataloader method
            },
        }
    )

