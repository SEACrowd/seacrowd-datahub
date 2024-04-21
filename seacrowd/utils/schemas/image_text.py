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
                "annotations": datasets.Sequence({
                    "x1": datasets.Value("int32"),
                    "y1": datasets.Value("int32"),
                    "x2": datasets.Value("int32"),
                    "y2": datasets.Value("int32"),
                    "x3": datasets.Value("int32"),
                    "y3": datasets.Value("int32"),
                    "x4": datasets.Value("int32"),
                    "y4": datasets.Value("int32"),
                    "transcript": datasets.Value("string"),
                })
            },
        }
    )

