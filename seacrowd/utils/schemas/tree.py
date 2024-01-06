"""
Tree Schema

This schema assumes a document with subnodes elements
and a tree hierarchy.

For example:
            NODE1    .....
        //
ROOT    -   NODE2    .....
        \\
            NODE3    .....
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "passage": {
            "id": datasets.Value("string"),
            "type": datasets.Value("string"),
            "text": datasets.Sequence(datasets.Value("string")),
            "offsets": datasets.Sequence(datasets.Value("int32")),
        },
        "nodes": [
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                "text": datasets.Value("string"),
                "offsets": datasets.Sequence(datasets.Value("int32")),
                "subnodes": datasets.Sequence(datasets.Value("string")),  # ids of subnodes
            }
        ],
    }
)
