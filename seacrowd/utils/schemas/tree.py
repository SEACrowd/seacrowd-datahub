"""\
Tree Schema

This schema assumes a document with subnodes elements
and a tree hierarchy.

For example:

                    SUBNODE1 - word1
                  //
            NODE1 - SUBNODE2 - word2
        //
ROOT    -   NODE2 - SUBNODE3 - word3
        \\
            NODE3 - SUBNODE4 - word4
                  \\
                    SUBNODE5 - word5

Schema structure:

        "id": sentence_id,
        "passage": {
            "id": sentence_id,
            "type": None,
            "text": "word1 word2 word3 word4 word5"
            "offsets": [0, 29]
        },
        "nodes": [
            {
                "id": 0,
                "type": ROOT,
                "text": "word1 word2 word3 word4 word5",
                "offsets": [0, 29],
                "subnodes": [1, 2, 3]
            },
            {
                "id": 1,
                "type": NODE1,
                "text": "word1 word2",
                "offsets": [0, 11],
                "subnodes": [4, 5]
            },
            {
                "id": 2,
                "type": NODE2,
                "text": "word3",
                "offsets": [12, 17],
                "subnodes": [6]
            },
            {
                "id": 3,
                "type": NODE3,
                "text": "word4 word5",
                "offsets": [18, 29],
                "subnodes": [7, 8]
            },
            {
                "id": 4,
                "type": SUBNODE1,
                "text": "word1",
                "offsets": [0, 5],
                "subnodes": []
            },
            {
                "id": 5,
                "type": SUBNODE2,
                "text": "word2",
                "offsets": [6, 11],
                "subnodes": []
            },
            {
                "id": 6,
                "type": SUBNODE3,
                "text": "word3",
                "offsets": [12, 17],
                "subnodes": []
            },
            {
                "id": 7,
                "type": SUBNODE4,
                "text": "word4",
                "offsets": [18, 23],
                "subnodes": []
            },
            {
                "id": 8,
                "type": SUBNODE5,
                "text": "word5",
                "offsets": [24, 29],
                "subnodes": []
            }
        ]
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
