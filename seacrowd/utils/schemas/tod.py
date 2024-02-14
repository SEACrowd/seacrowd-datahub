"""
End-to-end Task Oriented Dialogue Schema

This schema will adapt the WoZ schema that has been used in several end-to-end task-oriented dialogue (TOD) task.
URL: https://huggingface.co/datasets/woz_dialogue
"""
import datasets

"""
EXAMPLE:

[
	{
		"turn_label": [ [ "food", "eritrean" ] ],
		"system_utterance": "",
		"turn_idx": 0,
		"belief_state": [ { "slots": [ [ "food", "eritrean" ] ], "act": "inform" } ],
		"user_utterance": "Are there any eritrean restaurants in town?",
		"system_acts": []
	},
	{
		"turn_label": [ [ "food", "chinese" ] ],
		"system_utterance": "No, there are no eritrean restaurants in town. Would you like a different restaurant? ",
		"turn_idx": 1,
		"belief_state": [ { "slots": [ [ "food", "chinese" ] ], "act": "inform" } ],
		"user_utterance": "How about Chinese food?",
		"system_acts": []
	},
	{
		"turn_label": [ [ "area", "east" ] ],
		"system_utterance": "There is a wide variety of Chinese restaurants, do you have an area preference or a price preference to narrow it down?",
		"turn_idx": 2,
		"belief_state": [ { "slots": [ [ "food", "chinese" ] ], "act": "inform" }, { "slots": [ [ "area", "east" ] ], "act": "inform" } ],
		"user_utterance": "I would like the East part of town.",
		"system_acts": [ [ "area" ] ]
	},
	{
		"turn_label": [ [ "request", "postcode" ], [ "request", "phone" ], [ "request", "address" ] ],
		"system_utterance": "Yu Garden is a chinese restaurant in the east area.",
		"turn_idx": 3,
		"belief_state": [ { "slots": [ [ "slot", "postcode" ] ], "act": "request" }, { "slots": [ [ "slot", "phone" ] ], "act": "request" }, { "slots": [ [ "slot", "address" ] ], "act": "request" }, { "slots": [ [ "food", "chinese" ] ], "act": "inform" }, { "slots": [ [ "area", "east" ] ], "act": "inform" } ],
		"user_utterance": "Could I get the address, phone number, and postcode of Yu Garden?",
		"system_acts": []
	},
	{
		"turn_label": [],
		"system_utterance": "Phone is 01223 248882, address and postcode are 529 Newmarket Road Fen Ditton C.B 5, 8 P.A",
		"turn_idx": 4,
		"belief_state": [ { "slots": [ [ "food", "chinese" ] ], "act": "inform" }, { "slots": [ [ "area", "east" ] ], "act": "inform" } ],
		"user_utterance": "Thank you. That is all the information I needed. Bye bye!",
		"system_acts": []}
]
"""

features = datasets.Features(
    {
        "dialogue_idx": datasets.Value("int32"),
        "dialogue": [
            {
                "turn_label": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                "system_utterance": datasets.Value("string"),
                "turn_idx": datasets.Value("int32"),
                "belief_state": [
                    {
                        "slots": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                        "act": datasets.Value("string"),
                    }
                ],
                "user_utterance": datasets.Value("string"),
                "system_acts": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
            }
        ],
    }
)
