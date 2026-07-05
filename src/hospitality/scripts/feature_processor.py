import json
from math import floor

import orjson


with open("data/first_iteration/3_featured_extracted_data.jsonl", "r") as f:
    conversations = [orjson.loads(l) for l in f.readlines()]

for c in conversations:
    c["features"]["num_turns"] = len(c["messages"])
    c["features"]["num_user_turns"] = len([
        m for m in c["messages"] if m["role"] == "user"
    ])
    c["features"]["num_bot_turns"] = len([
        m for m in c["messages"] if m["role"] == "bot"
    ])
    c["features"]["avg_user_message_len"] = floor(
        sum([len(m["text"]) for m in c["messages"] if m["role"] == "user"])
        / len([m for m in c["messages"] if m["role"] == "user"])
    )
    c["features"]["booking_confirmed"] = (
        True
        if True in [True for m in c["messages"] if m["booking"] == "confirmed"]
        else False
    )

with open("data/first_iteration/4_complete_featured_data.jsonl", "w") as f:
    f.writelines([json.dumps(c) + "\n" for c in conversations])
