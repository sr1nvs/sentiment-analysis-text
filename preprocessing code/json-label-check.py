import json

input_file = "out.jsonl"

unique_labels = set()

with open(input_file, "r") as jsonl_file:
    for line in jsonl_file:
        data = json.loads(line)
        label = data["label"]
        unique_labels.add(label)

num_unique_labels = len(unique_labels)

print(f"Number of unique labels: {num_unique_labels}")

''' 

0 (sadness)
1 (joy)
2 (love)
3 (anger)
4 (fear)
5 (surprise)

'''