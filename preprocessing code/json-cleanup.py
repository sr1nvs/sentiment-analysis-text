import json

def check_jsonl_file(file_path):
    with open(file_path, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            try:
                entry = json.loads(line)
                text = entry.get('text')
                label = entry.get('label')

                if text is None or not isinstance(text, str):
                    print(f"Error in line {line_number}: Invalid or missing text.")
                if label is None or not isinstance(label, int):
                    print(f"Error in line {line_number}: Invalid or missing label.")
            except json.JSONDecodeError:
                print(f"Error in line {line_number}: JSON decoding error.")

check_jsonl_file('test.jsonl')