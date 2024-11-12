# Re-define and re-run the conversion function as it was cleared

import json
import ast
def convert_txt_to_json_safe(txt_path, json_path):
    data = []
    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # Use ast.literal_eval for safer evaluation
                entry = ast.literal_eval(line.strip())
                data.append(entry)
            except (SyntaxError, ValueError, NameError):
                # Skip lines that cannot be parsed
                continue

    # Write data to JSON file
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# Convert each file to JSON
convert_txt_to_json_safe('dev.txt', 'dev.json')
convert_txt_to_json_safe('test.txt', 'test.json')
convert_txt_to_json_safe('train.txt', 'train.json')