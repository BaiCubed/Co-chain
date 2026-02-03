import os
import csv
import json
from openai import OpenAI
from tqdm import tqdm


client = OpenAI(
    api_key="xxx",
    base_url="xxx"
)

input_csv_path = os.path.expanduser("./Auto-SLURP/data/test.csv")
output_json_path = "./explicit_knowledge_graph_test.json"

def build_prompt_for_row(row):
    
    query_text = row.get('sentence', '').replace('"', '\\"')
    intent_text = row.get('intent', '')
    slots_text = row.get('slots', "[]")
    action_text = row.get('action', '')
    
    instructions = """
Your task is to convert user query data into a knowledge graph of SPO triples.
Follow these rules strictly:
1.  The subject for 'has_intent' MUST be the original user query text, prefixed with 'query: '.
2.  The subject for 'implies_action' and 'has_parameter' MUST be the intent itself, prefixed with 'intent: '.
3.  For 'is_value_for', the subject is the text value and the object is the slot name.
4.  Do not return explanations, only the JSON list.

Here is a perfect example:
---
Input Data:
- query: "add dentist appointment for friday at five"
- intent: "calendar_set"
- slots: "['event_name : dentist appointment', 'date : friday', 'time : five']"
- action: "set"
Output JSON:
[
    {"subject": "query: add dentist appointment for friday at five", "predicate": "has_intent", "object": "intent: calendar_set"},
    {"subject": "intent: calendar_set", "predicate": "implies_action", "object": "set"},
    {"subject": "intent: calendar_set", "predicate": "has_parameter", "object": "event_name"},
    {"subject": "intent: calendar_set", "predicate": "has_parameter", "object": "date"},
    {"subject": "intent: calendar_set", "predicate": "has_parameter", "object": "time"},
    {"subject": "dentist appointment", "predicate": "is_value_for", "object": "event_name"},
    {"subject": "friday", "predicate": "is_value_for", "object": "date"},
    {"subject": "five", "predicate": "is_value_for", "object": "time"}
]
---

Now, process the following data:
"""
    
    current_data = f"""
Input Data:
- query: "{query_text}"
- intent: "{intent_text}"
- slots: "{slots_text}"
- action: "{action_text}"

---
Output JSON:
"""
    return instructions + current_data


def extract_triples_from_row(row):
    
    prompt = build_prompt_for_row(row)
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        parsed_json = json.loads(content)

        if isinstance(parsed_json, list):
            return parsed_json
        elif "triples" in parsed_json and isinstance(parsed_json["triples"], list):
             return parsed_json["triples"]
        else:
            return []

    except Exception as e:
        return None

def main():
    if not os.path.exists(input_csv_path):
        return

    all_triples = []
    with open(input_csv_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Extracting Explicit Knowledge"):
            triples = extract_triples_from_row(row)
            if triples:
                all_triples.extend(triples)

    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(all_triples, jsonfile, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete! A total of {len(all_triples)} triples were extracted.")
    print(f"Results have been saved to: {output_json_path}")

if __name__ == "__main__":
    main()