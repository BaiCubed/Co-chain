import os
import csv
import json
from openai import OpenAI
from tqdm import tqdm
from datetime import date
import re
import time

client = OpenAI(
    api_key="xxx",
    base_url="xxx"
)


input_csv_path = os.path.expanduser("./Auto-SLURP/data/train.csv")
output_json_path = "./prompt_trees.json"


VALID_INTENTS_LIST = """
- calendar: calendar_set, calendar_remove, calendar_query
- lists: lists_query, lists_remove, lists_createoradd
- music: play_music, music_likeness, playlists_createoradd, music_settings, music_dislikeness, music_query
- news: news_query, news_subscription
- alarm: alarm_set, alarm_query, alarm_remove, alarm_change
- email: email_sendemail, email_query, email_querycontact, email_subscription, email_addcontact, email_remove
- iot: iot_hue_lightother, iot_hue_lightcolor, iot_coffee, iot_hue_lightdim, iot_hue_lightup, audio_volume_mute, iot_hue_lightoff, audio_volume_up, iot_wemo_off, audio_volume_other, iot_cleaning, iot_wemo_on, audio_volume_down
- weather: weather_query
- datetime: datetime_query, datetime_convert
- qa: qa_stock, qa_factoid, general_quirky, qa_definition, general_joke, qa_maths
- general: general_greet
- transport: transport_taxi, transport_ticket, transport_query, transport_traffic
- recommendation: recommendation_events, recommendation_movies, recommendation_locations
- social: social_query, social_post
"""

SERVER_URL_LIST = """
- qa server (for qa_*, general_* intents): http://214.10.10.4:3005/qa 
- news server: http://214.10.10.4:3020/news
- weather server: https://geocoding-api.open-meteo.com/v1/search
- alarm server: http://214.10.10.4:3000/alarm
- audiobook server: http://214.10.10.4:3001/audiobook
- calendar server: http://214.10.10.4:3002/calendar
- email server: http://214.10.10.4:3005/email
- iot server: http://214.10.10.4:3007/iot
- lists server: http://214.10.10.4:3008/lists
- music server: http://214.10.10.4:3009/music
- transport server: http://214.10.10.4:3018/transport
- social server: http://214.10.10.4:3015/social
"""

def call_llm(prompt, expect_json=True, retries=3):
    last_exception = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
                response_format={"type": "json_object"} if expect_json else None
            )
            content = response.choices[0].message.content
            if expect_json:
                cleaned_content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
                return json.loads(cleaned_content)
            return content
        except Exception as e:
            last_exception = e
            time.sleep(1)
    
    return None

def generate_prompt_tree_for_row(row):
    prompt_tree = {}
    query = row.get('sentence', '')
    prompt_tree['L0_query'] = query

    prompt_L1 = f"""
    Analyze the user query to extract its intent and slots.
    The 'intent' value MUST be one of the intents from the provided list. Do not invent a new one.
    The 'slots' should be a dictionary of key-value pairs.
    Return a single JSON object with "intent" and "slots".

    Valid Intents List:
    {VALID_INTENTS_LIST}

    User Query: "{query}"
    
    JSON Output:
    """
    l1_result = call_llm(prompt_L1)
    if not l1_result or "intent" not in l1_result or "slots" not in l1_result:
        print(f"ID {row.get('id')} fail in L1.")
        return None
    prompt_tree['L1_intent_and_slots'] = l1_result
    
    today = date.today().strftime("%Y-%m-%d")
    prompt_L2 = f"""
    Analyze the input slots. Convert any relative time/date expressions (e.g., "tomorrow", "eight pm") into standard formats (YYYY-MM-DD, HH:MM).
    Assume the current date is {today}.
    Return a JSON object with only the formatted key-value pairs. If no formatting is needed, return an empty JSON object.

    Input Slots: {json.dumps(l1_result['slots'])}

    JSON Output:
    """
    l2_result = call_llm(prompt_L2)
    if l2_result is None:
        print(f"ID {row.get('id')} fail in L2.")
        return None
    prompt_tree['L2_formatted_params'] = l2_result

    prompt_L3 = f"""
    Given the user intent, select the single most appropriate server URL from the provided list.
    Use the examples as a guide. Return only the plain URL string.

    --- Examples ---
    - intent "calendar_set" -> calendar server
    - intent "weather_query" -> weather server
    - intent "general_joke" -> qa server
    - intent "play_music" -> music server
    - intent "transport_ticket" -> transport server
    --- End Examples ---

    URL List:
    {SERVER_URL_LIST}

    User Intent: "{l1_result['intent']}"

    Selected URL:
    """
    l3_result = call_llm(prompt_L3, expect_json=False)
    if not l3_result:
        print(f"ID {row.get('id')} fail in L3.")
        return None
    l3_result = l3_result.strip().split('\n')[0].replace("Selected URL:", "").strip()
    prompt_tree['L3_selected_url'] = l3_result

    final_params = l1_result['slots'].copy()
    final_params.update(l2_result)
    prompt_L4 = f"""
    Construct the final, executable URL by combining the base URL with the provided parameters as query strings.
    Ensure all parameter values are URL-encoded.
    Return only the final URL string.

    Base URL: {l3_result}
    Parameters: {json.dumps(final_params)}

    Final Executable URL:
    """
    l4_result = call_llm(prompt_L4, expect_json=False)
    if not l4_result:
        print(f"ID {row.get('id')} fail in L4.")
        return None
    prompt_tree['L4_final_executable_command'] = l4_result.strip()

    return prompt_tree

def main():
    if not os.path.exists(input_csv_path):
        return

    all_trees = []
    with open(input_csv_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_to_process = [row for i, row in enumerate(reader) if i < 100]

        for row in tqdm(rows_to_process, desc="Prompt Tree"):
            tree = generate_prompt_tree_for_row(row)
            if tree:
                all_trees.append({"id": row.get('id'), "tree": tree})

    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(all_trees, jsonfile, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()