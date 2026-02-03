import os
import csv
import json
import requests
from openai import OpenAI
from tqdm import tqdm
import sys
import re
import time
from difflib import SequenceMatcher
from datetime import date

key = 'xxx'
client = OpenAI(
    base_url="xxx",
    api_key=key
)


AUTOSLURP_TEST_DATA_PATH = os.path.expanduser("./Auto-SLURP/data/test.csv")
KNOWLEDGE_GRAPH_PATH = "./explicit_knowledge_graph_test.json"
PROMPT_TREES_PATH = "./prompt_trees.json"
LOG_OUTPUT_DIR = "./cochain_logs_final_claude_test"


VALID_INTENTS_CONTEXT = """
- calendar: calendar_set, calendar_remove, calendar_query
- lists: lists_query, lists_remove, lists_createoradd
- music: play_music, music_likeness, playlists_createoradd, music_settings, music_dislikeness, music_query
- news: news_query, news_subscription
- alarm: alarm_set, alarm_query, alarm_remove, alarm_change
- email: email_sendemail, email_query, email_querycontact, email_subscription, email_addcontact, email_remove
- iot: iot_hue_lightother, iot_hue_lightcolor, iot_coffee, iot_hue_lightdim, iot_hue_lightup, audio_volume_mute, iot_hue_lightoff, audio_volume_up, iot_wemo_off, audio_volume_other, iot_cleaning, iot_wemo_on, audio_volume_down
- weather: weather_query
- datetime: datetime_query, datetime_convert
- stock: qa_stock
- qa: qa_factoid, general_quirky, qa_definition, general_joke, qa_maths
- greet: general_greet
- currency: qa_currency
- transport: transport_taxi, transport_ticket, transport_query, transport_traffic
- recommendation: recommendation_events, recommendation_movies, recommendation_locations
- podcast: play_podcasts
- audiobook: play_audiobook
- radio: play_radio, radio_query
- takeaway: takeaway_query, takeaway_order
- social: social_query, social_post
- cooking: cooking_recipe
- phone: phone_text, phone_notification
- game: play_game
"""

SERVER_MAPPING_CONTEXT = """
- The alarm server at http://127.0.0.1:3000/alarm handles intents: alarm_query, alarm_set, alarm_remove, alarm_change.
- The audiobook server at http://127.0.0.1:3001/audiobook handles intents: play_audiobook.
- The calendar server at http://127.0.0.1:3002/calendar handles intents: calendar_query, calendar_remove, calendar_set.
- The cooking server at http://127.0.0.1:3003/cooking handles intents: cooking_recipe.
- The datetime server at http://127.0.0.1:3004/datetime handles intents: datetime_convert, datetime_query.
- The email server at http://127.0.0.1:3005/email handles intents: email_query, email_sendemail.
- The game server at http://127.0.0.1:3006/game handles intents: play_game.
- The iot server at http://127.0.0.1:3007/iot handles intents like: audio_volume_up, audio_volume_down, iot_cleaning, iot_coffee.
- The lists server at http://127.0.0.1:3008/lists handles intents: lists_query, lists_remove, lists_createoradd.
- The music server at http://127.0.0.1:3009/music handles intents like: play_music, music_query.
- The phone server at http://127.0.0.1:3010/phone handles intents: phone_text, phone_notification.
- The podcasts server at http://127.0.0.1:3011/podcasts handles intents: play_podcasts.
- The radio server at http://127.0.0.1:3013/radio handles intents: play_radio, radio_query.
- The recommendation server at http://127.0.0.1:3014/recommendation handles intents: recommendation_events, recommendation_movies, recommendation_locations.
- The social server at http://127.0.0.1:3015/social handles intents: social_query, social_post.
- The takeaway server at http://127.0.0.1:3017/takeaway handles intents: takeaway_query, takeaway_order.
- The transport server at http://127.0.0.1:3018/transport handles intents: transport_taxi, transport_ticket, transport_query.
- The news subscription server at http://127.0.0.1:3020/news handles the 'news_subscription' intent.
- The weather task (intent: weather_query) is a two-step process: first get coordinates from https://geocoding-api.open-meteo.com/v1/search, then get the forecast from https://api.open-meteo.com/v1/forecast.
- The currency task (intent: qa_currency) uses the API: https://api.freecurrencyapi.com/v1/latest.
- The stock task (intent: qa_stock) is a two-step process: first get a stock symbol using a search API, then get data from http://api.marketstack.com/v1/eod.
- The news query task (intent: news_query) uses the API: http://api.mediastack.com/v1/news.
- The general QA task (intents: qa_factoid, general_joke, etc.) uses the search API: http://api.serpstack.com/search.
"""

def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except FileNotFoundError: print(f"fail'{file_path}'"); sys.exit(1)
    except json.JSONDecodeError as e: print(f"fail '{file_path}': {e}"); sys.exit(1)

def retrieve_knowledge(query_text, knowledge_graph):
    query_keywords = set(re.findall(r'\w+', query_text.lower()))
    relevant_triples = []
    for triple in knowledge_graph:
        subject_keywords = set(re.findall(r'\w+', str(triple.get("subject", "")).lower()))
        object_keywords = set(re.findall(r'\w+', str(triple.get("object", "")).lower()))
        if query_keywords & (subject_keywords | object_keywords):
            relevant_triples.append(triple)
    return relevant_triples[:5]

def retrieve_best_prompt_tree(current_intent, current_query, all_prompt_trees):
    candidate_trees = [item.get("tree") for item in all_prompt_trees if item.get("tree", {}).get("L1_intent_and_slots", {}).get("intent") == current_intent]
    search_space = candidate_trees if candidate_trees else [item.get("tree") for item in all_prompt_trees]
    best_match_tree, highest_similarity = None, -1.0
    for tree in search_space:
        if not tree: continue
        tree_query = tree.get("L0_query", "")
        if not tree_query: continue
        similarity = SequenceMatcher(None, current_query, tree_query).ratio()
        if similarity > highest_similarity:
            highest_similarity, best_match_tree = similarity, tree
    return best_match_tree

def get_intent_from_query(query, retries=2):
    prompt = f"Analyze the query and determine its intent. Choose ONLY from this list: {VALID_INTENTS_CONTEXT}\n\nQuery: \"{query}\"\n\nIntent:"
    for _ in range(retries):
        try:
            response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=50)
            return response.choices[0].message.content.strip()
        except Exception: time.sleep(1)
    return None

def build_cochain_prompt(query_text, knowledge_triples, prompt_tree_example):
    knowledge_str = json.dumps(knowledge_triples, indent=2)
    prompt_tree_str = json.dumps(prompt_tree_example, indent=2)
    prompt = f"""
You are an expert personal assistant system. Your task is to analyze a user's query and generate a final, executable API command in a single step.

---
[CONTEXT]

1.  **Valid Intents List**:
{VALID_INTENTS_CONTEXT}

2.  **Server and Tool Mapping**:
{SERVER_MAPPING_CONTEXT}

3.  **Relevant Knowledge from Knowledge Graph**:
{knowledge_str}

4.  **Prompt Tree Example (Your reasoning template)**:
{prompt_tree_str}

---
[CRITICAL INSTRUCTIONS]

1.  **NEVER Ask Questions**: Your ONLY valid output is a single URL string. Do NOT ask clarifying questions.
2.  **Assume Missing Information**: If a critical piece of information is missing, you MUST make a reasonable assumption to complete the task.
3.  **Default Time Rule**: Specifically, for a 'calendar_set' intent, if the user does not provide a specific time, you MUST assume the time is "09:00".
4.  **Format Dates and Times**: You MUST replace all human-readable dates/times with machine-readable formats ('YYYY-MM-DD', 'HH:MM'). The current date is {date.today().strftime("%Y-%m-%d")}.
5.  **Include Intent Parameter**: The final URL MUST include the determined 'intent' as a query parameter (e.g., `&intent=calendar_set`).
6.  **Default Location Rule**: Specifically, for a 'weather_query' intent, if the user does not provide a specific location ('place_name'), you MUST assume the location is "new york" and add it as the 'name' parameter.
---
[CURRENT TASK]
User Query: "{query_text}"

---
[YOUR OUTPUT]
Generate ONLY the final executable command.

Your Output:
"""
    return prompt

def call_llm_for_command(prompt, retries=3):
    last_exception = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=1024)
            command = response.choices[0].message.content.strip()
            if command.startswith("http"):
                return command
            else: last_exception = ValueError(f"LLM returned a non-URL command: {command}")
        except Exception as e: last_exception = e
        time.sleep(1)
    return None


def run_evaluation():
    knowledge_graph = load_json_data(KNOWLEDGE_GRAPH_PATH)
    prompt_trees = load_json_data(PROMPT_TREES_PATH)
    
    if not os.path.exists(LOG_OUTPUT_DIR):
        os.makedirs(LOG_OUTPUT_DIR)

    with open(AUTOSLURP_TEST_DATA_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        tasks = [row for row in reader]

    for task in tqdm(tasks, desc="eval"):
        task_id = task.get('id')
        query = task.get('sentence') or task.get('query')

        if not query:
            continue
            
        current_intent = get_intent_from_query(query)
        if not current_intent:
            current_intent = ""

        relevant_knowledge = retrieve_knowledge(query, knowledge_graph)
        best_tree_example = retrieve_best_prompt_tree(current_intent, query, prompt_trees)

        if not best_tree_example:
            continue

        final_prompt = build_cochain_prompt(query, relevant_knowledge, best_tree_example)
        
        executable_command = call_llm_for_command(final_prompt)

        log_file_path = os.path.join(LOG_OUTPUT_DIR, f"{task_id}.log")
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"--- Task ID: {task_id} ---\nQuery: {query}\n\n--- Generated Cochain Prompt ---\n{final_prompt}\n\n")
            
            if executable_command:
                log_file.write(f"--- Executable Command from Cochain ---\n{executable_command}\n\n")
                try:
                    log_file.write("--- Server Response ---\n")
                    response = requests.get(executable_command, timeout=10)
                    log_file.write(f"Status Code: {response.status_code}\n{response.text}")
                except requests.exceptions.RequestException as e:
                    log_file.write(f"fail: {e}\n")
            else:
                log_file.write("--- Cochain Failed to Generate Command ---\n")

if __name__ == "__main__":
    run_evaluation()