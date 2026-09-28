import json
import os
import re
import random  
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime



API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gemini-3-pro-preview" 
MAX_WORKERS = 50
BATCH_SIZE = 20  

INPUT_PATH = ""
OUTPUT_PATH = ""

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


SYSTEM_PROMPT = """
[ROLE]
User-centered cognitive miner

[TASK]
You will receive one logical memory L and a sequence of semantic memories S, where each S represents a single semantic memory.

Please determine whether each S in the list has a conflict with the given L.

During the processing of L and S, pay attention to inferring the user’s hidden preferences and implicit intentions in L and S, and monitor all conflicts.

[CONCEPTS]

1. Contextual Activation
Consider L as a genuine background circuit within the cognitive miner system.

2. Competition Monitoring
When intention S attempts to load or execute, the cognitive miner monitors the interaction between S and L.

3. Conflict Signal Output
- If S and L cannot be smoothly concurrent, it is determined as Conflict.
- If the situation described by L is clearly an optimal alternative of S, this value dimension competition must also be extracted.
- If there is no signal conflict, output None.

[CRITICAL DISTINCTION]

You must distinguish between "conflict" and "irrelevant information".

Irrelevant information:
L and S discuss topics on different dimensions. They are not considered a conflict as long as they are not logically mutually exclusive.

[OUTPUT FORMAT]

Return JSON only.

{
    "results": [
        {
            "s_id": 0,
            "logic_type": "Specify the type of conflict",
            "reasoning": "Explanation in English",
            "confidence": 0.8
        }
    ]
}

If there is no conflict between L and all semantic memories, return:

{
  "results": []
}
"""



def parse_time(time_str):
    try:
        return datetime.strptime(time_str, "%b %d, %Y, %H:%M:%S")
    except Exception:
        return datetime.min

def clean_and_parse_json(raw_content):
    if not raw_content: return None
    try:
        return json.loads(raw_content.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return None
    return None


def run_pipeline():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    memories = raw_data

    TARGET_DOMAIN_PAIRS = [
        ("Preferences", "States"), ("Plans", "States"), ("Social_Relationships", "States"),
        ("Past_Experience", "States"), ("Preferences", "Assets"), ("Plans", "Assets"),
        ("Goals", "Assets"), ("Social_Relationships", "Personal_Background"),
        ("Plans", "Personal_Background"), ("Preferences", "Personal_Background"), ("Plans", "Plans"),
        ("Preferences", "Goals"), ("Social_Relationships", "Goals"), ("Opinions", "Goals"),
        ("Past_Experience", "Goals"), ("Preferences", "Plans"), ("Social_Relationships", "Plans"),
        ("Preferences", "Social_Relationships"),
        ("Social_Relationships", "Preferences"), ("Plans", "Preferences"), ("Goals", "States")
    ]

    mem_by_tag = {}
    for m in memories:
        tag = m.get("tag")
        mem_by_tag.setdefault(tag, []).append(m)
    
    t_to_d_mapping = {}
    for d_tag, t_tag in TARGET_DOMAIN_PAIRS:
        for t in mem_by_tag.get(t_tag, []):
            t_unique_id = f"{t.get('index')}_{t.get('timestamp')}_{t_tag}"
            
            valid_ds_for_t = []
            for d in mem_by_tag.get(d_tag, []):
                valid_ds_for_t.append(d)
                    
            if valid_ds_for_t:
                if t_unique_id not in t_to_d_mapping:
                    t_to_d_mapping[t_unique_id] = {'t_data': t, 'd_list': []}
                t_to_d_mapping[t_unique_id]['d_list'].extend(valid_ds_for_t)

    batched_candidates = []
    for info in t_to_d_mapping.values():
        target_t = info['t_data']
        all_matched_ds = info['d_list']
        

        unique_ds_dict = {d['memory_content']: d for d in all_matched_ds}
        unique_ds = list(unique_ds_dict.values())
        
        for i in range(0, len(unique_ds), BATCH_SIZE):
            batch_of_ds = unique_ds[i:i + BATCH_SIZE]
            batched_candidates.append((target_t, batch_of_ds))

    print(f"Total prompt batches created: {len(batched_candidates)} (Batch size: {BATCH_SIZE})")

    random.seed(42)
    random.shuffle(batched_candidates)
    
    final_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pair = {executor.submit(process_batch_pair, batch): batch for batch in batched_candidates}
        
        for i, future in enumerate(tqdm(as_completed(future_to_pair), total=len(batched_candidates))):
            processed_count = i + 1
            batch_results = future.result()
            
            if batch_results:
                final_results.extend(batch_results)
            
            if processed_count % 50 == 0:
                print(f"\n{processed_count} | {len(final_results)}")
                with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"\nFound {len(final_results)} Logical Pairs")

if __name__ == "__main__":
    run_pipeline()