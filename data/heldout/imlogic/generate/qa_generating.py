import json
import os
import statistics
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import time

API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gemini-3-pro-preview"
MAX_WORKERS = 20
MAX_RETRIES = 3 




INPUT_FILE = ""
OUTPUT_FILE = ""

# ================= 2. Prompts =================

# Generator Prompt
SYSTEM_PROMPT = """
### Role
You are an expert in generating adversarial evaluation datasets. Your objective is to create test cases where a user's **Distractor (Memory_S)** is logically blocked by a **True Constraint (Memory_L)**.

### Input Data
* **Memory_S (Distractor)**: The user's explicit intent.
* **Memory_L (True Constraint)**: The user's implicit context that creates a conflict.
* **Query_Time**: The timestamp that MUST be used in the query (this is the later time between S and L).

### GENERATION PROCESS & RULES

#### STEP 1: USER QUERY GENERATION
Generate a natural **First-Person** user query that expresses intent corresponding to Memory_S.

* **TIME INCLUSION & CONCISENESS (MANDATORY)**: The query **MUST** rigidly start with the exact date provided in **Query_Time**, followed immediately by the direct question.
    * **DO NOT** include any conversational filler, feelings, backstory, or greetings.
    * **Expected Format**: "[Query_Time], [Direct Question]?" (e.g., "On Sep 28, 2032, should I host a hackathon?").
* **Semantic Alignment**: The query must reflect 1-2 key vocabulary words or the core intent of Memory_S.
* **STRICT BAN ON Memory_L KEYWORDS (Zero Leakage)**:
    * The query is **ABSOLUTELY FORBIDDEN** from mentioning any specific keywords, numbers, medical terms, or entities found in Memory_L.
    * The user must appear completely oblivious to the constraint.
* **Question Types (Select one)**:
    * **Recommendation**: User describes a need/scenario and asks for specific items/actions.
    * **Advice**: User proposes a preliminary idea/plan and asks for an opinion.
    * **Conversation**: User expresses a feeling/intention/opinion, expecting validation.

#### STEP 2: OPTION GENERATION
Generate four distinct response options based on what information the AI has retrieved.
**CRITICAL**: All options must sound like a helpful, polite, and professional AI assistant. They **MUST BE** similar in length, tone, and sentence structure to prevent selection bias.

* **Correct (Full Context: Knows S + Knows L)**
    * **Logic**: Retrieves both S and L.
    * **Action**: Acknowledges the user's want but **REFUSES** or pivots based explicitly on the hard fact from Memory_L. Prioritizes S over L.

* **Trap_Preference (Partial Context: Knows S, Misses L)**
    * **Logic**: Retrieves S, fails to retrieve L.
    * **Action**: **AGREES** with the request solely because it aligns with the user's known preferences (Memory_S). Validates the user's taste ignoring the risk.

* **Trap_Fabrication (Hallucination: Knows S, Misses L + Invents Fact)**
    * **Logic**: Retrieves S, misses L, and **hallucinates** a justification.
    * **Action**: Validates the request by fabricating specific, non-existent contextual conditions. Avoids generic common sense errors in favor of plausible-sounding but false situational facts.

* **Trap_Generic (No Context: Misses S, Misses L)**
    * **Logic**: Fails to retrieve any personal info. Acts as a standard base model.
    * **Action**: Provides a standard, helpful, and universally applicable response. Encourages the action based on general benefits, unaware of the specific personal risk (L).

### Output Format (JSON Only)
{
  "type": "Recommendation/Advice/Conversation",
  "query": "The generated natural, first-person query with Query_Time...",
  "options": {
    "Correct": "...",
    "Trap_Preference": "...",
    "Trap_Fabrication": "...",
    "Trap_Generic": "..."
  },
  "explanation": "Brief analysis of why D is blocked by T."
}
"""

JUDGER_PROMPT = """
You are a strict, expert Judger for adversarial datasets.
Evaluate the provided Generated QA based on these S and L.

### Evaluation Criteria:
1. **Logic**: Is the reasoning logical? S should be correctly blocked or constrained by L.
2. **Semantic & Zero Leakage**: 
   - The query MUST be strongly related to S.
   - The query MUST NOT contain ANY keywords, hints, or direct references to L.
3. **Tone & Naturalness**: The query and options must sound natural and realistic.
4. **Option Bias Avoidance**: The 4 options must NOT have obvious differences in length, style, or format. (Check the provided Length Statistics).
5. **Accuracy**: The 'Correct' option MUST be the best and safest response when knowing both S and L.

### Output Format (JSON Only):
{
    "status": "PASS" or "NOT PASS",
    "error_category": "Query Error" or "Answer Error" or "Other Option Error" or "None",
    "reason": "Provide a brief reason if NOT PASS. Keep it empty if PASS."
}
"""


CURATOR_PROMPT = """
You are an expert Data Curator. The previous generation failed the Judger's evaluation.
Your task is to fix the errors based on the feedback while strictly following the original generation rules.

### Rules to strictly maintain:
1. Query MUST strictly be formatted as "[Query_Time], [Direct Question]?". Absolutely NO conversational filler or background stories.
2. Query MUST reflect Memory_S, but MUST completely hide Memory_L.
3. The 4 options MUST have similar lengths and formatting to avoid bias.
4. The 'Correct' option must refuse/pivot based on L.

### Output Format (JSON Only):
{
  "type": "Recommendation/Advice/Conversation",
  "query": "Fixed concise query strictly starting with Query_Time...",
  "options": {
    "Correct": "...",
    "Trap_Preference": "...",
    "Trap_Fabrication": "...",
    "Trap_Generic": "..."
  },
  "explanation": "Brief analysis of why S is blocked by L."
}
"""



client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

max_api_retries = 3

def call_llm_json(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result_json = json.loads(content)
        if isinstance(result_json, list):
            if len(result_json) > 0 and isinstance(result_json[0], dict):
                result_json = result_json[0]
            else:
                return None
                
        return result_json if isinstance(result_json, dict) else None
    except Exception as e:
        error_str = str(e)
        print(f"[LLM Error : {error_str}")
        return None  
    
    

def calc_similarity(s1, s2):
    return round(difflib.SequenceMatcher(None, str(s1), str(s2)).ratio(), 3)

def calc_options_bias_metrics(options: dict) -> str:
    if not options: return "N/A"
    lengths = {k: len(str(v)) for k, v in options.items()}
    vals = list(lengths.values())
    if len(vals) < 4: return "Missing options."
    
    max_len, min_len = max(vals), min(vals)
    diff = max_len - min_len
    avg = sum(vals) / len(vals)
    
    metrics = f"Lengths: {lengths}. Max length diff: {diff} chars. Avg length: {avg:.1f}."
    return metrics


from datetime import datetime

def parse_time(time_str):
    try:
        return datetime.strptime(time_str, "%b %d, %Y, %H:%M:%S")
    except Exception:
        return datetime.min


def process_item(item_data):
    try:
        item = item_data
        if isinstance(item, list) and len(item) > 0:
            item = item[0]
        if not isinstance(item, dict): return None

        Memory_S = item.get("Memory_S", "")
        d_time = item.get("d_time")
        Memory_L = item.get("Memory_L", "")
        t_time = item.get("t_time")
        reason = item.get("reason", "")
        dt_d = parse_time(d_time)
        dt_t = parse_time(t_time)
        query_time = d_time if dt_d >= dt_t else t_time
        if not query_time:
            print("not query time")

        # ==================== STEP 1: Generate ====================
        gen_user_content = f"""
            ### Input Data:
            Memory_S (Distractor): {Memory_S}
            Memory_L (True Constraint): {Memory_L}
            Query_Time: {query_time}

            ### Output Format (JSON Only)
            {{
                "type": "Recommendation/Advice/Conversation",
                "query": "Must with query time date, The generated natural, first-person query...",
                "options": {{
                    "Correct": "...",
                    "Trap_Preference": "...",
                    "Trap_Fabrication": "...",
                    "Trap_Generic": "..."
                    }},
                "explanation": "Brief analysis of why D is blocked by T."
            }}"""
        current_qa = call_llm_json(SYSTEM_PROMPT, gen_user_content)
        if not current_qa: return None

        curation_history = []
        final_pass = False

        # ==================== STEP 2: Loop (Judge & Curate) ====================
        for attempt in range(MAX_RETRIES + 1):
            query = current_qa.get("query", "")
            # print(query)
            options_metrics = calc_options_bias_metrics(current_qa.get("options", {}))
            sim_d = calc_similarity(query, Memory_S)
            sim_t = calc_similarity(query, Memory_L)
            
            # --- Judge ---
            judge_user_content = json.dumps({
                "Logic_Context": {"D": Memory_S, "D_Time": d_time, "T": Memory_L, "T_Time": t_time, "Reason": reason},
                "Generated_QA": current_qa,
                "Reference_Metrics": {
                    "Options_Bias": options_metrics,
                    "Sim_Query_D": sim_d,
                    "Sim_Query_T": sim_t
                }
            }, ensure_ascii=False, indent=2)

            judge_res = call_llm_json(JUDGER_PROMPT, judge_user_content)
            if not judge_res:
                judge_res = {"status": "NOT PASS", "error_category": "Judge Error", "reason": "Judge failed to respond."}

            status = judge_res.get("status", "NOT PASS")
            curation_history.append({
                "attempt": attempt,
                "qa_state": current_qa,
                "judge_feedback": judge_res
            })

            if status == "PASS" :
                final_pass = True
                break
                
            if attempt < MAX_RETRIES:
                # --- Curate ---
                curate_user_content = json.dumps({
                    "Logic": {"D": Memory_S, "T_Time": t_time, "Query_Time": query_time, "Reason": reason},
                    "Original_QA": current_qa,
                    "Error_Category": judge_res.get("error_category", ""),
                    "Curating_Suggests": judge_res.get("reason", "")
                }, ensure_ascii=False, indent=2)

                curated_qa = call_llm_json(CURATOR_PROMPT, curate_user_content)
                if curated_qa and "options" in curated_qa:
                    current_qa = curated_qa
                else:
                    break

        new_item = item.copy()
        new_item["query_type"] = current_qa.get("type", "")
        new_item["query"] = current_qa.get("query", "")
        new_item["options"] = current_qa.get("options", {})
        new_item["correct_answer"] = current_qa.get("options", {}).get("Correct", "")
        new_item["explanation"] = current_qa.get("explanation", "")
        new_item["is_pass"] = final_pass
        new_item["curation_history"] = curation_history
        
        return new_item
    except Exception as e:
        print(f"[Process Error]: {str(e)}")
        return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    print(f"正在读取: {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("数据为空，退出")
        return

    print(f"共 {len(data)} 条数据，开始并发处理 (Workers: {MAX_WORKERS})...")
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data}
        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Generating & Curating"):
            res = future.result()
            if res:
                results.append(res)

    print(f"正在保存结果到: {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    pass_count = sum(1 for r in results if r.get("is_pass"))
    print(f"完成！成功率: {pass_count}/{len(results)}")

if __name__ == "__main__":
    main()