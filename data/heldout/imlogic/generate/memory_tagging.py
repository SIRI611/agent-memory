import json
import os
import requests
import re  
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


INPUT_FILE = ""
OUTPUT_FILE = ""


LLM_API_KEY = ""
LLM_BASE_URL = ""
LLM_MODEL_NAME = "gemini-3-pro-preview"
MAX_WORKERS = 30

SYSTEM_PROMPT = """
# Role
You are a memory classification expert. Your task is to label user inputs with the single most accurate **Top-Level Tag**.

# Constraints
1. **Analyze First**: You must briefly reason step-by-step to distinguish between categories (e.g., Is this a Habit or a Preference? Is this a Past Event or Background Info?).
2. **Output Format**: After your analysis, strictly output the final label on a new line: `Tag: [Tag_Name]`.
3. **Strict Taxonomy**: Use ONLY the 10 tags listed below.
4. **Single Tag**: If multiple tags apply, choose the most specific or dominant one.
5. **Language**: The input may be in various languages, but the Output Tag Name must be **English**.

# Taxonomy

1. **Personal_Background**
   - **Definition**: Static info about who the user is.
   - **Includes**: Identity (Name, Age, Gender), **Personality (MBTI, Traits, Character)**, Education, **Occupation (Job Title, Professional Skills, Work Environment, Tools used)**, Location, Hometown.

2. **Assets**
   - **Definition**: Material wealth and ownership.
   - **Includes**: **Financial Status (Money, Savings, Investments)**, Physical Possessions (House, Car, Phone), Pets, Important Items.

3. **Past_Experience**
   - **Definition**: Episodic memories of specific past events.
   - **Includes**: Life stories, Anecdotes, Specific activities participated in (e.g., trips, hikes), Historical events.

4. **States**
   - **Definition**: Temporary conditions.
   - **Includes**: Physical state (Health, Fatigue, Hunger) and Mental state (Mood, Emotions, Stress).

5. **Preferences**
   - **Definition**: Subjective tastes and likes/dislikes.
   - **Includes**: Food, Entertainment, Sports, Reading, Music, Travel styles, Shopping habits, Interaction preferences.

6. **Opinions**
   - **Definition**: Abstract thoughts and attitudes.
   - **Includes**: Subjective views, Positive/Negative attitudes towards topics, Beliefs, Values, Curiosity.

7. **Goals**
   - **Definition**: Future aspirations.
   - **Includes**: Short-term objectives, Long-term dreams/missions.

8. **Plans**
   - **Definition**: Concrete future arrangements.
   - **Includes**: Specific schedules, Future commitments, Appointments, To-do lists.

9. **Social_Relationships**
   - **Definition**: The user's social network.
   - **Includes**: Specific mentions of Family, Friends, Colleagues, Partners, or Adversaries.

10. **Others**
    - **Definition**: Unclassifiable information.

# Examples

Input: "I am an ENFP and I love brainstorming."
Output: 
Analysis: The user mentions their MBTI type (Identity).
Tag: Personal_Background

Input: "Martin Mark uses Python and collaborative tools to solve problems."
Output:
Analysis: This describes professional skills and work methods.
Tag: Personal_Background

Input: "I joined a conservation group during a nature hike last year."
Output: 
Analysis: This is a specific past event/story.
Tag: Past_Experience

Input: "I plan to visit Japan next month."
Output: 
Analysis: This is a concrete future arrangement.
Tag: Plans
"""

def call_llm_for_tag(memory_content):
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Input: \"{memory_content}\"\nOutput:"}
        ],
        "temperature": 0.0, 
        "max_tokens": 500  
    }

    try:
        response = requests.post(LLM_BASE_URL, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        match = re.search(r"Tag:\s*([A-Za-z_]+)", content, re.IGNORECASE)
        
        if match:

            return match.group(1).strip()
        else:
            lines = content.split('\n')
            for line in reversed(lines):
                if "Tag:" in line:
                    return line.split("Tag:")[1].strip()
            return "Error"
            
    except Exception as e:
        print(f"Error processing '{memory_content[:20]}...': {e}")
        return "Error"

def process_memory_item(item_info):
    m_idx, content = item_info
    tag = call_llm_for_tag(content)
    return (m_idx, tag)


def main():
    print(f"Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Error: File not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = []
    print("Preparing tasks...")

    for m_idx, memory in enumerate(data):
        content = memory.get("memory_content", "")
        if content:
            tasks.append((m_idx, content))

    total_tasks = len(tasks)
    print(f"Total memories to tag: {total_tasks}")

    print(f"Starting processing with {MAX_WORKERS} workers...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_memory_item, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=total_tasks, desc="Tagging"):
            try:
                m_idx, tag = future.result()
                data[m_idx]["tag"] = tag
            except Exception as e:
                print(f"Worker exception: {e}")

    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Done!")

if __name__ == "__main__":
    main()