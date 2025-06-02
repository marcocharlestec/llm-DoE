import json
import pandas as pd
import datetime
import ssl
import certifi
import httpx
import random
from openai import OpenAI

ssl_context = ssl.create_default_context(cafile=certifi.where())
http_client = httpx.Client(verify=ssl_context)

client = OpenAI(
    api_key="",  
    http_client=http_client
)

print("Starting ChatGPT Query Script")
print(datetime.datetime.now())

with open('instructions.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

random.shuffle(tasks)  
print("Instructions loaded and randomized")
print(datetime.datetime.now())

results = []

for idx, task in enumerate(tasks):
    prompt = task.get("prompt", "")
    print(f"Sending task {idx + 1} to ChatGPT...")

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # You can change this to "gpt-4" if needed
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        result_text = response.choices[0].message.content.strip()
        print(f"Task {idx + 1} completed.")

    except Exception as e:
        print(f"Error with task {idx + 1}: {e}")
        result_text = f"ERROR: {e}"

    results.append({
        "Question #": idx + 1,
        "Prompt": prompt,
        "Resolution": task.get("resolution"),
        "WordLengthPattern": str(task.get("WordLengthPattern")),
        "ClearInteractions": task.get("clearIteractions"),
        "Response": result_text
    })

df = pd.DataFrame(results)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"responses_chatgpt_{timestamp}.csv"
df.to_csv(output_file, index=False)

print(f" Responses saved to {output_file}")
