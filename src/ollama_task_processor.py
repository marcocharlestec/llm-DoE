import json
import requests
import pandas as pd
import datetime

print("Starting Python Function")
print(datetime.datetime.now())

# Load JSON instructions
with open('Instructions.txt', 'r') as f:
    tasks = json.load(f)

print("Loaded instruction file successfully")
print(datetime.datetime.now())

# Ollama API settings
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'llama3:latest'  

print("Configured LLM access successfully")
print(datetime.datetime.now())

# Store results
results = []

for idx, task in enumerate(tasks):
    instruction = task.get('instruction', '')
    input_text = task.get('input', '')
    correct_response = task.get('correctResponse', '')
    prompt = f"{instruction}\n\n{input_text}\n\n{correct_response}"

    print(f"Sending task {idx + 1} to Ollama...")

    response = requests.post(OLLAMA_URL, json={{
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }})

    if response.status_code == 200:
        result_text = response.json()["response"]
        print(f"Task {idx + 1} completed.")

        results.append({{
            'Question #': idx + 1,
            'Instruction': instruction,
            'Input': input_text,
            'Prompt': prompt,
            'Response': result_text
        }})
    else:
        print(f"Error with task {idx + 1}: {response.text}")
        results.append({{
            'Question #': idx + 1,
            'Instruction': instruction,
            'Input': input_text,
            'Prompt': prompt,
            'Response': f"ERROR: {response.text}"
        }})

print("All instructions were submitted successfully and responses file is being updated")
print(datetime.datetime.now())

# Convert to DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv('responses_test1.csv', index=False)
print("Responses saved to responses.csv")
