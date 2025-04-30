import json
from ollama import chat
from ollama import ChatResponse


with open("result_pn.json", "r") as f:
    data = json.load(f)


response_format = """{"agree":<agree>, "disagree":<disagree>}""" 

def format_policy(policy, description):
    response: ChatResponse = chat(model='qwen2.5:32b-instruct-q4_K_M', messages=[
    {
        'role': 'user',
        'content': f'I have the following policy that I want to turn into an embedding:\n```{policy}\n{description}```\n\nCan you create two perspectives from the policy; one sentence that agrees with the policy, and another that disagrees. Please answer your response using the json:\n``````',
    },
    ])
    response = response['message']['content']
    response = response.replace("```", "").replace("\n", "").replace("  ", " ").replace("\t", "").replace("json", "")
    
    print(response)

    try:
        response = json.loads(response)
        agree, disagree = response.values()
    except Exception as e:
        print(e)
        return format_policy(policy, description)

    return agree, disagree


for (policy, description) in list(data["policies"].items()):

    if isinstance(description, dict):
        continue

    positive, negative = format_policy(policy, description)

    data["policies"][policy] = {
        "description": description,
        "positive": positive,
        "negative": negative
    }

    with open("result_pn.json", "w") as f:
        json.dump(data, f, indent="\t")
    
