import json, numpy as np, base64

with open("emb_data.json", "r") as f:
    data = json.load(f)

parties = dict([(k, np.frombuffer(base64.b64decode(v), dtype=np.float32)) for (k, v) in data["parties"].items()])
policies = dict([(k, np.frombuffer(base64.b64decode(v), dtype=np.float32)) for (k, v) in data["policies"].items()])

def closest(parties, user_emb):
    distances = {party: 1 - np.dot(user_emb, emb) for party, emb in parties.items()}
    return sorted(distances, key=distances.get)

user_emb = np.zeros_like(list(policies.values())[0])
n = 1

for policy in policies:
    user_emb += policies[policy] * float(input("-1 to 1 >"))
    
    n += 1
    print(closest(parties, user_emb / n))