import json
import numpy as np
#import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from openTSNE import TSNE
import base64

with open("result.json", "r") as f:
    data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
positive_embs = model.encode([i["positive"] for i in data.values()])
negative_embs = model.encode([i["negative"] for i in data.values()])
delta_embs = positive_embs - negative_embs

#delta_embs = model.encode([i["ideal_inspiration"] for i in data.values()])


def slerp(v0:np.ndarray, v1:np.ndarray, t:float):
    DOT_THRESHOLD = 0.9995

    dot = v0.dot(v1)
    if abs(dot) > DOT_THRESHOLD:
        v_out =  v0 * (1 - t) + v1 * t
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v_out =  v0 * s0.item() + v1 * s1.item()
    
    return v_out

def slerp_average(x:np.ndarray):
    slerped = [i for i in x]
    while not len(slerped) == 1:            
        slerped.append(slerp(slerped[0], slerped[1], 0.5))
        slerped = slerped[2:]
    return slerped[-1]


party_embs = {}

for idx, (k, v) in enumerate(data.items()):
    party_embs.setdefault(v["party_src"], [])
    party_embs[v["party_src"]].append(delta_embs[idx])

for party in party_embs:
    party_embs[party] = slerp_average(np.array(party_embs[party]))

party_labels = list(party_embs.keys())
party_embs = np.array(list(party_embs.values()))

pca = PCA(n_components=8)
pca.fit(delta_embs)
party_embs = pca.transform(party_embs)
delta_embs = pca.transform(delta_embs)

emb_data = {
    "parties": {},
    "policies": {}
}
for idx, party in enumerate(party_labels):
    emb_data["parties"][party] = base64.b64encode(party_embs[idx].astype(np.float32)).decode()

for idx, policy in enumerate(data):
    emb_data["policies"][data[policy]["ideal_inspiration"]] =  base64.b64encode(delta_embs[idx].astype(np.float32)).decode()


with open("emb_data.json", "w") as f:
    json.dump(emb_data, f, indent="\t")