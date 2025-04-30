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


party_embs = {}

for idx, (k, v) in enumerate(data.items()):
    party_embs.setdefault(v["party_src"], [])
    party_embs[v["party_src"]].append(delta_embs[idx])

for party in party_embs:
    party_embs[party] = np.mean(np.array(party_embs[party]), axis=0)

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