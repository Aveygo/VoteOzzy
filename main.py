import json, numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from collections import deque
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.spatial import geometric_slerp


def slerp(v1, v2, t, DOT_THR=0.9995, zdim=-1):

    v1_norm = v1 / np.sqrt(np.sum(v1**2))
    v2_norm = v2 / np.sqrt(np.sum(v2**2))
    dot = (v1_norm * v2_norm).sum(zdim)

    if (np.abs(dot) > DOT_THR).any():
        return (1 - t) * v1 + t * v2    

    theta = np.acos(dot)
    theta_t = theta * t
    sin_theta = np.sin(theta)
    sin_theta_t = np.sin(theta_t)

    s1 = np.sin(theta - theta_t) / sin_theta
    s2 = sin_theta_t / sin_theta

    return (s1 * v1) + (s2 * v2)


class Master:
    def __init__(self):
        with open("dataset/result_pn.json", "r") as f:
            self.data = json.load(f)
        
        self.politician_embeddings = {}
        
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sort_policies()
        self.calculate_stance_embeddings()
        self.calculate_politician_embeddings()
        self.calculate_party_embeddings()

    def calculate_party_embeddings(self):
        self.party_embeddings = {}
        parties = list(set([i["party"] for i in self.data["politicians"].values()]))

        for party in tqdm(parties, "Calculating embeddings for each party..."):
            party_members = [k for k, v in self.data["politicians"].items() if v["party"] == party]
            party_embeddings = np.array([self.politician_embeddings[member]["mean"] for member in party_members])

            self.party_embeddings[party] = {
                "mean": np.mean(party_embeddings, axis=0).squeeze(),
                "std": np.std(party_embeddings, axis=0).squeeze()
            }

    def calculate_politician_embeddings(self):
        for politician in tqdm(self.data["politicians"], "Calculating embeddings for each politician..."):
            mean, std = self.calculate_politician_embedding(self.data["politicians"][politician]["stances"])
            self.politician_embeddings[politician] = {
                "mean": mean,
                "std": std
            }

    def calculate_stance_embeddings(self):
        for key, value in tqdm(self.data["policies"].items(), "Calculating positive/negative embeddings..."):
            positive = self.model.encode(value["positive"])
            negative = self.model.encode(value["negative"])

            assert not positive.shape == (), f"{key} failed shape"
            assert not negative.shape == (), f"{key} failed shape"

            self.data["policies"][key]["positive_emb"] = positive / np.sqrt(np.sum(positive**2))
            self.data["policies"][key]["negative_emb"] = negative / np.sqrt(np.sum(negative**2))

    def calculate_politician_embedding(self, stances: dict[str, list]):


        weights = {
            "Voted consistently for": 1,
            "Voted almost always for": 0.75,
            "Voted generally for": 0.5,
            "Voted a mixture of for and against": 0,
            "Voted generally against": -0.5,
            "Voted almost always against": -0.75,
            "Voted consistently against": -1,
            "We can't say anything concrete about how they voted on": 0
        }

        politician_embeddings = []
        
        for stance in stances:

            weight = weights[stance]
            for policy in stances[stance]:
                positive = self.data["policies"][policy]["positive_emb"]
                negative = self.data["policies"][policy]["negative_emb"]

                politician_embeddings.append((positive-negative)*weight)
        
        politician_embeddings = np.array(politician_embeddings)
        return np.mean(politician_embeddings, axis=0).squeeze(), np.std(politician_embeddings, axis=0).squeeze()

    def sort_policies(self):
        """
        # Sorting by policy clusters
        policies = list(self.data["policies"].keys())
        linkage_data = linkage(self.model.encode(policies), method='ward', metric='euclidean')
        tree = self.build_tree(linkage_data, num_items=len(policies))
        sorted_indices = self.bft(tree, num_items=len(policies))
        self.sorted_policies = [policies[i] for i in sorted_indices]
        """

        weights = {
            "Voted consistently for": 1,
            "Voted almost always for": 0.75,
            "Voted generally for": 0.5,
            "Voted a mixture of for and against": 0,
            "Voted generally against": -0.5,
            "Voted almost always against": -0.75,
            "Voted consistently against": -1,
            "We can't say anything concrete about how they voted on": 0
        }

        # Sorting by 'closest to 50/50'
        scores = {}
        for target_policy in self.data["policies"]:
            yays = 1
            nays = 1
            for politician in self.data["politicians"]:
                for stance in self.data["politicians"][politician]["stances"]:
                    for policy in self.data["politicians"][politician]["stances"][stance]:
                        if policy == target_policy:
                            if weights[stance] > 0:
                                yays += 1
                            else:
                                nays += 1
            
            distance_to_one = abs(1 - yays/nays)
            scores[target_policy] = distance_to_one
        
        scores = dict(sorted(scores.items(), key=lambda item: item[1]))
        print(scores)
        self.sorted_policies = list(scores.keys())

    def build_tree(self, linkage_data, num_items):
        tree = {}
        for idx, (left, right, distance, _) in enumerate(linkage_data):
            cluster_id = num_items + idx
            tree[cluster_id] = (int(left), int(right), distance)
        return tree
    
    def bft(self, tree, num_items):
        queue = deque()
        result = []

        root = num_items + len(tree) - 1
        queue.append(root)

        while queue:
            node = queue.popleft()
            if node < num_items:
                result.append(node)
            else:
                left, right, distance = tree[node]
                queue.append(left)
                queue.append(right)

        return result
        
class User:
    def __init__(self, master:Master):
        self.embeddings = []
        self.master = master
        self.progress = 0

    def get_current_question(self) -> tuple[str, str]:
        policy = self.master.sorted_policies[self.progress]
        description = self.master.data["policies"][policy]["description"]
        return policy, description
    
    def mahalanobis_distance(self, avg1, std1, avg2, std2):
        cov1 = np.diag(std1**2)
        cov2 = np.diag(std2**2)
        cov = (cov1 + cov2) / 2

        diff = avg1 - avg2
        cov_inv = np.linalg.pinv(cov)

        dist = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))
        return dist
    

    def cosine_distance(self, vec1, std1, vec2, std2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        cosine_similarity = dot_product / (norm1 * norm2)

        return 1 - cosine_similarity
    
    def euclidean_distance(self, vec1, std1, vec2, std2):
        return np.linalg.norm(vec1 - vec2)

    def answer_current_question(self, response:int):
        assert response >= 0 and response <= 4, "user must respond with preference from 0 to 5!" 
        
        weight = [-1, -0.5, 0.0, 0.5, 1.0][response]
        
        policy = self.master.sorted_policies[self.progress]

        positive = self.master.data["policies"][policy]["positive_emb"]
        negative = self.master.data["policies"][policy]["negative_emb"]
        self.embeddings.append((positive-negative) * weight)

        self.progress += 1

    def get_top_politicians(self, n=5):
        scores = {}
        our_embeddings = np.array(self.embeddings) 
        mean1, std1 = np.mean(our_embeddings, axis=0).squeeze(), np.std(our_embeddings, axis=0).squeeze()
        
        for politician, v in self.master.politician_embeddings.items():
            mean2, std2 = v["mean"], v["std"]
            scores[politician] = self.cosine_distance(mean1, std1, mean2, std2)
        
        x = [v["mean"] for k, v in self.master.politician_embeddings.items()]
        pca = PCA(n_components=2)
        pca.fit(x)

        x = pca.transform(x)
        user = pca.transform([mean1])
        plt.cla()
        plt.clf()
        plt.scatter([i[0] for i in x], [i[1] for i in x])
        plt.scatter([user[0][0]], [user[0][1]])
        plt.savefig("plt.png")

        scores = dict(sorted(scores.items(), key=lambda item: item[1]))
        return list(scores.keys())[:n]

    def get_top_parties(self, n=5):
        scores = {}
        our_embeddings = np.array(self.embeddings) 
        mean1, std1 = np.mean(our_embeddings, axis=0).squeeze(), np.std(our_embeddings, axis=0).squeeze()
        for party, v in self.master.party_embeddings.items():
            if "party" in party.lower():
                mean2, std2 = v["mean"], v["std"]
                party = party.split(" Representative")[0].split(" Senator")[0]
                scores[party] = self.cosine_distance(mean1, std1, mean2, std2).mean()
        
        #print(scores)

        scores = dict(sorted(scores.items(), key=lambda item: item[1]))
        return list(scores.keys())[:n]


if __name__ == "__main__":
    
    us = User(Master())

    while True:
        print(us.get_current_question())
        us.answer_current_question(int(input("> ")) - 1)

        if len(us.embeddings) > 5:
            print(us.get_top_politicians(5))
            print(us.get_top_parties(5))

    




