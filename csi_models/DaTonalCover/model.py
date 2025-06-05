import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import pandas as pd
import random
from itertools import combinations



class DaTonalCoverNN(nn.Module):
    def __init__(self):
        super(DaTonalCoverNN, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

class DaTonalCover:
    def __init__(self, instrumental_threshold=8):
        self.instrumental_threshold = instrumental_threshold
        self.nn_model = DaTonalCoverNN()
        self.is_model_loaded = False

    def compute_tonal_similarity(self, hpcp_a, hpcp_b):
    
        mean_a = np.mean(hpcp_a, axis=0)
        mean_b = np.mean(hpcp_b, axis=0)

        # calculate cosine similarity
        similarity = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b))
        return similarity

    def load_hpcp_feature(self,file_path):
        with h5py.File(file_path, 'r') as f:
            return np.array(f['hpcp'])

    def extract_similarity_features(self, pair_list, feature_base_path):
        """
        pair_list: List of tuples [(pid1, pid2, label), ...]
        feature_base_path: path to folder with W_*/P_*.h5
        """
        features, labels = [], []
        import time

        start = time.time()



        for pid1, pid2, label in pair_list:
            h1 = self.find_h5_file(pid1, feature_base_path)
            h2 = self.find_h5_file(pid2, feature_base_path)
            if not h1 or not h2:
                continue  # skip if file not found

            f1 = self.load_hpcp_feature(h1)
            f2 = self.load_hpcp_feature(h2)
            sim = self.compute_tonal_similarity(f1, f2)
            #print([sim])
            features.append([sim])  # must be 2D: [[0.87]]
            labels.append(label)
        end = time.time()
        print(end - start)

        return np.array(features), np.array(labels)

    def find_h5_file(self,pid, base_path):
        for root, _, files in os.walk(base_path):
            for f in files:
                if f.startswith(pid) and f.endswith(".h5"):
                    return os.path.join(root, f)
        return None

    def train_tonal_model(self, X_train, y_train,epoch_number=10,learning_rate=0.001, model_path="DaTonalCover.pth"):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_model = self.nn_model.to(device)
    

        optimizer = optim.Adam(self.nn_model.parameters(),lr=learning_rate)
        criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epoch_number):
            self.nn_model.train()
            running_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = self.nn_model(x_batch).squeeze()
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"[{epoch+1}] Loss: {running_loss/len(loader):.4f}")

        torch.save(self.nn_model.state_dict(), model_path)
        return self.nn_model

    def generate_pairs_from_csv(self, csv_path, pairs_limit=1000, num_negative_pairs_per_clique=1):
        df = pd.read_csv(csv_path)
        grouped = df.groupby("clique")

        positive_pairs = []
        negative_pairs = []

        # Generate all positive pairs
        for clique_id, group in grouped:
            pids = group['id'].tolist()
            for pid1, pid2 in combinations(pids, 2):
                positive_pairs.append((pid1, pid2, 1))
        ## 33 slajdy,26 u 2 osobt
        # Generate all negative pairs
        all_cliques = list(grouped.groups.keys())
        for clique_id, group in grouped:
            this_pids = group['id'].tolist()
            other_cliques = [cid for cid in all_cliques if cid != clique_id]
            for pid in this_pids:
                for _ in range(num_negative_pairs_per_clique):
                    rand_clique = random.choice(other_cliques)
                    rand_pid = df[df['clique'] == rand_clique].sample(1)['id'].values[0]
                    negative_pairs.append((pid, rand_pid, 0))

        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        if(pairs_limit==0):
            return all_pairs
        else:
            return all_pairs[:pairs_limit]

    