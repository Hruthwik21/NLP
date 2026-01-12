import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
import zipfile
import io
import os
import random
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

FILE_URL = "https://mattmahoney.net/dc/text8.zip"
FILENAME = "text8"
BATCH_SIZE = 1024
EMBED_DIM = 100
WINDOW_SIZE = 5
MIN_FREQ = 5
NEG_SAMPLES = 5
EPOCHS = 3
LEARNING_RATE = 0.003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")

def load_data():
    if not os.path.exists(FILENAME):
        print(f"Downloading {FILENAME} dataset...")
        r = requests.get(FILE_URL)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall()
    
    with open(FILENAME, 'r') as f:
        text = f.read()
    return text.split()

print("Loading and preprocessing data...")
tokens = load_data()
print(f"Total tokens in raw text: {len(tokens)}")

word_counts = Counter(tokens)
sorted_vocab = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
vocab = {w: i for i, (w, c) in enumerate(sorted_vocab) if c >= MIN_FREQ}
idx_to_word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)
print(f"Vocabulary size (min_freq={MIN_FREQ}): {vocab_size}")

total_count = sum(word_counts.values())
freqs = {w: c/total_count for w, c in word_counts.items()}
subsampling_p = {w: 1 - np.sqrt(1e-5 / freqs[w]) for w in freqs}
train_tokens = [vocab[w] for w in tokens if w in vocab and random.random() > subsampling_p[w]]
print(f"Tokens after subsampling: {len(train_tokens)}")

word_freqs = np.array([word_counts[idx_to_word[i]] for i in range(vocab_size)])
word_freqs = word_freqs ** 0.75
word_freqs = word_freqs / np.sum(word_freqs)
NEG_DIST = torch.from_numpy(word_freqs).float()

class Word2VecDataset(Dataset):
    def __init__(self, token_indices, window_size):
        self.tokens = token_indices
        self.window_size = window_size
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        center = self.tokens[idx]
        dynamic_window = random.randint(1, self.window_size)
        
        start = max(0, idx - dynamic_window)
        end = min(len(self.tokens), idx + dynamic_window + 1)
        
        context_indices = self.tokens[start:idx] + self.tokens[idx+1:end]
        
        if len(context_indices) == 0:
            return torch.tensor(center), torch.tensor(center)
            
        target = random.choice(context_indices)
        return torch.tensor(center), torch.tensor(target)

dataset = Word2VecDataset(train_tokens, WINDOW_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, embed_dim, neg_dist):
        super(SkipGramNeg, self).__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)
        self.neg_dist = neg_dist
        
        initrange = 0.5 / embed_dim
        self.center_embed.weight.data.uniform_(-initrange, initrange)
        self.context_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, center_words, context_words):
        batch_size = center_words.shape[0]
        
        center_vecs = self.center_embed(center_words)
        context_vecs = self.context_embed(context_words)
        
        pos_score = torch.sum(center_vecs * context_vecs, dim=1)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)
        
        neg_samples = torch.multinomial(self.neg_dist, batch_size * NEG_SAMPLES, replacement=True)
        neg_samples = neg_samples.view(batch_size, NEG_SAMPLES).to(center_words.device)
        
        neg_vecs = self.context_embed(neg_samples)
        
        neg_score = torch.bmm(neg_vecs, center_vecs.unsqueeze(2)).squeeze()
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        
        return (pos_loss + neg_loss).mean()

model = SkipGramNeg(vocab_size, EMBED_DIM, NEG_DIST).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nStarting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for i, (center, context) in enumerate(dataloader):
        center, context = center.to(DEVICE), context.to(DEVICE)
        
        optimizer.zero_grad()
        loss = model(center, context)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i > 0 and i % 2000 == 0:
            print(f"Epoch {epoch+1}, Step {i}/{len(dataloader)}, Avg Loss: {total_loss/2000:.4f}")
            total_loss = 0

print("Training complete.")

my_vectors = model.center_embed.weight.data.cpu().numpy()

print("\nLoading Gensim GloVe model for comparison (this may download ~60MB)...")
try:
    gensim_model = api.load("glove-wiki-gigaword-50")
    gensim_available = True
except Exception as e:
    print(f"Could not load Gensim model: {e}")
    gensim_available = False

def get_my_similarity(w1, w2):
    if w1 not in vocab or w2 not in vocab:
        return 0.0
    v1 = my_vectors[vocab[w1]].reshape(1, -1)
    v2 = my_vectors[vocab[w2]].reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]

test_pairs = [("man", "woman"), ("king", "queen"), ("paris", "france"), ("apple", "food")]

print("\n--- 1. Cosine Similarity Comparison ---")
print(f"{'Pair':<20} | {'My SGNS Model':<15} | {'Gensim (GloVe)':<15}")
print("-" * 55)

for w1, w2 in test_pairs:
    my_sim = get_my_similarity(w1, w2)
    gen_sim = gensim_model.similarity(w1, w2) if gensim_available and w1 in gensim_model and w2 in gensim_model else 0.0
    print(f"{w1}-{w2:<14} | {my_sim:.4f}          | {gen_sim:.4f}")

def solve_analogy(w_a, w_b, w_c):
    if w_a not in vocab or w_b not in vocab or w_c not in vocab:
        return "N/A (Word not in vocab)"
    
    va = my_vectors[vocab[w_a]]
    vb = my_vectors[vocab[w_b]]
    vc = my_vectors[vocab[w_c]]
    target = vb - va + vc
    
    norms = np.linalg.norm(my_vectors, axis=1)
    normed_vecs = my_vectors / norms[:, np.newaxis]
    target_normed = target / np.linalg.norm(target)
    
    scores = np.dot(normed_vecs, target_normed)
    
    top_k = np.argsort(scores)[::-1][:5]
    
    for idx in top_k:
        word = idx_to_word[idx]
        if word not in [w_a, w_b, w_c]:
            return word
    return idx_to_word[top_k[0]]

print("\n--- 2. Word Analogy Task ---")
analogies = [("man", "king", "woman"), ("paris", "france", "rome"), ("walk", "walking", "swim")]

for a, b, c in analogies:
    res = solve_analogy(a, b, c)
    print(f"{a} : {b} :: {c} : {res}")

def detect_bias():
    if "he" not in vocab or "she" not in vocab:
        print("Gender words not in vocab, cannot perform bias check.")
        return

    gender_vec = my_vectors[vocab["he"]] - my_vectors[vocab["she"]]
    
    occupations = ["engineer", "nurse", "doctor", "teacher", "mechanic", "homemaker"]
    
    print("\n--- 3. Bias Detection (Projection onto 'he-she' axis) ---")
    print("Positive values -> closer to 'he', Negative values -> closer to 'she'")
    print(f"{'Occupation':<15} | {'Bias Score'}")
    print("-" * 30)
    
    for job in occupations:
        if job in vocab:
            vec = my_vectors[vocab[job]]
            score = np.dot(vec, gender_vec) / (np.linalg.norm(vec) * np.linalg.norm(gender_vec))
            print(f"{job:<15} | {score:.4f}")
        else:
            print(f"{job:<15} | Not in vocab")

detect_bias()