import requests
import re
import random
from collections import defaultdict, Counter

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""

def clean_and_tokenize(text):
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return tokens

def train_ngram_models(tokens, max_n=5):
    models = {}
    for n in range(1, max_n + 1):
        print(f"Training {n}-gram model...")
        model = defaultdict(Counter)
        for i in range(len(tokens) - n):
            history = tuple(tokens[i : i + n - 1])
            next_word = tokens[i + n - 1]
            model[history][next_word] += 1
        models[n] = model
    return models

def predict_next_word(input_text, models, max_n=5):
    tokens = re.findall(r'\b[a-zA-Z]+\b', input_text.lower())
    for n in range(max_n, 1, -1):
        req_history_len = n - 1
        if len(tokens) >= req_history_len:
            history = tuple(tokens[-req_history_len:])
            if history in models[n]:
                best_word = models[n][history].most_common(1)[0][0]
                return best_word, n 
    if () in models[1]:
        return models[1][()].most_common(1)[0][0], 1
    return "the", 0

def run_prediction_demo():
    urls = [
        "https://www.gutenberg.org/files/1661/1661-0.txt", 
        "https://www.gutenberg.org/files/2852/2852-0.txt",
        "https://www.gutenberg.org/files/244/244-0.txt" 
    ]

    full_text = ""
    print("Downloading books from Project Gutenberg...")
    for url in urls:
        full_text += fetch_text_from_url(url) + " "
    
    tokens = clean_and_tokenize(full_text)
    print(f"Total words processed: {len(tokens)}")
    
    models = train_ngram_models(tokens, max_n=5)
    
    samples = [
        "The adventure of the",
        "It was a dark",
        "Sherlock Holmes sat in", 
        "elementary my dear watson",
    ]
    
    print("\n" + "="*50)
    print(f"PREDICTIONS (Author Style: Arthur Conan Doyle)")
    print("="*50)
    
    for input_text in samples:
        prediction, n_used = predict_next_word(input_text, models, max_n=5)
        output_text = f"{input_text} {prediction}"
        print(f"Input: \"{input_text}\"")
        print(f"Output: \"{output_text}\" (Used {n_used}-gram)\n")

if __name__ == "__main__":
    run_prediction_demo()