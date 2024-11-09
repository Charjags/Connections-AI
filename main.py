import json
import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import AgglomerativeClustering
from flask import Flask, request, jsonify

# Load pre-trained Google News Word2Vec embeddings
model_path = 'Connections-AI/GoogleNews-vectors-negative300.bin'  # Update with the actual path to your file
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Load the dataset
with open('Connections-AI/connections.json') as f:
    data = json.load(f)

# Extract groups and words for validation
validation_data = {}
for entry in data:
    for answer in entry['answers']:
        group = answer['group']
        words = answer['members']
        validation_data[group] = words

# Function to get embeddings for words
def get_embedding(word):
    try:
        return model[word]
    except KeyError:
        # Handle out-of-vocabulary words by returning zeros
        return np.zeros(model.vector_size)

# Calculate similarity matrix for a list of words
def compute_similarity_matrix(words):
    embeddings = np.array([get_embedding(word) for word in words])
    similarity_matrix = np.inner(embeddings, embeddings)  # Cosine similarity
    return similarity_matrix

# Validate each group in the dataset
similarity_scores = {}
for group, words in validation_data.items():
    similarity_scores[group] = compute_similarity_matrix(words)

# Clustering function
def cluster_words(similarity_matrix):
    clustering_model = AgglomerativeClustering(
        n_clusters=4, metric='precomputed', linkage='average'
    )
    labels = clustering_model.fit_predict(1 - similarity_matrix)
    return labels

# Validate clustering against dataset
group_clusters = {}
for group, words in validation_data.items():
    sim_matrix = compute_similarity_matrix(words)
    labels = cluster_words(sim_matrix)
    group_clusters[group] = dict(zip(words, labels))

# Formulate a guess based on clusters and previous guesses
def formulate_guess(words, labels, previous_guesses):
    for cluster_id in set(labels):
        group_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        group_words = [words[i] for i in group_indices]
        if group_words not in previous_guesses:
            return group_words
    return None

# Main game function
def play_game(input_data):
    words = input_data['words']
    strikes = input_data['strikes']
    previous_guesses = input_data['previousGuesses']

    similarity_matrix = compute_similarity_matrix(words)
    labels = cluster_words(similarity_matrix)
    guess = formulate_guess(words, labels, previous_guesses)

    if guess is None or strikes >= 4:
        return {'guess': [], 'endTurn': True}
    else:
        return {'guess': guess, 'endTurn': False}

# Set up Flask application
app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_request():
    input_data = request.get_json()
    response = play_game(input_data)
    return jsonify(response)

if __name__ == '__main__':
    app.run()
