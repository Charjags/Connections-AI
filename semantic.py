import spacy
import numpy as np

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_md")

def cosine_similarity_matrix(vectors):
    """Compute a cosine similarity matrix for all vectors."""
    dot_products = np.dot(vectors, vectors.T)
    norms = np.linalg.norm(vectors, axis=1)
    similarity_matrix = dot_products / np.outer(norms, norms)
    return similarity_matrix

def calculate_average_similarity(words):
    """Calculate the average pairwise cosine similarity among the words."""
    # Get word vectors from spaCy
    vectors = np.array([nlp(word).vector for word in words])

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity_matrix(vectors)

    # Ignore self-similarity by setting the diagonal to NaN
    np.fill_diagonal(similarity_matrix, np.nan)

    # Calculate the average of the upper triangle of the similarity matrix
    avg_similarity = np.nanmean(similarity_matrix)
    return avg_similarity
