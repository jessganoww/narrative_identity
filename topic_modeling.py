from constants import seed_weights

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

def create_topics(docs, args):
    # Create embeddings
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = embedding_model.encode(docs)

    # Dimensionality reduction
    umap_model = UMAP(n_neighbors=args.n_neighbors, n_components=2, metric="cosine", random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)

    # Document clustering
    hdbscan_model = HDBSCAN(min_cluster_size=args.min_cluster_size, metric="euclidean", prediction_data=True)

    # Clustered documents tokenization
    cv_model = CountVectorizer(stop_words="english")

    # Topic representation
    ctfidf_model = ClassTfidfTransformer()

    topic_model = BERTopic(umap_model=umap_model,
                           hdbscan_model=hdbscan_model,
                           vectorizer_model=cv_model,
                           ctfidf_model=ctfidf_model,
                           seed_topic_list=seed_weights,
                           calculate_probabilities=True)

    topic, probs = topic_model.fit_transform(docs)

    return topic_model, reduced_embeddings, cv_model
