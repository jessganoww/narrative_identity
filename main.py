import os
import argparse

from constants import hc2 as hc
from utils import create_dir, save_args
from preprocessing import preprocess_transcript
from topic_modeling import create_topics
from visualization import generate_barchart, generate_topic_heatmap
from analysis import compare_scores

parser = argparse.ArgumentParser()

parser.add_argument("base_dir", help="", type=str)
parser.add_argument("exp_dir", help="", type=str)
parser.add_argument("--n_neighbors", help="UMAP", default=3, type=int)
parser.add_argument("--min_cluster_size", help="", default=2, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    EXP_PATH = create_dir(args.base_dir, args.exp_dir)

    save_args(args, EXP_PATH)

    # disable parallelism to avoid deadlock
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    doc_df = preprocess_transcript("transcripts/mrs-r.csv")
    docs = doc_df['Transcript'].to_list()

    print("Creating topic models...")
    topic_model, reduced_embeddings, cv_model = create_topics(docs, args)
    generate_barchart(topic_model, os.path.join(EXP_PATH, "barchart.png"))

    print("Computing similarity score...")
    scores = compare_scores(doc_df, hc, topic_model, cv_model)
    generate_topic_heatmap(scores, hc, os.path.join(EXP_PATH, "heatmap.png"))

