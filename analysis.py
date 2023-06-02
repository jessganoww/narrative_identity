import pandas as pd

from constants import hc1

def compute_jaccard_index(w1: list, w2: list) -> float:
    n = set(w1).intersection(w2)
    d = set(w1).union(w2)

    return len(n) / len(d)

def compare_scores(doc_df: pd.DataFrame, human_codes: list, topic_model, cv_model) -> list:
    scores = []

    for hc in human_codes:
        code_df = doc_df[doc_df["Codes"].str.contains(hc, na=False)]
        docs = code_df["Transcript"].to_list()

        code_dtm = cv_model.fit_transform(docs)
        vocab = cv_model.get_feature_names_out()
        counts = code_dtm.toarray().sum(axis=0)

        top_n_df = pd.DataFrame({"Vocab": vocab, "Counts": counts})
        top_n_df.sort_values(by="Counts", ascending=False, inplace=True)

        hc_vocab = top_n_df["Vocab"].head(10)
        hc_score = []

        for i in range(len(topic_model.get_topics())-1):
            topic_vocab = list(map(lambda x: x[0], topic_model.get_topic(i)))
            topic_score = compute_jaccard_index(hc_vocab, topic_vocab)
            hc_score.append(topic_score)

        scores.append(hc_score)

    return scores

