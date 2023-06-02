import matplotlib.pyplot as plt
import seaborn as sns

def generate_barchart(topic_model, file_name: str) -> None:
    fig = topic_model.visualize_barchart(n_words=10, height=300, width=325)
    fig.write_image(file_name)

def generate_topic_heatmap(scores: list, human_codes: list, file_name: str) -> None:
    sns.set(font_scale=0.8)

    hm = sns.heatmap(scores, annot=True, yticklabels=human_codes,
                     cmap=sns.color_palette("flare", as_cmap=True))
    plt.title(f"Annotation-Topic Similarity Score (n={len(scores[0])})")
    fig = hm.get_figure()
    fig.savefig(file_name, bbox_inches="tight")


