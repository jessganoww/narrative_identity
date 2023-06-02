import numpy as np
import pandas as pd
import contractions
import re

# TODO modify, without hard coding
def preprocess_transcript(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    df.drop(df[df["Transcript"].str.startswith("Taylor: ")].index, inplace=True)

    df["Transcript"] = df["Transcript"].apply(lambda x: re.sub(r"Mrs\.?\s+R\s*:\s*", "", x))
    df["Transcript"] = df["Transcript"].apply(lambda x: contractions.fix(x))

    return df
