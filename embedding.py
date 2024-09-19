import pandas as pd
from utils import text_preprocessing

def embed_dataset(vector_store, dataset_path) -> None:
    df = pd.read_csv(dataset_path)

    df = df.fillna(value='')
    feature_cols = ["ProductBrand", "Description", "Gender", "PrimaryColor"]
    values = df[feature_cols].values
    overall_infos = []
    for value in values:
        result = " ".join(value)
        overall_infos.append(result)
    df["overall_info"] = overall_infos
    # df["cleaned_info"] = text_preprocessing(df['overall_info'])
    df["cleaned_info"] = df['overall_info'].apply(lambda x: text_preprocessing(x))

    chunks = df["cleaned_info"].tolist()
    metadatas = []
    for i in range(len(chunks)):
        metadatas.append({'index': i})

    vector_store.add_texts(chunks, metadatas=metadatas)