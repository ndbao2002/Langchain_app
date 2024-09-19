from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st
import pandas as pd

from embedding import embed_dataset
from utils import get_product_index, gerenate_answer

@st.cache_resource
def init():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        collection_name="myntra",
        embedding_function=embeddings,
        persist_directory="./dataset/chroma_langchain_db",  # Where to save data locally
    )

    csv_path = './dataset/myntra_products_catalog.csv'
    df = pd.read_csv(csv_path)
    if len(vector_store.get()['ids']) == 0:
        embed_dataset(vector_store, csv_path)
    return df, vector_store


if __name__ == "__main__":
    df, vector_store = init()

    num = st.slider("Number of product(s)", 1, 5, 3)

    user_ask = st.text_input(
        "Write down some preference about product 👇",
        placeholder="Some questions",
    )

    if user_ask:
        with st.chat_message("user"):
            st.write(user_ask)

        idxs = get_product_index(vector_store=vector_store, 
                                 text=user_ask,
                                 k=num)
        
        for idx in idxs:
            ai_message = gerenate_answer(profile=df.iloc[idx])
            with st.chat_message("assistant"):
                st.write(ai_message)
