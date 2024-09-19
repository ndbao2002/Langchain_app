from sentence_transformers import util
from nltk.corpus import stopwords
import pandas as pd

from langchain_cohere import ChatCohere
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
stop = stopwords.words('english')


def text_preprocessing(text) -> str:
    text = text.lower()
    word_tokens = text.split(' ')
    keywords = [item for item in word_tokens if item not in stop]

    return ' '.join(keywords)


def get_product_index(vector_store, text, k=3) -> list:
    list_query = vector_store.similarity_search(query=text_preprocessing(text), k=k)

    idxs = []
    for query in list_query:
        idxs.append(query.metadata['index'])

    return idxs

def gerenate_answer(profile) -> str:
    chat = ChatCohere(temperature=1)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that can write introduction about 
        some product based on product's information.
        """


    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = """
        Write a brief introduction about product
        based on product information which structured below:

        ### START OF PRODUCT INFORMATION ###
        {text}
        ### END OF PRODUCT INFORMATION ###

        Introduction should be as short as possible.
        Product's price must be included in introduction.

        """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = chat_prompt | chat

    text = f"""
    Product ID: {profile['ProductID']}.
    Product Name: {profile['ProductName']}.
    Product Brand: {profile['ProductBrand']}.
    Gender: {profile['Gender']}.
    Price: {profile['Price (INR)']} INR.
    PrimaryColor: {profile['PrimaryColor']}.
    Description: {profile['Description']}.
    """

    response = chain.invoke(input={'text': text})
    if response.content[0] == '"':
        return response.content[1:-1]
    else:
        return response.content