import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import Pinecone

embeddings = OpenAIEmbeddings()
index_name = os.environ['PINECONE_INDEX_NAME']

vector_db = Pinecone(embedding=embeddings)

vector_db.delete(delete_all=True, index_name=index_name, namespace="test")
