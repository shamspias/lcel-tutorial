import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import Pinecone

embeddings = OpenAIEmbeddings()
index_name = os.environ['PINECONE_INDEX_NAME']

loader = PyPDFLoader("data/ahmed.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
vectorstore = Pinecone.from_documents(docs, embeddings, index_name=index_name, namespace="test")
