import os
import asyncio

from langchain_pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough, RunnableParallel

embeddings = OpenAIEmbeddings()
index_name = os.environ['PINECONE_INDEX_NAME']

vectorstore = Pinecone.from_existing_index(embedding=embeddings, index_name=index_name, namespace="ahmed")

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Multi model fallback
google_gemini = ChatGoogleGenerativeAI(model="gemini-pro")
chat_openai = ChatOpenAI(model="gpt-4-0125-preview")
openai_gpt_3 = ChatOpenAI(model="gpt-3.5-turbo-0125")
mistral = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")
model = (
    google_gemini
    .with_fallbacks([openai_gpt_3, chat_openai, mistral])
    .configurable_alternatives(
        ConfigurableField(id="model"),
        default_key="google_gemini",
        chat_openai=chat_openai,
        openai_gpt_3=openai_gpt_3,
        mistral=mistral

    )
)

output_parser = StrOutputParser()
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = (
        setup_and_retrieval
        | prompt
        | model
        | output_parser
)


async def main():
    async for chunk in chain.astream("tell me something about Ahmed"):
        print(chunk, end="", flush=True)


# Run the main coroutine
if __name__ == "__main__":
    asyncio.run(main())
