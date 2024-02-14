import os
import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import GigaChat
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import Pinecone

embeddings = OpenAIEmbeddings()
index_name = os.environ['PINECONE_INDEX_NAME']

vectorstore = Pinecone.from_existing_index(embedding=embeddings, index_name=index_name, namespace="ahmed")

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = GigaChat()
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser


# Sync
# invoke
# print(chain.invoke("where did ahmed works now?"))

# stream
# for chunk in chain.stream("where did ahmed works now?"):
#     print(chunk, end="", flush=True)

# batch
# print(chain.batch(["where did ahmed works now?", "Where Ahmed use to worked?", "you know where ahmed study?"]))

# Async
# Invoke
# async def main():
#     print(await chain.ainvoke("where did ahmed study now?"))


# Stream
async def main():
    async for chunk in chain.astream("where did ahmed works now?"):
        print(chunk, end="", flush=True)


# Batch async def main(): print(await chain.abatch(["where did ahmed works now?", "Where Ahmed use to worked?",
# "you know where ahmed study?"]))


# Run the main coroutine
if __name__ == "__main__":
    asyncio.run(main())
