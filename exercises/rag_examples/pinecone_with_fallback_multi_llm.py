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


class LangChainService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.index_name = os.environ.get('PINECONE_INDEX_NAME',
                                         'your_default_index_name_here')  # Provide a default value for safety
        self.vectorstore = Pinecone.from_existing_index(
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace="ahmed"
        )
        self.retriever = self.vectorstore.as_retriever()
        self.template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.model = self.setup_model()
        self.output_parser = StrOutputParser()
        self.setup_and_retrieval = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        )
        self.chain = (
                self.setup_and_retrieval
                | self.prompt
                | self.model
                | self.output_parser
        )

    def setup_model(self):
        google_gemini = ChatGoogleGenerativeAI(model="gemini-ulti")
        chat_openai = ChatOpenAI(model="gpt-4-0125-preview")
        openai_gpt_3 = ChatOpenAI(model="gpt-3.5-turbo-0125")
        mistral = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")
        model = (
            openai_gpt_3
            .with_fallbacks([google_gemini, chat_openai, mistral])
            .configurable_alternatives(
                ConfigurableField(id="model"),
                default_key="openai_gpt_3",
                chat_openai=chat_openai,
                google_gemini=google_gemini,
                mistral=mistral
            )
        )
        return model

    async def get_response(self, question):
        async for chunk in self.chain.astream(question):
            yield chunk


async def main():
    lang_chain_service = LangChainService()
    async for response in lang_chain_service.get_response("tell me about Ahmed"):
        print(response, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
