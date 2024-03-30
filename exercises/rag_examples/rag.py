import os
import asyncio
from operator import itemgetter
from langchain.schema import format_document
from langchain_pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.messages import get_buffer_string
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory


class LangChainService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.index_name = os.environ['PINECONE_INDEX_NAME']
        self.vectorstore = Pinecone.from_existing_index(
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace="legaldata"
        )
        self.retriever = self.vectorstore.as_retriever()

        self.message_history = RedisChatMessageHistory(url="redis://localhost:6379/2", ttl=600,
                                                       session_id="123")

        self.memory = ConversationBufferWindowMemory(
            return_messages=True,
            output_key="answer",
            input_key="question",
            chat_memory=self.message_history,
            max_token_limit=5000
        )
        self.loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )

        self.default_document_prompt = PromptTemplate.from_template(template="{page_content}")

        self._template = """Given the following conversation and a follow up question, check Follow Up Input and chat 
        history if chat history is empty only check Follow Up Input is related to a tech or help types of 
        question that can solve by legal answer or not if not then at beginning of the standalone question write 
        "None" and then write the given question else rephrase the follow up question to be a standalone 
        question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

        self.condense_question_prompt = PromptTemplate.from_template(self._template)

        self.template = os.environ['LLM_INSTRUCTION_PROMPT'] + """\nAnswer the question based only on the following 
        context: {context}

Question: {question}
"""
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.model = self.setup_model()
        self.input_model = self.setup_model(optional=True)
        self.output_parser = StrOutputParser()

        self.setup_and_retrieval = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        )

        self.retrieved_documents = self.get_retrieved_documents()

        self.standalone_question = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            ) | self.condense_question_prompt | self.input_model | self.output_parser,
        )

        self.final_chain = (
                self.loaded_memory
                | self.standalone_question
                | self.retrieved_documents
                | self.prompt
                | self.model
        )

        self.chain = (
                self.setup_and_retrieval
                | self.prompt
                | self.model
                | self.output_parser
        )

    def _combine_documents(self, docs, document_separator="\n\n"):
        doc_strings = [format_document(doc, self.default_document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    def get_retrieved_documents(self):
        return {
            "context": lambda x: "" if "none" in x["standalone_question"][:6].lower() else itemgetter(
                "standalone_question") | self.retriever | self._combine_documents,
            "question": lambda x: x["standalone_question"] if not (
                    "none" in x["standalone_question"][:6].lower()) else x["standalone_question"][6:],
        }

    def setup_model(self, optional: bool = False):
        google_gemini = ChatGoogleGenerativeAI(model="gemini-ulti")
        chat_openai = ChatOpenAI(model="gpt-4-0125-preview")
        chat_openai_fallback = ChatOpenAI(model="gpt-3.5-turbo-0125")
        mistral = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")

        if optional:
            model = (
                mistral
                .with_fallbacks([chat_openai_fallback, google_gemini])
                .configurable_alternatives(
                    ConfigurableField(id="model"),
                    default_key="mistral",
                    chat_openai_fallback=chat_openai_fallback,
                    google_gemini=google_gemini
                )
            )
        else:
            model = (
                chat_openai_fallback
                .with_fallbacks([chat_openai, mistral, google_gemini])
                .configurable_alternatives(
                    ConfigurableField(id="model"),
                    default_key="chat_openai_fallback",
                    chat_openai=chat_openai,
                    mistral=mistral,
                    google_gemini=google_gemini
                )
            )
        return model

    async def get_response(self, question):
        inputs = {"question": question}

        content = ""
        async for chunk in self.final_chain.astream(inputs):
            yield chunk.content
            content += chunk.content

        self.memory.save_context({"question": question}, {"answer": content})
        self.memory.load_memory_variables({})


async def main():
    lang_chain_service = LangChainService()
    async for response in lang_chain_service.get_response("Anything Else?"):
        print(response, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
