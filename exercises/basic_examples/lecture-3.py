from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.chat_models import ChatFireworks
from langchain_community.chat_models import GigaChat
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, ConfigurableField

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
chat_openai = ChatOpenAI(model="gpt-3.5-turbo")
openai = OpenAI(model="gpt-3.5-turbo-instruct")
mistral = ChatFireworks(model="mistral-small")
gigachat = GigaChat(verify_ssl_certs=False)
model = (
    chat_openai
    .with_fallbacks([mistral, gigachat])
    .configurable_alternatives(
        ConfigurableField(id="model"),
        default_key="chat_openai",
        openai=openai,
        gigachat=gigachat,
        mistral=mistral,
    )
)

chain = (
        {"topic": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

print(chain.invoke("Ice cream"))
