from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.chat_models import ChatFireworks
from langchain_community.chat_models import GigaChat
from langchain_groq.chat_models import ChatGroq
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, ConfigurableField

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
chat_openai = ChatOpenAI(model="gpt-3.5-turbo")
firework_models = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")
mistral = ChatMistralAI(model="mistral-small")
groq_models = ChatGroq(model_name="mixtral-8x7b-32768")
gigachat = GigaChat(verify_ssl_certs=False)
model = (
    groq_models
    .with_fallbacks([mistral, firework_models, chat_openai, gigachat])
    .configurable_alternatives(
        ConfigurableField(id="model"),
        default_key="groq_models",
        gigachat=gigachat,
        chat_openai=chat_openai,
        mistral=mistral,
        firework_models=firework_models
    )
)

chain = (
        {"topic": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

print(chain.invoke("Ice cream"))
