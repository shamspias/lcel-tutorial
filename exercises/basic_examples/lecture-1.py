from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo-0125")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print(chain.invoke({"topic": "ice cream"}))
