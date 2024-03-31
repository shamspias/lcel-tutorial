from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatVertexAI(model_name="gemini-1.0-pro-vision-001")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

print(chain.invoke({"topic": "ice cream"}))
