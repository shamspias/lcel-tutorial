import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo-0125")
output_parser = StrOutputParser()

chain = prompt | model | output_parser


# Sync
# Invoke
# print(chain.invoke({"topic": "ice cream"}))
# Stream
# for chunk in chain.stream({"topic": "Ice Cream"}):
#     print(chunk, end="", flush=True)
# Batch
# print(chain.batch([{"topic": "ice cream"}, {"topic": "Cake"}, {"topic": "Football"}]))

# Async
# Invoke
# async def main():
#     await chain.ainvoke({"topic": "ice cream"})
#

# Stream
async def main():
    async for chunk in chain.astream({"topic": "Ice Cream"}):
        print(chunk, end="", flush=True)


# Batch
# async def main():
#     print(await chain.abatch([{"topic": "ice cream"}, {"topic": "Cake"}, {"topic": "Football"}]))


# Run the main coroutine
if __name__ == "__main__":
    asyncio.run(main())
