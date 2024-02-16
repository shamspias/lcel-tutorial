import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo-0125")
output_parser = JsonOutputParser()


def _extract_country_names(inputs):
    """A function that does not operates on input streams and breaks streaming."""
    if not isinstance(inputs, dict):
        return ""

    if "countries" not in inputs:
        return ""

    countries = inputs["countries"]

    if not isinstance(countries, list):
        return ""

    country_names = [
        country.get("name") for country in countries if isinstance(country, dict)
    ]
    return country_names


chain = (
        model | output_parser | _extract_country_names
)  # This parser only works with OpenAI right now


async def main():
    async for chunk in chain.astream(
            'output a list of the countries france, spain there some other like  japan and bangladesh and their '
            'populations in JSON format. Use a dict with'
            'an outer key of "countries" which contains a list of countries. Each country should have the key `name` '
            'and `population`',
    ):
        print(chunk, flush=True)


# Run the main coroutine
if __name__ == "__main__":
    asyncio.run(main())
