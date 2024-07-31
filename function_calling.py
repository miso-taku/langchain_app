import json

from langchain.agents import tool
from langchain.output_parsers import JsonOutputToolsParser
from langchain_openai import ChatOpenAI

def get_openai_apikey() -> str:
    secret_path = "secret.json"
    with open(secret_path) as f:
        secret = json.load(f)
        return secret["OPENAI_API_KEY"]



llm = ChatOpenAI(api_key=get_openai_apikey(), model="gpt-4o-mini")

@tool
def get_word_length(word: str) -> int:
    """ Returns the length of the word """
    return len(word)

llm_with_tools = llm.bind_tools([get_word_length])
chain = llm_with_tools | JsonOutputToolsParser()
res = chain.invoke("abcdef って何文字？")
print(res)