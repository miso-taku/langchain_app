import json
from typing import Union
from operator import itemgetter

from langchain.agents import tool
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain.output_parsers import JsonOutputToolsParser
from langchain_openai import ChatOpenAI


def get_openai_apikey() -> str:
    secret_path = "secret.json"
    with open(secret_path) as f:
        secret = json.load(f)
        return secret["OPENAI_API_KEY"]



llm = ChatOpenAI(api_key=get_openai_apikey(), model="gpt-4o-mini")

@tool
def add(first_int: int, second_int: int) -> int:
    """ Add two integers """
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    """ Raise base to the power of exponent """
    return base ** exponent

@tool
def multiply(first_int: int, second_int: int) -> int:
    """ Multiply two integers """
    return first_int * second_int

tools = [add, exponentiate, multiply]
tool_map = {tool.name: tool for tool in tools}

def call_tool(tool_invocation: dict) -> Union[str, Runnable]:
    tool = tool_map[tool_invocation["type"]]
    return RunnablePassthrough.assign(output=itemgetter("args") | tool)

llm_with_tools = llm.bind_tools([add, exponentiate, multiply])

chain = (llm_with_tools | JsonOutputToolsParser() | RunnableLambda(call_tool).map())

res = chain.invoke("""
以下の計算をしてください。
- 123 + 4567
- 123 * 4567
- 12**12
""")

print(res)
