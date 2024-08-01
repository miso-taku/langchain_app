import json
from typing import Optional
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

def get_openai_apikey() -> str:
    secret_path = "secret.json"
    with open(secret_path) as f:
        secret = json.load(f)
        return secret["OPENAI_API_KEY"]

class Item(BaseModel):
    item_name: str = Field(description="商品名")
    price: Optional[int] = Field(description="価格")
    color: Optional[str] = Field(description="色")

system = "与えられた商品の情報を構造化してください。"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{item_info}"),
    ]
)

llm = ChatOpenAI(api_key=get_openai_apikey(), model="gpt-4o-mini")
structued_llm = llm.with_structured_output(Item)
chain = prompt | structued_llm
res = chain.invoke({"item_info": "林檎　赤　100円"})
print(res)
print(res.json(ensure_ascii=False))
res = chain.invoke({"item_info": "バナナ　80円"})
print(res)