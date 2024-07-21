import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_gpt_openai_apikey() -> str:
    # secret.jsonファイルからOpenAIのAPIキーを読み込む
    with open("secret.json") as f:
        secret = json.load(f)
    return secret["OPENAI_API_KEY"]

def main():
    # ユーザーからの入力を受け取る
    user_input = input("聞きたいことを入力してや？: ")

    # ChatOpenAIインスタンスの作成
    llm = ChatOpenAI(
        api_key=get_gpt_openai_apikey(),
        model="gpt-4o-mini",
        temperature=0.8
    )

    # ユーザーの質問を受取、ChatGPTに渡すためのテンプレートを作成
    prompt = ChatPromptTemplate.from_messages([
        ("system", "絶対に関西弁で返答してください"),
        ("user", user_input)
    ])

    # 出力パーサーのインスタンスを作成
    output_parser = StrOutputParser()

    # プロンプト、LLM、出力パーサーをチェーンで連結
    chain = prompt | llm | output_parser

    # チェーンを実行してレスポンスを取得
    response = chain.invoke({"input": user_input})

    print(response)


if __name__ == "__main__":
    main()