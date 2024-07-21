import json

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


def get_gpt_openai_apikey() -> str:
    # secret.jsonファイルからOpenAIのAPIキーを読み込む
    with open("secret.json") as f:
        secret = json.load(f)
    return secret["OPENAI_API_KEY"]


def chatbot(message, history):

    # ChatOpenAIモデルの初期化
    chat = ChatOpenAI(
        api_key=get_gpt_openai_apikey(),
        model="gpt-4o-mini",
        temperature=0.8
    )

    # システムメッセージの初期化
    system_message = SystemMessage(content="You are a helpful AI assistant.")

    # チャット履歴をLangChain形式に変換
    messages = [system_message]
    for human, ai in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    
    # 新しい人間のメッセージを追加
    messages.append(HumanMessage(content=message))
    
    # ChatOpenAIモデルを使用して応答を生成
    response = chat(messages)
    
    # AIの応答を返す
    return response.content

def main():
    # インターフェースの作成
    iface = gr.ChatInterface(
        chatbot,
        title="LangChain ChatOpenAI Assistant",
        description="ChatGPTを使用したシンプルなチャットボットです。質問してみてください。"
    )

    # インターフェースの起動
    iface.launch()

if __name__ == "__main__":
    main()