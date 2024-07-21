import json

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_gpt_openai_apikey() -> str:
    # secret.jsonファイルからOpenAIのAPIキーを読み込む
    with open("secret.json") as f:
        secret = json.load(f)
    return secret["OPENAI_API_KEY"]

def main():
    # Streamlitアプリのページ設定
    st.set_page_config(
        page_title="ChatGPT app",
        page_icon="🤖"
    )
    st.header("ChatGPT app")  # ヘッダーの設定

    # セッション状態にメッセージ履歴がなければ初期化
    if "message history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "絶対に関西弁で返答してください")
        ]
    
    # ChatOpenAIインスタンスの作成
    llm = ChatOpenAI(
        api_key=get_gpt_openai_apikey(),
        model="gpt-4o-mini",
        temperature=0.8
    )

    # ユーザーとAIのメッセージ履歴からプロンプトを作成
    prompt = ChatPromptTemplate.from_messages([
        *st.session_state.message_history,
        ("user", "{user_message}")
    ])

    # 出力パーサーのインスタンスを作成
    output_parser = StrOutputParser()

    # プロンプト、LLM、出力パーサーをチェーンで連結
    chain = prompt | llm | output_parser

    # ユーザーからの入力を受け取る
    if user_message := st.chat_input("聞きたいことを入力してや。"):
        with st.spinner("AI が考え中..."):
            # チェーンを実行してレスポンスを取得
            response = chain.invoke({"user_message": user_message})

        # ユーザーメッセージをセッション状態のメッセージ履歴に追加
        st.session_state.message_history.append(("user", user_message))

        # AIのレスポンスをセッション状態のメッセージ履歴に追加
        st.session_state.message_history.append(("ai", response))

        # メッセージ履歴を表示
        for role, message in st.session_state.get("message_history", []):
            # st.chat_message(role).markdown(message)
            with st.chat_message(role):
                st.markdown(message)
                

if __name__ == "__main__":
    main()