# LLMモデルを選択できるStreamlit chatアプリケーション
# ライブラリのインポート
import getpass
import json
import os

import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# APIキーの取得
def get_apikey(model: str) -> str:
    with open("secret.json") as f:
        secret = json.load(f)
    if model == "gpt-4o mini":
        return secret["OPENAI_API_KEY"]
    elif model == "Claude 3.5 Sonnet":
        return secret["ANTHROPIC_API_KEY"]

# モデルの料金情報
MODEL_PRICES = {
    "input": {
        "gpt-4o-mini": 0.15 / 1_000_000,
        "claude-3-5-sonnet-20240620": 3 / 1_000_000
    },
    "output": {
        "gpt-4o-mini": 0.6 / 1_000_000,
        "claude-3-5-sonnet-20240620": 15 / 1_000_000
    }
}

# 画面の初期化
def init_page():
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")

# メッセージの初期化
def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    # clear_button が押された場合や message_history がまだ存在しない場合に初期化
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ]

# モデルの選択
def select_model():
    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.01とする
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    models = ("gpt-4o mini", "Claude 3.5 Sonnet")
    model = st.sidebar.radio("Choose a model:", models)
    api_key = get_apikey(model)
    if model == "gpt-4o mini":
        st.session_state.model_name = "gpt-4o-mini"
        return ChatOpenAI(
            api_key=api_key,
            model_name=st.session_state.model_name,
            temperature=temperature
        )
    elif model == "Claude 3.5 Sonnet":
        st.session_state.model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            api_key=api_key,
            model_name=st.session_state.model_name,
            temperature=temperature
        )

# chainの初期化
def init_chain():
    # チェーンの初期化
    st.session_state.llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        *st.session_state.message_history,
        ("user", "{user_input}")
    ])

    output_parser = StrOutputParser()
    return prompt | st.session_state.llm | output_parser

# メッセージのコスト計算
def get_message_counts(text):
    if "gpt" in st.session_state.model_name:
        encoding = tiktoken.encoding_for_model(st.session_state.model_name)
    else:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 仮のものを利用
    return len(encoding.encode(text))

def calc_and_display_costs():
    output_cnt = 0
    input_cnt = 0
    for role, message in st.session_state.message_history:
        # tiktokenでトークン数を計算
        token_cnt = get_message_counts(message)
        if role == "ai":
            output_cnt += token_cnt
        else:
            input_cnt += token_cnt
    # 初期状態で System Message のみが履歴に入っている場合はまだAPIコールが行われていない
    if len(st.session_state.message_history) == 1:
        return
    
    input_cost = input_cnt * MODEL_PRICES["input"][st.session_state.model_name] * input_cnt
    output_cost = output_cnt * MODEL_PRICES["output"][st.session_state.model_name] * output_cnt
    
    cost = output_cost + input_cost

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${cost:.5f}**")
    st.sidebar.markdown(f"- Input cost: ${input_cost:.5f}")
    st.sidebar.markdown(f"- Output cost: ${output_cost:.5f}")


def main():
    init_page()
    init_messages()
    chain = init_chain()

    # チャット履歴の表示
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.chat_message('user').markdown(user_input)

        # LLMの返答を Streaming 表示する
        with st.chat_message('ai'):
            response = st.write_stream(chain.stream({"user_input": user_input}))

        # チャット履歴に追加
        st.session_state.message_history.append(("user", user_input))
        st.session_state.message_history.append(("ai", response))

    # コストを計算して表示
    calc_and_display_costs()


if __name__ == '__main__':
    main()
