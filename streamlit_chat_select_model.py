# LLMモデルを選択できるStreamlit chatアプリケーション
# ライブラリのインポート
import json

import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# OpenAI APIキーの取得
def get_gpt_openai_apikey() -> str:
    with open("secret.json") as f:
        secret = json.load(f)
    return secret["OPENAI_API_KEY"]

# Anthropic APIキーの取得
def get_anthropic_apikey() -> str:
    with open("secret.json") as f:
        secret = json.load(f)
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

    models = ("GPT-4o mini", "Claude 3.5 Sonnet")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-4o mini":
        st.session_state.model_name = "gpt-3.5-turbo"
        return ChatOpenAI(
            temperature=temperature,
            model_name=st.session_state.model_name
        )
    elif model == "Claude 3.5 Sonnet":
        st.session_state.model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            temperature=temperature,
            model_name=st.session_state.model_name
        )

# chainの初期化
def init_chain():
    # チェーンの初期化
    st.session_state.llm = select_model()
    prompt = ChatPromptTemplate([
        *st.session_state.message_history,
        ("user", "{user_input}")
    ])
    output_parser = StrOutputParser()
    return prompt | st.session_state.llm | output_parser
