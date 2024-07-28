# PDFに関する質問を受け付けて応答を返す
# モデルのインポート
import json
import os
from typing import Any

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI


def get_openai_apikey() -> str:
    """
    指定されたモデルのAPIキーを返します。まず環境変数から取得を試み、
    見つからない場合はsecret.jsonファイルから取得します。

    :return: APIキー
    :raises FileNotFoundError: secret.jsonファイルが見つからない場合
    """
    try:
        secret_path = "secret.json"
        if os.path.exists(secret_path):
            with open(secret_path) as f:
                secret = json.load(f)
                return secret["OPENAI_API_KEY"]
        else:
            raise FileNotFoundError("secret.json file not found")
    except (KeyError, FileNotFoundError) as e:
        raise ValueError(f"Error obtaining API key: {str(e)}")


def init_page() -> str:
    st.set_page_config(
        page_title="Ask My PDF"
    )
    st.sidebar.title("Options")


def select_model(temperature: float=0) -> ChatOpenAI:
    models = ("gpt-4o", "gpt-4o-mini")
    model = st.sidebar.radio("Chose a model", models)
    if model == "gpt-4o":
        return ChatOpenAI(
            api_key=get_openai_apikey(),
            temperature=temperature,
            model_name="gpt-4o"
        )
    elif model == "gpt-4o-mini":
        return ChatOpenAI(
            api_key=get_openai_apikey(),
            temperature=temperature,
            model_name="gpt-4o-mini"
        )


def init_qa_chain() -> Any:
    llm = select_model()
    prompt = ChatPromptTemplate.from_template("""
    以下の前提知識を用いて、ユーザーからの質問に答えてください。
    
    ===
    前提知識
    {context}
    
    ===
    ユーザーからの質問
    {question}                                              
    """)
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":5}
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def page_ask_my_pdf() -> None:
    chain = init_qa_chain()

    if query := st.text_input("質問を書いて下さい: ", key="input"):
        st.markdown("## Answer")
        st.write_stream(chain.stream(query))

def main():
    init_page()
    st.title("PDF QA")
    if "vectorstore" not in st.session_state:
        st.warning("まずはUpload PDFからPDFファイルをアップロードして下さい")
    else:
        page_ask_my_pdf()


if __name__ == "__main__":
    main()
