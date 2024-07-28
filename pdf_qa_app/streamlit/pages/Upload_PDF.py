# streamlit_pdf_qa.pyから呼び出されるPDFファイルをアップロードするページ
# ライブラリのインポート
import json
import os

import fitz
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def innit_page() -> None:
    """
    Streamlitページの初期化を行います。
    """
    st.set_page_config(
        page_title="PDF QA",
    )
    st.sidebar.title("Options")

def get_pdf_text() -> str:
    """
    PDFのテキスト情報を取得する
    """
    pdf_file = st.file_uploader(
        label="Upload PDF",
        type="pdf"
    )

    if pdf_file:
        pdf_text =""
        with st.spinner("Loading PDF ..."):
            # PyPDFでPDFを読み取り
            pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in pdf_doc:
                pdf_text += page.get_text()

        # RecursiveCharacterTextSplotterでチャンクに分割
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small",
            chunk_size =500,
            chunk_overlap=0
        )
        return text_splitter.split_text(pdf_text)
    else:
        return None
    
def build_vector_store(pdf_text: str) -> None:
    """
    ベクトルDBの構築
    """
    with st.spinner("Saving to vector store ..."):
        if 'vectorsore' in st.session_state:
            st.session_state.vectorstore.add_texts(pdf_text)
        else:
            # ベクトルDBの初期化と文書の追加
            # st.session_state.vectorstore = FAISS.fromm_texts(
            #     pdf_text,
            #     OpenAIEmbeddings(model="text-embedding-3-small")
            # )
            from langchain_community.vectorstores.utils import DistanceStrategy
            st.session_state.vectorstore = FAISS.from_texts(
                pdf_text,
                OpenAIEmbeddings(
                    api_key=get_openai_apikey(),
                    model="text-embedding-3-small"
                    ),
                distance_strategy=DistanceStrategy.COSINE
            )

def page_pdf_upload_and_build_vector_db() -> None:
    st.title("PDF upload")
    pdf_text = get_pdf_text()
    if pdf_text:
        build_vector_store(pdf_text)

def main() -> None:
    innit_page()
    page_pdf_upload_and_build_vector_db()

if __name__ == "__main__":
    main()