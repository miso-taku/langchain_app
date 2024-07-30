import os
import json
import gradio as gr
from typing import List, Tuple, Optional, Dict
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader

def get_openai_apikey() -> str:
    """
    OpenAI APIキーを取得します。

    環境変数またはsecret.jsonファイルからAPIキーを取得します。

    Returns:
        str: OpenAI APIキー

    Raises:
        ValueError: APIキーの取得に失敗した場合
    """
    try:
        # 環境変数からAPIキーを取得
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key

        # secret.jsonファイルからAPIキーを取得
        secret_path = "secret.json"
        if os.path.exists(secret_path):
            with open(secret_path) as f:
                secret = json.load(f)
                return secret["OPENAI_API_KEY"]
        else:
            raise FileNotFoundError("secret.json file not found")
    except (KeyError, FileNotFoundError) as e:
        raise ValueError(f"Error obtaining API key: {str(e)}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    PDFファイルからテキストを抽出します。

    Args:
        pdf_path (str): PDFファイルのパス

    Returns:
        str: 抽出されたテキスト
    """
    pdf_text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def process_pdf(pdf_file: gr.File) -> Tuple[str, Optional[FAISS]]:
    """
    PDFファイルを処理し、テキストを抽出してベクトルストアを構築します。

    Args:
        pdf_file (gr.File): アップロードされたPDFファイル

    Returns:
        Tuple[str, Optional[FAISS]]: 処理結果メッセージとベクトルストアのタプル
    """
    try:
        # PDFからテキストを抽出
        pdf_text = extract_text_from_pdf(pdf_file.name)
        
        if not pdf_text.strip():
            return "PDFからテキストを抽出できませんでした。PDFが空か、テキスト抽出に対応していない可能性があります。", None

        # テキストを分割
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small",
            chunk_size=500,
            chunk_overlap=0
        )
        split_texts = text_splitter.split_text(pdf_text)

        # ベクトルストアを構築
        vectorstore = FAISS.from_texts(
            split_texts,
            OpenAIEmbeddings(
                api_key=get_openai_apikey(),
                model="text-embedding-3-small"
            )
        )

        return "PDFの処理が完了しました。質問を入力してください。", vectorstore
    except Exception as e:
        return f"PDFの処理中にエラーが発生しました: {str(e)}\n詳細: {type(e).__name__}", None

def answer_question(question: str, vectorstore: Optional[FAISS], model_name: str) -> str:
    """
    ユーザーの質問に対して回答を生成します。

    Args:
        question (str): ユーザーからの質問
        vectorstore (Optional[FAISS]): FAISSベクトルストア
        model_name (str): 使用するGPTモデルの名前

    Returns:
        str: 生成された回答
    """
    if vectorstore is None:
        return "PDFがアップロードされていないか、処理中にエラーが発生しました。先にPDFをアップロードしてください。"

    # モデル名の検証と適切なモデルの選択
    if model_name == "GPT-4o mini":
        model = "gpt-4o-mini"
    elif model_name == "GPT-4o":
        model = "gpt-4o"
    else:
        return f"サポートされていないモデル: {model_name}"

    # LLMの初期化
    llm = ChatOpenAI(
        api_key=get_openai_apikey(),
        temperature=0,
        model_name=model
    )

    # プロンプトの設定
    prompt = ChatPromptTemplate.from_template("""
    以下の前提知識を用いて、ユーザーからの質問に答えてください。
    
    ===
    前提知識
    {context}
    
    ===
    ユーザーからの質問
    {question}                                              
    """)

    # リトリーバーの設定
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # チェーンの構築
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 回答の生成
    return chain.invoke(question)

def pdf_qa_app() -> gr.Blocks:
    """
    Gradio PDFのQ&Aアプリケーションを構築します。
    モデル選択機能を含みます。

    Returns:
        gr.Blocks: Gradioアプリケーションのインターフェース
    """
    with gr.Blocks() as app:
        gr.Markdown("# PDF QA")
        gr.Markdown("PDFをアップロードし、質問に対する回答を取得するアプリケーション")

        with gr.Row(): # ここから横方向に要素配置
            with gr.Column(): # ここから縦方向に要素配置
                # モデル選択コンポーネント
                model_dropdown = gr.Dropdown(
                    choices=["GPT-4o mini", "GPT-4o"],
                    label="モデルを選択",
                    value="GPT-4o mini"
                )

                # PDFアップロードコンポーネント
                pdf_file = gr.File(label="PDFをアップロード")
                upload_button = gr.Button("PDFを処理")
                result = gr.Textbox(label="処理結果")
        
            with gr.Column(): # ここから縦方向に要素配置
                # 質問応答コンポーネント
                question_input = gr.Textbox(label="質問を入力してください")
                answer_button = gr.Button("回答を取得")
                answer_output = gr.Textbox(label="回答")

        # ベクトルストアを保存するための状態変数
        vectorstore_state = gr.State()

        # PDFアップロードと処理のイベントハンドラ
        upload_button.click(
            fn=process_pdf,
            inputs=[pdf_file],
            outputs=[result, vectorstore_state]
        )

        # 質問応答のイベントハンドラ
        answer_button.click(
            fn=answer_question,
            inputs=[question_input, vectorstore_state, model_dropdown],
            outputs=[answer_output]
        )

    return app

if __name__ == "__main__":
    pdf_qa_app().launch()
