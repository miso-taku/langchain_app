import json
import os
import re
import streamlit as st
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from src.code_interpreter import CodeInterpreterClient
from tools.code_interpreter import code_interpreter_tool


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


@st.cache_data
def load_system_prompt(file_path: str) -> str:
    """システムプロンプトをファイルから読み込む関数
    
    Args:
        file_path (str): プロンプトファイルのパス
    
    Returns:
        str: 読み込まれたプロンプトの内容
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def csv_upload() -> None:
    """CSVファイルをアップロードするフォームを表示し、ファイルをセッションに保存する関数"""
    with st.form("csv-upload-form", clear_on_submit=True):
        file = st.file_uploader(label='Upload your CSV here😇', type='csv')
        submitted = st.form_submit_button("Upload CSV")
        if submitted and file is not None:
            if file.name not in st.session_state.uploaded_files:
                assistant_api_file_id = st.session_state.code_interpreter_client.upload_file(file.read())
                st.session_state.custom_system_prompt += \
                    f"\nアップロードファイル名: {file.name} (Code Interpreterでのpath: /mnt/data/{assistant_api_file_id})\n"
                st.session_state.uploaded_files.append(file.name)
        else:
            st.write("データ分析したいファイルをアップロードしてね")

    if st.session_state.uploaded_files:
        st.sidebar.markdown("## Uploaded files:")
        for file_name in st.session_state.uploaded_files:
            st.sidebar.markdown(f"- {file_name}")


def initialize_page() -> None:
    """ページの初期設定を行う関数"""
    st.set_page_config(page_title="Data Analysis Agent", page_icon="🤗")
    st.header("Data Analysis Agent 🤗", divider='rainbow')
    st.sidebar.title("Options")

    # 初期化ボタンとメッセージ・セッションの初期化
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.code_interpreter_client = CodeInterpreterClient()
        st.session_state.memory = ConversationBufferWindowMemory(
            return_messages=True, memory_key="chat_history", k=10
        )
        st.session_state.custom_system_prompt = load_system_prompt("./prompt/system_prompt.txt")
        st.session_state.uploaded_files = []


def select_model() -> ChatOpenAI:
    """ユーザーが選択したモデルを返す関数
    
    Returns:
        ChatOpenAI: 選択されたモデルのインスタンス
    """
    models = ("gpt-4o-mini", "gpt-4o")
    model = st.sidebar.radio("Choose a model:", models)
    return ChatOpenAI(api_key=get_openai_apikey(),temperature=0, model_name=model)


def create_agent() -> AgentExecutor:
    """エージェントを作成する関数
    
    Returns:
        AgentExecutor: 作成されたエージェント
    """
    tools = [code_interpreter_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system", st.session_state.custom_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = select_model()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, memory=st.session_state.memory)


def parse_response(response: str) -> tuple[str, list[str]]:
    """response からテキストと画像パスを抽出する関数
    
    Args:
        response (str): エージェントのレスポンス
    
    Returns:
        tuple[str, list[str]]: テキストと画像パスのタプル
    """
    img_pattern = re.compile(r'<img\s+[^>]*?src="([^"]+)"[^>]*?>')
    image_paths = img_pattern.findall(response)
    text = img_pattern.sub('', response).strip()
    return text, image_paths


def display_content(content: str) -> None:
    """レスポンスの内容を表示する関数
    
    Args:
        content (str): エージェントのレスポンス内容
    """
    text, image_paths = parse_response(content)
    st.write(text)
    for image_path in image_paths:
        st.image(image_path, caption="")


def main() -> None:
    """メイン関数"""
    initialize_page()
    csv_upload()
    data_analysis_agent = create_agent()

    for msg in st.session_state.memory.chat_memory.messages:
        with st.chat_message(msg.type):
            display_content(msg.content)

    if prompt := st.chat_input(placeholder="分析したいことを書いてね"):
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = data_analysis_agent.invoke({'input': prompt}, config=RunnableConfig({'callbacks': [st_cb]}))
            display_content(response["output"])


if __name__ == '__main__':
    main()

