# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_009/main.py
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
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

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

CUSTOM_SYSTEM_PROMPT = """
あなたは、ユーザーのリクエストに基づいてインターネットで調べ物を行うアシスタントです。
利用可能なツールを使用して、調査した情報を説明してください。
既に知っていることだけに基づいて答えないでください。回答する前にできる限り検索を行ってください。
(ユーザーが読むページを指定するなど、特別な場合は、検索する必要はありません。)

検索結果ページを見ただけでは情報があまりないと思われる場合は、次の2つのオプションを検討して試してみてください。

- 検索結果のリンクをクリックして、各ページのコンテンツにアクセスし、読んでみてください。
- 1ページが長すぎる場合は、3回以上ページ送りしないでください（メモリの負荷がかかるため）。
- 検索クエリを変更して、新しい検索を実行してください。
- 検索する内容に応じて検索に利用する言語を適切に変更してください。
  - 例えば、プログラミング関連の質問については英語で検索するのがいいでしょう。

ユーザーは非常に忙しく、あなたほど自由ではありません。
そのため、ユーザーの労力を節約するために、直接的な回答を提供してください。

=== 悪い回答の例 ===
- これらのページを参照してください。
- これらのページを参照してコードを書くことができます。
- 次のページが役立つでしょう。

=== 良い回答の例 ===
- これはサンプルコードです。 -- サンプルコードをここに --
- あなたの質問の答えは -- 回答をここに --

回答の最後には、参照したページのURLを**必ず**記載してください。（これにより、ユーザーは回答を検証することができます）

ユーザーが使用している言語で回答するようにしてください。
ユーザーが日本語で質問した場合は、日本語で回答してください。ユーザーがスペイン語で質問した場合は、スペイン語で回答してください。
"""


def init_page():
    st.set_page_config(
        page_title="Web Browsing Agent",
        page_icon="🤗"
    )
    st.header("Web Browsing Agent 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "こんにちは！なんでも質問をどうぞ！"}
        ]
        st.session_state['memory'] = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            k=10
        )

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

def create_agent():
    tools = [search_ddg, fetch_page]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = select_model()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state['memory']
    )


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_agent()

    for msg in st.session_state['memory'].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="2023 FIFA 女子ワールドカップの優勝国は？"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # コールバック関数の設定 (エージェントの動作の可視化用)
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=True)

            # エージェントを実行
            response = web_browsing_agent.invoke(
                {'input': prompt},
                config=RunnableConfig({'callbacks': [st_cb]})
            )
            st.write(response["output"])


if __name__ == '__main__':
    main()
