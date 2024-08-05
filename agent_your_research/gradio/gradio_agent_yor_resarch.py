# gradioを使用したWebブラウジングエージェント
# ライブラリのインポート
import os
import json
import gradio as gr
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Tuple, Optional

# custom tools
from tools.search_ddg import search_duckduckgo
from tools.fetch_page import fetch_page_content

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
あなたは、インターネット検索を駆使して、ユーザーのリクエストに的確かつ詳細に答える
アシスタントです。以下の点に留意して、日本語で回答してください。

・最新かつ信頼性の高い情報に基づいて回答する。
・直接的で明確な回答を心がけ、具体例や詳細な説明を加える。
・複数の情報源を参考に、多角的な視点で回答する。
・検索キーワードを工夫し、適切な言語で検索する。
・回答の最後に、参照したウェブサイトのURLを記載する。

== 望ましくない回答 ==

- 〇〇についてはこちらをご覧ください。
- このコードを参考にすれば解決できるかもしれません。

== 望ましい回答 ==

- 〇〇は△△という理由で□□という影響を与えると考えられています。（参照：URL）
- 〇〇の解決策としては、△△や□□といった方法が考えられます。（参照：URL）
- 〇〇のコード例は以下の通りです。（コードスニペット）（参照：URL）
"""

def create_agent(model_name: str) -> AgentExecutor:
    """
    指定されたモデル名を使用してエージェントを作成します。

    Args:
        model_name (str): 使用するモデル名

    Returns:
        AgentExecutor: 作成されたエージェント
    """
    tools = [search_duckduckgo, fetch_page_content]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = ChatOpenAI(
        api_key=get_openai_apikey(),
        temperature=0,
        model_name=model_name
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            k=10
        )
    )

def chat(message: str, history: List[Tuple[str, str]], model_name: str) -> str:
    """
    チャットメッセージを処理し、エージェントからの応答を取得します。

    Args:
        message (str): ユーザーからのメッセージ
        history (List[Tuple[str, str]]): チャット履歴
        model_name (str): 使用するモデル名

    Returns:
        str: エージェントからの応答
    """
    agent = create_agent(model_name)
    
    # Convert Gradio history to LangChain format
    for human, ai in history:
        agent.memory.chat_memory.add_user_message(human)
        agent.memory.chat_memory.add_ai_message(ai)
    
    response = agent.invoke({'input': message})
    
    # Return the user message and AI response as a pair
    return history + [[message, response["output"]]]

def clear_conversation() -> Optional[None]:
    """
    チャット履歴をクリアします。

    Returns:
        None: チャット履歴をクリアするためにNoneを返します
    """
    return None

with gr.Blocks() as demo:
    gr.Markdown("# Web Browsing Agent")
    
    with gr.Row():
        with gr.Column(scale=1):
            model = gr.Radio(["gpt-4o-mini", "gpt-4o"], label="Choose a model", value="gpt-4o-mini")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450)
            msg = gr.Textbox(label="Enter your message")
            clear = gr.Button("メッセージをクリア")
        
    
    msg.submit(chat, inputs=[msg, chatbot, model], outputs=[chatbot])
    clear.click(clear_conversation, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()
