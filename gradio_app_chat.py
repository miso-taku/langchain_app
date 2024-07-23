import json
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_gpt_openai_apikey() -> str:
    # secret.jsonファイルからOpenAIのAPIキーを読み込む
    with open("secret.json") as f:
        secret = json.load(f)
    return secret["OPENAI_API_KEY"]

# メッセージ履歴を保持するグローバル変数
message_history = [
    ("system", "絶対に関西弁で返答してください")
]

def chat_with_ai(user_message, history):
    global message_history
    
    # ChatOpenAIインスタンスの作成
    llm = ChatOpenAI(
        api_key=get_gpt_openai_apikey(),
        model="gpt-4o-mini",
        temperature=0.8
    )

    # 出力パーサーのインスタンスを作成
    output_parser = StrOutputParser()

    # ユーザーメッセージをメッセージ履歴に追加
    message_history.append(("user", user_message))
    
    # プロンプトを作成
    prompt = ChatPromptTemplate.from_messages(message_history)
    
    # プロンプト、LLM、出力パーサーをチェーンで連結
    chain = prompt | llm | output_parser
    
    # チェーンを実行してレスポンスを取得
    response = chain.invoke({"user_message": user_message})
    
    # AIのレスポンスをメッセージ履歴に追加
    message_history.append(("ai", response))
    
    # Gradioの履歴形式に変換
    gradio_history = history + [(user_message, response)]
    
    # デバッグ用のプリント文
    print(f"User message: {user_message}")
    print(f"AI response: {response}")
    print(f"Gradio history: {gradio_history}")
    
    return gradio_history

# Gradioインターフェースの作成
with gr.Blocks() as iface:
    chatbot = gr.Chatbot(height=300)
    msg = gr.Textbox(placeholder="聞きたいことを入力してや。", container=False, scale=7)
    clear = gr.Button("会話をリセット")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = chat_with_ai(user_message, history[:-1])[-1][1]
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

# アプリの起動
if __name__ == "__main__":
    iface.launch()