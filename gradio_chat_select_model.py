import os
import json
import tiktoken
import gradio as gr
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


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

def get_apikey(model: str) -> str:
    """
    指定されたモデルのAPIキーを返します。まず環境変数から取得を試み、
    見つからない場合はsecret.jsonファイルから取得します。

    :param model: モデル名
    :return: APIキー
    :raises ValueError: モデル名が無効な場合
    :raises FileNotFoundError: secret.jsonファイルが見つからない場合
    """
    try:
        secret_path = "secret.json"
        if os.path.exists(secret_path):
            with open(secret_path) as f:
                secret = json.load(f)
            if model == "gpt-4o-mini":
                return secret["OPENAI_API_KEY"]
            elif model == "claude-3-5-sonnet-20240620":
                return secret["ANTHROPIC_API_KEY"]
        else:
            raise FileNotFoundError("secret.json file not found")
    except (KeyError, FileNotFoundError) as e:
        raise ValueError(f"Error obtaining API key: {str(e)}")
    
    raise ValueError("Invalid model name")

def get_message_counts(text: str, model_name: str) -> int:
    """
    指定されたテキストのトークン数を計算します。

    :param text: テキスト
    :param model_name: モデル名
    :return: トークン数
    """
    if "gpt" in model_name:
        encoding = tiktoken.encoding_for_model(model_name)
    else:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 仮のものを利用
    return len(encoding.encode(text))

def calculate_costs(message_history: List[Tuple[str, str]], model_name: str) -> Tuple[float, float, float]:
    """
    メッセージ履歴に基づいてコストを計算します。

    :param message_history: メッセージ履歴
    :param model_name: モデル名
    :return: 総コスト, 入力コスト, 出力コスト
    """
    output_cnt = sum(get_message_counts(message, model_name) for role, message in message_history if role == "ai")
    input_cnt = sum(get_message_counts(message, model_name) for role, message in message_history if role != "ai")
    
    input_cost = input_cnt * MODEL_PRICES["input"][model_name]
    output_cost = output_cnt * MODEL_PRICES["output"][model_name]
    total_cost = input_cost + output_cost
    
    return total_cost, input_cost, output_cost

def init_chain(model_name: str, temperature: float, api_key: str) -> ChatPromptTemplate:
    """
    LLMチェーンを初期化します。

    :param model_name: モデル名
    :param temperature: 温度
    :param api_key: APIキー
    :return: 初期化されたLLMチェーン
    :raises ValueError: モデル名が無効な場合
    """
    if model_name == "gpt-4o-mini":
        llm = ChatOpenAI(
            api_key=api_key, 
            model_name="gpt-4o-mini", 
            temperature=temperature,
            streaming=False
            )
    elif model_name == "claude-3-5-sonnet-20240620":
        llm = ChatAnthropic(
            api_key=api_key, 
            model_name="claude-3-5-sonnet-20240620", 
            temperature=temperature,
            streaming=False)
    else:
        raise ValueError("Invalid model name")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{user_input}")
    ])
    
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

def chat(user_input: str, history: List[Tuple[str, str]], model: str, temperature: float) -> Tuple[List[Tuple[str, str]], str]:
    """
    チャット応答を生成し、コスト情報を返します。

    :param user_input: ユーザー入力
    :param history: チャット履歴
    :param model: モデル名
    :param temperature: 温度
    :return: 更新されたチャット履歴, コスト情報
    """
    try:
        api_key = get_apikey(model)
        chain = init_chain(model, temperature, api_key)
        
        # LLMの応答を生成
        ai_response = chain.invoke({"user_input": user_input})
        
        # 履歴を更新
        history.append(("user", user_input))
        history.append(("ai", ai_response))
        
        # コスト計算
        total_cost, input_cost, output_cost = calculate_costs(history, model)
        cost_info = f"総コスト: ${total_cost:.5f} (入力: ${input_cost:.5f}, 出力: ${output_cost:.5f})"
        
        return history, cost_info
    except Exception as e:
        return history, f"エラーが発生しました: {str(e)}"

def create_interface() -> gr.Interface:
    """
    Gradioインターフェースを作成します。

    :return: Gradioインターフェース
    """
    with gr.Blocks() as demo:
        gr.Markdown("# My Great ChatGPT 🤗")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="チャット履歴")
                msg = gr.Textbox(label="メッセージを入力してください")
                clear = gr.Button("会話をクリア")
            
            with gr.Column(scale=1):
                model = gr.Radio(["gpt-4o-mini", "claude-3-5-sonnet-20240620"], label="モデルを選択", value="gpt-4o-mini")
                temperature = gr.Slider(0, 1, value=0, step=0.01, label="Temperature")
                cost_info = gr.Textbox(label="コスト情報", interactive=False)
        
        def on_model_change(model_choice):
            return "gpt-4o-mini" if model_choice == "gpt-4o-mini" else "claude-3-5-sonnet-20240620"
        
        msg.submit(chat, [msg, chatbot, model, temperature], [chatbot, cost_info])
        clear.click(lambda: [], [], chatbot, queue=False)
        model.change(on_model_change, model, model)
    
    return demo

def main() -> None:
    """
    メイン関数。Gradioインターフェースを起動します。
    """
    demo = create_interface()
    demo.launch()

if __name__ == "__main__":
    main()
