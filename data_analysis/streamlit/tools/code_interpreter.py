import streamlit as st
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

class ExecPythonInput(BaseModel):
    """Pythonコードの入力を表すモデルクラス"""
    code: str = Field()

@tool(args_schema=ExecPythonInput)
def code_interpreter_tool(code: str) -> dict:
    """
    Code Interpreter を使用して、Pythonコードを実行します。
    
    Args:
        code (str): 実行するPythonコード
    
    Returns:
        dict: Code Interpreter の実行結果
            - text: Code Interpreter が出力したテキスト（コード実行結果が主）
            - files: Code Interpreter が保存したファイルのパス
              （ファイルは `./files/` 以下に保存されます）
    
    Note:
        - pandasやmatplotlibなどのライブラリを使ってデータの加工や可視化が可能
        - 数式の計算や統計的な分析も可能
        - 自然言語処理ライブラリを使ったテキストデータの分析も可能
        - Code Interpreterはインターネットに接続できないため、外部サイトの情報取得や新しいライブラリのインストールは不可
        - Code Interpreterに実行したコードも出力させるようにすると、ユーザーが結果を検証しやすくなる
        - 多少のコードミスは自動修正されることがある
    """
    return st.session_state.code_interpreter_client.run(code)

