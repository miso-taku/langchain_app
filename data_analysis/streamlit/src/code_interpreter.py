import os
import magic
import traceback
import mimetypes
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Optional

load_dotenv()

class CodeInterpreterClient:
    """
    OpenAI's Assistants API の Code Interpreter Tool を使用して
    Python コードを実行したり、ファイルを読み取って分析を行うクラス

    このクラスは以下の機能を提供します：
    1. OpenAI Assistants APIを使ったPythonコードの実行
    2. ファイルのアップロードとAssistants APIへの登録
    3. アップロードしたファイルを使ったデータ分析とグラフ作成

    主要なメソッド：
    - upload_file(file_content): ファイルをアップロードしてAssistants APIに登録する
    - run(prompt): Assistants APIを使ってPythonコードを実行したり、ファイル分析を行う

    Example:
    ===============
    from src.code_interpreter import CodeInterpreterClient
    code_interpreter = CodeInterpreterClient()
    code_interpreter.upload_file(open('file.csv', 'rb').read())
    code_interpreter.run("file.csvの内容を読み取ってグラフを書いてください")
    """

    def __init__(self, api_key: str):
        self.file_ids: List[str] = []
        self.openai_client = OpenAI(api_key=api_key)
        self.assistant_id = self._create_assistant_agent()
        self.thread_id = self._create_thread()
        self._create_file_directory()
        self.code_interpreter_instruction = """
        与えられたデータ分析用のPythonコードを実行してください。
        実行した結果を返して下さい。あなたの分析結果は不要です。
        もう一度繰り返します、実行した結果を返して下さい。
        ファイルのパスなどが少し間違っている場合は適宜修正してください。
        修正した場合は、修正内容を説明してください。
        """

    def _create_file_directory(self) -> None:
        """ファイルを保存するディレクトリを作成する"""
        directory = "./files/"
        os.makedirs(directory, exist_ok=True)

    def _create_assistant_agent(self) -> str:
        """
        OpenAIのアシスタントエージェントを作成する
        
        Returns:
            str: 作成されたアシスタントのID
        """
        self.assistant = self.openai_client.beta.assistants.create(
            name="Python Code Runner",
            instructions="You are a python code runner. Write and run code to answer questions.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4o",
            tool_resources={
                "code_interpreter": {
                    "file_ids": self.file_ids
                }
            }
        )
        return self.assistant.id

    def _create_thread(self) -> str:
        """
        OpenAIのスレッドを作成する
        
        Returns:
            str: 作成されたスレッドのID
        """
        thread = self.openai_client.beta.threads.create()
        return thread.id

    def upload_file(self, file_content: bytes) -> str:
        """
        ファイルをアップロードしてアシスタントエージェントに登録する
        
        Args:
            file_content (bytes): アップロードするファイルの内容
        
        Returns:
            str: アップロードされたファイルのID
        """
        file = self.openai_client.files.create(
            file=file_content,
            purpose='assistants'
        )
        self.file_ids.append(file.id)
        # アシスタントに新しいファイルを追加して更新する
        self._add_file_to_assistant_agent()
        return file.id

    def _add_file_to_assistant_agent(self) -> None:
        """ファイルをアシスタントエージェントに追加する"""
        self.assistant = self.openai_client.beta.assistants.update(
            assistant_id=self.assistant_id,
            tool_resources={
                "code_interpreter": {
                    "file_ids": self.file_ids
                }
            }
        )

    def run(self, code: str) -> Tuple[Optional[str], List[str]]:
        """
        与えられたPythonコードを実行して結果を返す
        
        Args:
            code (str): 実行するPythonコード
        
        Returns:
            Tuple[Optional[str], List[str]]: 実行結果のテキストと生成されたファイルのパス
        """
        prompt = f"""
        以下のコードを実行して結果を返して下さい。
        ファイルの読み込みなどに失敗した場合、可能な範囲で修正して再実行して下さい。
        ```python
        {code}
        ```
        あなたの見解や感想は不要なのでコードの実行結果を返して下さい
        """

        # スレッドにメッセージを追加
        self.openai_client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=prompt
        )

        # アシスタントを実行してレスポンスを取得
        run = self.openai_client.beta.threads.runs.create_and_poll(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            instructions=self.code_interpreter_instruction
        )

        if run.status == 'completed':
            message = self.openai_client.beta.threads.messages.list(
                thread_id=self.thread_id,
                limit=1  # 最後のメッセージを取得
            )
            try:
                text_content = ""
                file_ids = []
                for content in message.data[0].content:
                    if content.type == "text":
                        text_content = content.text.value
                        file_ids.extend([
                            annotation.file_path.file_id
                            for annotation in content.text.annotations
                        ])
                    elif content.type == "image_file":
                        file_ids.append(content.image_file.file_id)
                    else:
                        raise ValueError("Unknown content type")
            except:
                print(traceback.format_exc())
                return None, []
        else:
            raise ValueError("Run failed")

        file_names = [self._download_file(file_id) for file_id in file_ids]

        return text_content, file_names

    def _download_file(self, file_id: str) -> str:
        """
        ファイルをダウンロードしてローカルに保存する
        
        Args:
            file_id (str): ダウンロードするファイルのID
        
        Returns:
            str: 保存されたファイルのパス
        """
        data = self.openai_client.files.content(file_id)
        data_bytes = data.read()

        # ファイルの内容からMIMEタイプを取得
        mime_type = magic.from_buffer(data_bytes, mime=True)

        # MIMEタイプから拡張子を取得
        extension = mimetypes.guess_extension(mime_type)

        # 拡張子が取得できない場合はデフォルトの拡張子を使用
        if not extension:
            extension = ""

        file_name = f"./files/{file_id}{extension}"
        with open(file_name, "wb") as file:
            file.write(data_bytes)

        return file_name

