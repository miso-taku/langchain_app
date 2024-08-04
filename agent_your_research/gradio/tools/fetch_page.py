import requests
import html2text
from readability import Document
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Any


class FetchPageInput(BaseModel):
    """
    ウェブページ取得のための入力モデル。
    
    Attributes:
        url (str): 取得するページのURL。
        page_num (int): 取得するページの番号。
    """
    url: str = Field(description="取得するページのURLを入力してください")
    page_num: int = Field(0, description="取得するページの番号を入力してください")


@tool(args_schema=FetchPageInput)
def fetch_page_content(url: str, page_num: int = 0, timeout_sec: int = 10) -> Dict[str, Any]:
    """
    指定されたURLからウェブページのコンテンツを取得するツール。

    `status` と `page_content`（`title`、`content`、`has_next`インジケーター）を返します。
    statusが200でない場合は、ページの取得時にエラーが発生しています。（他のページの取得を試みてください）

    デフォルトでは、最大2,000トークンのコンテンツのみが取得されます。
    ページにさらにコンテンツがある場合、`has_next`の値はTrueになります。
    続きを読むには、同じURLで`page_num`パラメータをインクリメントして、再度入力してください。
    （ページングは0から始まるので、次のページは1です）

    1ページが長すぎる場合は、**3回以上取得しないでください**（メモリの負荷がかかるため）。

    Parameters
    ----------
    url : str
        取得するページのURL。
    page_num : int, optional
        取得するページの番号（デフォルトは0）。
    timeout_sec : int, optional
        タイムアウト時間（秒、デフォルトは10）。

    Returns
    -------
    Dict[str, Any]:
        - status (int): HTTPステータスコード。
        - page_content (Dict[str, Any]): ページコンテンツの詳細（`title`、`content`、`has_next`インジケーター）。
    """
    try:
        response = requests.get(url, timeout=timeout_sec)
        response.encoding = 'utf-8'
    except requests.exceptions.Timeout:
        return {
            "status": 500,
            "page_content": {'error_message': 'Timeout Error. Please try to fetch other pages.'}
        }

    if response.status_code != 200:
        return {
            "status": response.status_code,
            "page_content": {'error_message': 'Could not download page. Please try to fetch other pages.'}
        }

    try:
        doc = Document(response.text)
        title = doc.title()
        html_content = doc.summary()
        content = html2text.html2text(html_content)
    except Exception as e:
        return {
            "status": 500,
            "page_content": {'error_message': f'Could not parse page. Error: {str(e)}'}
        }

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name='gpt-3.5-turbo',
        chunk_size=1000,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_text(content)

    if page_num >= len(chunks):
        return {
            "status": 500,
            "page_content": {'error_message': 'Invalid page_num parameter. Please try to fetch other pages.'}
        }
    elif page_num >= 3:
        return {
            "status": 503,
            "page_content": {'error_message': "Fetching more content will overload memory. Please use current information."}
        }
    else:
        return {
            "status": 200,
            "page_content": {
                "title": title,
                "content": chunks[page_num],
                "has_next": page_num < len(chunks) - 1
            }
        }
