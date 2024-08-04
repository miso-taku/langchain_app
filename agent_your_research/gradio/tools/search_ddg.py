from itertools import islice
from typing import List, Dict
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field


"""
Sample Response of DuckDuckGo python library
--------------------------------------------
[
    {
        'title': '日程・結果｜Fifa 女子ワールドカップ オーストラリア&ニュージーランド 2023｜なでしこジャパン｜日本代表｜Jfa｜日本サッカー協会',
        'href': 'https://www.jfa.jp/nadeshikojapan/womensworldcup2023/schedule_result/',
        'body': '日程・結果｜FIFA 女子ワールドカップ オーストラリア&ニュージーランド 2023｜なでしこジャパン｜日本代表｜JFA｜日本サッカー協会. FIFA 女子ワールドカップ. オーストラリア&ニュージーランド 2023.'
    }, ...
]
"""

class SearchDDGInput(BaseModel):
    """
    検索入力モデル。
    
    Attributes:
        query (str): 検索したいキーワード。
    """
    query: str = Field(description="検索したいキーワードを入力してください")

@tool(args_schema=SearchDDGInput)
def search_duckduckgo(query: str, max_result_num: int = 5) -> List[Dict[str, str]]:
    """
    DuckDuckGo検索を実行するためのツールです。
    検索したいキーワードを入力して使用してください。
    検索結果の各ページのタイトル、スニペット（説明文）、URLが返されます。
    このツールから得られる情報は非常に簡素化されており、時には古い情報の場合もあります。

    必要な情報が見つからない場合は、必ず `WEB Page Fetcher` ツールを使用して各ページの内容を確認してください。
    文脈に応じて最も適切な言語を使用してください（ユーザーの言語と同じである必要はありません）。
    例えば、プログラミング関連の質問では、英語で検索するのが最適です。

    Parameters
    ----------
    query : str
        検索したいキーワード。
    max_result_num : int, optional
        取得する検索結果の最大数（デフォルトは5）。

    Returns
    -------
    List[Dict[str, str]]
        検索結果のリスト。各結果はタイトル、スニペット、URLを含む辞書です。
    """
    ddg_search = DDGS()
    search_results = ddg_search.text(query, region='wt-wt', safesearch='off', backend="lite")
    results = [
        {
            "title": result.get('title', ""),
            "snippet": result.get('body', ""),
            "url": result.get('href', "")
        }
        for result in islice(search_results, max_result_num)
    ]
    return results
