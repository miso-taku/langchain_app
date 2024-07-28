# PDFを読み込み、質問に対する回答を返すStreamlitアプリケーション
# ライブラリのインポート
import streamlit as st

def init_page() -> None:
    """
    Streamlitページの初期化を行います。
    """
    st.set_page_config(
        page_title="PDF QA",
        page_icon="📄",
    )

def main() -> None:
    """
    メイン関数。Streamlitページを起動します。
    """
    init_page()
    st.title("PDF QA")
    st.write("PDFを読み込み、質問に対する回答を返すStreamlitアプリケーション")

    st.sidebar.success("メニュー")
    st.markdown(
        """
        - このアプリケーションは、PDFファイルを読み込み、質問に対する回答を返します。
        - まずはメニューからPDFファイルをアップロードしてください。
        - PDFをアップロードしたら、質問を入力して回答を取得できます。
        """
    )

if __name__ == "__main__":
    main()