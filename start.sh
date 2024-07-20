#!/bin/bash

# Streamlitの起動
streamlit run streamlit_app.py --server.port 8501 &

# Gradioの起動
python gradio_app.py &

# フォアグラウンドでtailを実行してコンテナを終了させない
tail -f /dev/null
