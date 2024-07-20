# ベースイメージ
FROM python:3.12-slim

# 作業ディレクトリの設定
WORKDIR /app

# Poetryのインストール
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get clean

# Poetryのパスを環境変数に追加
ENV PATH="/root/.local/bin:$PATH"

# Poetry設定の無効化（仮想環境作成を無効化）
RUN poetry config virtualenvs.create false

# Poetryによるパッケージのインストール
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# アプリケーションのコピー
COPY . .

# StreamlitとGradioの両方を起動するスクリプトをコピー
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ポートの公開
EXPOSE 8501 7860

# アプリケーションの起動
CMD ["/app/start.sh"]
