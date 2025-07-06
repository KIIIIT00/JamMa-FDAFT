# ベースイメージとしてNVIDIA CUDA 11.8開発環境を使用
# Python 3.8が標準で利用可能なUbuntu 20.04 (Focal Fossa) をベースにします。
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04

# 環境変数を設定し、Pythonの出力がバッファリングされないようにします。
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 必要なシステムパッケージをインストールします。
# - git: ソースコード管理のため
# - build-essential: Pythonパッケージのコンパイルに必要なツールチェーン
# - python3.8, python3.8-distutils, python3-pip: Python 3.8環境のセットアップ
# - python3.8-dev: Python C拡張のビルドに必要なヘッダーファイルなど (OpenCVの安定性向上)
# - libsm6, libxext6, libxrender1: OpenCVのGUI機能の依存関係（headless環境でも安全）
# - wget: Structured Forestsモデルのダウンロード用
# - libgl1-mesa-glx, libglib2.0-0: OpenGL関連のライブラリ（libGL.so.1エラー対策）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3.8 \
    python3.8-distutils \
    python3-pip \
    python3.8-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.8をデフォルトのpython3とpip3コマンドとして設定します。
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip 1

# 作業ディレクトリを /app に設定します。
WORKDIR /app

# 現在のJamMa-FDAFTプロジェクトの全ファイルをコンテナ内の /app にコピーします。
COPY . /app

# プロジェクトのrequirements.txtに記載されているPython依存関係をインストールします。
# --no-cache-dir オプションは、Dockerイメージのサイズを削減するためにpipのキャッシュを使用しません。
# PyTorchの公式ホイールがホストされているURLを追加し、torch/torchvisionなどを確実に見つけられるようにします。
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# mamba-ssmをインストールします。これがtritonを依存関係としてインストールする可能性があります。
RUN pip install triton==2.0.0
RUN pip install mamba-ssm==2.0.3

# プロジェクトのsetup.pyスクリプトを実行し、追加のセットアップを行います。
# 'full'モードは、開発に必要なすべての依存関係をインストールし、
# 必要なディレクトリ構造（'data/', 'assets/', 'weight/'など）を作成し、
# Structured Forestsモデル（もし必要であれば）をダウンロードします。
# 'full'は引数ではなく'--mode'オプションの値として渡します。
# RUN python3 setup.py --mode full

# コンテナが起動した際にデフォルトでBashシェルを実行するように設定します。
# これにより、コンテナ内で手動でコマンドを実行して環境を探索できます。
# 必要に応じて、`python3 demo_jamma_fdaft.py` や `bash scripts/reproduce_train/jamma_fdaft_train.sh`
# などのコマンドに置き換えることも可能です。
CMD ["/bin/bash"]