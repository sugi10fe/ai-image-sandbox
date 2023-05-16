# ai-image-sandbox

## 前提条件

* ハードウェア
  * AMDのCPU
  * NVidiaのビデオカード
* OS
  * windows 11
* ソフトウェア
  * [python](https://www.python.org/)
  * [pyenv-win](https://github.com/pyenv-win/pyenv-win)
  * [poetry](https://python-poetry.org/docs/)
  * [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

## 構築手順

```powershell
# 使用するverのpythonをDL
pyenv install $(cat ./.python-version)

# pythonのバージョンを強制的に.python-versionに合わせる
poetry env use $(pyenv which python)

# 仮想環境に入る
poetry shell

# 依存のインストール
poetry install
```

## 生成スクリプト

pythonで実行可能 / --helpでオプションを確認可能

* `scripts/fromtxt.py`

## 注意事項

modelsディレクトリをignoreしていますが、必ずしもここにモデルファイルを置く必要はありません
