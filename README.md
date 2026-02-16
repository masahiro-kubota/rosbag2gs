# gsplat-workspace

[gsplat](https://github.com/nerfstudio-project/gsplat) v1.5.3 の examples を動かすための作業環境。

## 前提条件

- Python 3.10
- CUDA 12.x (toolkit + driver)
- [uv](https://docs.astral.sh/uv/)

## セットアップ

### 1. データセットのダウンロード

Tanks & Temples + Deep Blending データセットを https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ からダウンロード:

```bash
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip -d data/
```

含まれるシーン:

| カテゴリ | シーン |
|---|---|
| tandt (Tanks & Temples) | train, truck |
| db (Deep Blending) | drjohnson, playroom |

### 2. 依存パッケージのインストール

```bash
# メインの依存をインストール
uv sync

# CUDA 拡張を含むパッケージを別途インストール（環境の torch を使ってビルドするため --no-build-isolation が必要）
uv pip install wheel
uv pip install --no-build-isolation \
  "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
  "fused-bilagrid @ git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16" \
  "ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.0"
```

### 3. 動作確認

```bash
uv run python -c "import gsplat; print(gsplat.__version__)"
uv run python examples/simple_trainer.py mcmc --help
```

## 学習

```bash
# 基本的な学習 (MCMC)
uv run python examples/simple_trainer.py mcmc \
  --data_dir data/tandt/truck \
  --data_factor 4

# 3DGUT を有効にした学習
uv run python examples/simple_trainer.py mcmc \
  --data_dir data/tandt/truck \
  --data_factor 4 \
  --with_ut --with_eval3d
```

## ビューワー

```bash
# 学習済みモデルを表示
uv run python examples/simple_viewer.py \
  --ckpt results/truck/ckpt_29999_rank0.pt
```

## 注意事項

- `examples/` は gsplat リポジトリの v1.5.3 タグから抽出したもの
- gsplat 本体は [wheel index](https://docs.gsplat.studio/whl/pt21cu121) からプリビルド版を使用（ソースビルド不要）
- CUDA 拡張パッケージ (fused-ssim, fused-bilagrid, ppisp) は `uv sync` では正しくビルドできないため `--no-build-isolation` で別途インストールが必要

## 今後の改善項目

- **CUDA 拡張パッケージを `uv sync` に統合する**: 現在 fused-ssim, fused-bilagrid, ppisp は `uv pip install --no-build-isolation` で別途インストールしており、`uv.lock` で管理されない。`[tool.uv.extra-build-dependencies]` で `torch==2.1.2+cu121` を明示するか、pytorch-cu121 index の `explicit = true` を外すことで `uv sync` 一発にできる可能性がある
- **examples を最新の main ブランチに追従させる**: 現在は v1.5.3 タグの examples を使用。main ブランチには `color_correct` 等の新機能が追加されているが、対応する gsplat wheel がまだリリースされていない。次の gsplat リリース後に更新する
