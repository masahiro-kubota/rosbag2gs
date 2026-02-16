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

tandt_db データセットには `images_4/` 等のダウンサンプル済み画像が含まれないため `--data_factor 1`（フル解像度）を使用する。

```bash
# 基本的な学習 (MCMC)
uv run python examples/simple_trainer.py mcmc \
  --data_dir data/tandt/truck \
  --data_factor 1 \
  --disable_viewer

# 3DGUT を有効にした学習（カメラ歪み・ローリングシャッター対応）
# tandt_db はピンホールカメラのため 3DGUT の恩恵は少ないが、動作確認として使える
uv run python examples/simple_trainer.py mcmc \
  --data_dir data/tandt/truck \
  --data_factor 1 \
  --with_ut --with_eval3d \
  --disable_viewer
```

> **Note:** `--disable_viewer` を省略すると viser ビューワーサーバー (port 8080) が起動し、学習完了後もプロセスが終了しない。ブラウザで `http://localhost:8080` にアクセスすると学習中のリアルタイムビューワーとして使えるが、バックグラウンド実行時は `--disable_viewer` を付けること。

## ROS2 MCAP バッグからの 3DGS

ROS2 の MCAP バッグファイルから画像とカメラポーズを抽出し、COLMAP 形式に変換して 3DGS 学習を行う。

### 前提

- MCAP バッグに以下のトピックが必要:
  - カメラ画像 (CompressedImage)
  - カメラ内部パラメータ (CameraInfo)
  - 車両ポーズ (Odometry)
  - `tf_static` (TFMessage) — base_link からカメラ光学フレームまでの変換チェーン
- `tf_static` が別のバッグに入っている場合は `--tf-mcap` で指定
- トピック名は YAML 設定ファイルで指定（`config/` にサンプルあり）

### 変換

```bash
# 単一カメラ（YAML 設定ファイルを使う場合）
python3 scripts/mcap_to_colmap.py \
  --config config/front.yaml \
  --mcap path/to/bag.mcap \
  --tf-mcap path/to/tf_static.mcap \
  --output data/front

# 単一カメラ（--camera ショートカット）
# /sensing/camera/<name>/image_raw/compressed 等のトピック命名規則に従う場合
python3 scripts/mcap_to_colmap.py \
  --camera front \
  --mcap path/to/bag.mcap \
  --output data/front

# 複数カメラ（複数台を1つのデータセットに統合）
python3 scripts/mcap_to_colmap.py \
  --camera front left right rear \
  --mcap path/to/bag.mcap \
  --tf-mcap path/to/tf_static.mcap \
  --output data/multi_cam
```

複数カメラ指定時の動作:
- 各カメラに異なる `camera_id` を割り当て、`cameras.bin` に複数カメラとして記録
- 画像ファイル名にカメラ名プレフィックスを付与（例: `front_0000001234.567890.jpg`）して衝突回避
- シーン中心は全カメラの位置から計算

設定ファイルの例 (`config/camera1.yaml`):

```yaml
camera:
  image_topic: /camera/front/image
  camera_info_topic: /camera_info
pose:
  topic: /vehicle/odometry
tf_static:
  # base_link からカメラ光学フレームまでの TF チェーンを指定
  chain:
    - base_link
    - camera_mount_link
    - front_camera_link
```

処理内容:
1. `camera_info` からカメラ内部パラメータ (fx, fy, cx, cy) を取得
2. Odometry トピックから車両ポーズを取得し、画像タイムスタンプに補間
3. `tf_static` のチェーンを適用してカメラの world ポーズを計算
4. 座標をシーン中心にシフト（地図座標系の巨大な絶対値を回避）
5. COLMAP 形式 (`cameras.bin`, `images.bin`, `points3D.bin`) として出力

### 検証

```bash
# pycolmap で読み込み確認
uv run python -c "
import pycolmap
sm = pycolmap.SceneManager('data/front/sparse/0/')
sm.load()
print(f'Cameras: {len(sm.cameras)}, Images: {len(sm.images)}, Points3D: {len(sm.points3D)}')
"
```

### 学習

```bash
# 単一カメラ
uv run python examples/simple_trainer.py mcmc \
  --data_dir data/front \
  --data_factor 1 \
  --result_dir results/front \
  --disable_viewer

# 複数カメラ統合
uv run python examples/simple_trainer.py mcmc \
  --data_dir data/multi_cam \
  --data_factor 1 \
  --result_dir results/multi_cam \
  --disable_viewer \
  --disable_video
```

> **Note:** 複数カメラ統合時は `--disable_video` を付けること。トラジェクトリ動画の生成は全カメラのポーズを補間するため、異なる方向のカメラ間でワープが発生し、レンダリングに非常に時間がかかる。

## ビューワー

```bash
# 学習済みモデルを表示
uv run python examples/simple_viewer.py \
  --ckpt results/truck/ckpt_29999_rank0.pt
```

## PLY エクスポート

学習済みチェックポイントから PLY ファイルを出力する。出力した PLY は CloudCompare や他の 3DGS ビューワーで閲覧できる。

```bash
uv run python export_ply.py \
  --ckpt results/front/ckpts/ckpt_29999_rank0.pt \
  --output results/front/ply/output.ply
```

## 注意事項

- `examples/` は gsplat リポジトリの v1.5.3 タグから抽出したもの
- gsplat 本体は [wheel index](https://docs.gsplat.studio/whl/pt21cu121) からプリビルド版を使用（ソースビルド不要）
- CUDA 拡張パッケージ (fused-ssim, fused-bilagrid, ppisp) は `uv sync` では正しくビルドできないため `--no-build-isolation` で別途インストールが必要

## 今後の改善項目

- **CUDA 拡張パッケージを `uv sync` に統合する**: 現在 fused-ssim, fused-bilagrid, ppisp は `uv pip install --no-build-isolation` で別途インストールしており、`uv.lock` で管理されない。`[tool.uv.extra-build-dependencies]` で `torch==2.1.2+cu121` を明示するか、pytorch-cu121 index の `explicit = true` を外すことで `uv sync` 一発にできる可能性がある
- **examples を最新の main ブランチに追従させる**: 現在は v1.5.3 タグの examples を使用。main ブランチには `color_correct` 等の新機能が追加されているが、対応する gsplat wheel がまだリリースされていない。次の gsplat リリース後に更新する
