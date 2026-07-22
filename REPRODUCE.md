# 再現手順（Reproduction Guide）

このリポジトリには、立体十字構造体（Stereo-Cross Container）検証の**ソースコード・レポート・生結果 JSON・実行ログ**がすべて含まれます。巨大な蒸留チェックポイント（`.pt`、各約475MB）は GitHub 制限のため含めず、下記コマンドで再生成できます。

## 環境

```bash
git clone https://github.com/Ag3497120/Vera.git
cd Vera
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- macOS Apple Silicon では PyTorch が MPS を自動利用します（CPU でも可、遅い）
- 初回実行時に Hugging Face からモデル／wikitext-2 をダウンロードします（ネット必須）

## ディレクトリ

| パス | 内容 |
|------|------|
| `experiments/stereo_cross_bridge/` | Part I: 重み空間（Qwen1.5-0.5B-Chat） |
| `experiments/stereo_cross_activation/` | Part II–III: 活性空間・蒸留・反応地図・圧縮（GPT-2） |
| `README.md` | 仮説→結果の全経緯（年代記） |
| `archive/` | `experiments/` のスナップショット写し（同内容） |

各実験ディレクトリの `*.md` がプロトコル・事前登録フォーク・判定です。`results_*.json` が生データです。

---

## Part I — 重み空間（Qwen1.5-0.5B-Chat）

作業ディレクトリ: `experiments/stereo_cross_bridge/`

モデルは HF キャッシュの `Qwen/Qwen1.5-0.5B-Chat` を自動検出します（未取得なら先に `huggingface-cli download Qwen/Qwen1.5-0.5B-Chat`）。

```bash
cd experiments/stereo_cross_bridge

# I-1 共有ブリッジ vs SVD（例: down_proj 浅層）
python3 shared_bridge_vs_svd.py --module down_proj --layers 5 --layer-start 0 --ranks 32,64,128

# 深さ帯
python3 shared_bridge_vs_svd.py --module down_proj --layers 5 --layer-start 19 --ranks 32,64,128

# 他モジュール
python3 shared_bridge_vs_svd.py --module gate_proj --layers 5 --ranks 32,64,128
python3 shared_bridge_vs_svd.py --module up_proj --layers 5 --ranks 32,64,128
python3 shared_bridge_vs_svd.py --module o_proj --layers 5 --ranks 32,64,128
python3 shared_bridge_vs_svd.py --module q_proj --layers 5 --ranks 32,64,128
python3 shared_bridge_vs_svd.py --module v_proj --layers 5 --ranks 32,64,128
python3 shared_bridge_vs_svd.py --module k_proj --layers 5 --ranks 32,64,128

# Hybrid / ヘッド分解 / Procrustes / 訓練込み
python3 hybrid_shared_plus_residual.py
python3 shared_bridge_vs_svd_per_head.py
python3 procrustes_layer_align.py
python3 trained_bridge_gd.py
```

対応レポート: `SUMMARY.md`, `MODULE_COMPARE.md`, `DEPTH_COMPARE.md`, `ENTRANCE_COMPARE.md`, `ATTN_COMPARE.md`, `PER_HEAD_COMPARE.md`, `PROCRUSTES_COMPARE.md`, `TRAINED_BRIDGE.md`

---

## Part II — 活性空間・容れ物（GPT-2 small）

作業ディレクトリ: `experiments/stereo_cross_activation/`

```bash
cd experiments/stereo_cross_activation

# II-1 活性共有プローブ + GPT-2 重み再現
python3 activation_shared_probe.py
python3 gpt2_weight_bridge.py

# II-2 マトリョーシカ機能パッチ
python3 matryoshka_patch.py

# II-3 ソフト射影 + LM ボトルネック蒸留
python3 soft_and_distill.py

# II-4 KL 蒸留 r=128（チェックポイント kl_distill_student_r128.pt を生成）
python3 kl_distill_bottleneck.py

# II-5 KL 蒸留 r=256 warm-start（r128 .pt が必要 → kl_distill_student_r256.pt）
python3 kl_distill_r256.py
```

対応レポート: `ACTIVATION_PROBE.md`, `MATRYOSHKA.md`, `SOFT_DISTILL.md`, `KL_DISTILL.md`, `KL_DISTILL_R256.md`

### チェックポイント再生成の目安

| 成果物 | 生成スクリプト | おおよその時間 (MPS) |
|--------|----------------|----------------------|
| `bases_cache_soft_distill.npz` | `soft_and_distill.py` / `matryoshka_patch.py` | 数分 |
| `kl_distill_student_r128.pt` | `kl_distill_bottleneck.py` | ~2 時間 |
| `kl_distill_student_r256.pt` | `kl_distill_r256.py`（r128 必須） | ~3 時間 |

リポジトリ同梱の `bases_cache_soft_distill.npz` / `mid_means_weight_compress.npz` は再計算を省略できます。`.pt` のみ手元で再生成してください。

---

## Part III — 活用ロードマップ

```bash
cd experiments/stereo_cross_activation

# Step 1 反応地図（r256 生徒が必要）
python3 response_map.py

# Step 4 重み圧縮（実行中／治癒蒸留含む）
python3 weight_compress.py

# Step 3 Matryoshka 蒸留 — 未実装（README ロードマップ参照）
# Step 5 異モデル接合 — 未実装
```

対応レポート: `RESPONSE_MAP.md`（Step 1 完了）。Step 4 は `results_weight_compress.json` / `weight_compress_run.log` を参照（治癒完了後に `WEIGHT_COMPRESS.md` を追記予定）。

---

## 結果の照合

各スクリプトはカレントディレクトリに `results_*.json` を書き出します。同梱の JSON と数値を比較してください。乱数・HF 版差・MPS 非決定性で末尾桁は揺れ得ますが、フォーク判定（COMPRESS_ONLY / IDENTITY_AT_256 / PARTS_LIKE 等）は再現範囲内です。

評価セットの標準: wikitext-2 test、40×256 = 10,240 トークン（活性系）。ベースライン GPT-2 small ppl ≈ 54.47。

---

## ライセンス・出典

- 実験コード・レポート: このリポジトリの内容（Vera）
- モデル: [Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat), [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- データ: [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext)
