# Vera Demo CLI — 実証済み効果を GPT-2 で体感する

個人規模で実証済みの立体十字（Stereo-Cross）機能を、**実際に触って感じる**ための CLI です。  
スケールアップ（9B 接合など）は計算資源が足りないため、末尾の **Call for collaborators** を参照してください。

## セットアップ

```bash
cd Vera
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

チェックポイント（各 ~0.5GB）は git に含めていません。実験マシン上の成果物を指します:

```bash
# 例: このリポジトリ外に置いた成果物
export VERA_ARTIFACTS=/path/to/verantyx-cli/experiments/stereo_cross_activation

# 必要なファイル
#   bases_cache_soft_distill.npz
#   matryoshka_student.pt          (or kl_distill_student_r256.pt)
#   student_b_distilgpt2.pt        (join 用)
#   means_b_distilgpt2.npz         (join 用)
```

## コマンド

```bash
# 成果物の有無 + 実証サマリ（GPU 不要）
python -m demo status

# 一通り体感（Matryoshka → Memory → Join → Hub → Vision）
python -m demo tour

# 個別
python -m demo matryoshka --prompt "The capital of France is"
python -m demo memory
python -m demo join --prompt "Once upon a time in a small village"
python -m demo hub --prompt "The scientific method requires"
python -m demo vision --open          # 可視化 HTML を開く
```

## 体感できること

| デモ | 対応する実証 | 感じ方 |
|------|-------------|--------|
| `matryoshka` | MATRYOSHKA_VIABLE | ランクを下げても文法は残り、上げると品質が単調に戻る |
| `memory` | MEMORY_VIABLE | 実メモリ再注入が、ランダム座標より答えトークンを押し上げる |
| `join` | JOIN_VIABLE | GPT-2 の座標を DistilGPT2 に渡すと流畅、ランダムは崩壊 |
| `hub` | PARTS_LIKE | ハブ次元を消すと、ランダム次元を消すより壊れる |
| `vision` | 全体 | 実証マップと、スケールアップで開きうる設計空間 |

## 可視化・協力募集

- [`vision.html`](./vision.html) — ブラウザ用ボード
- [`CALL_FOR_COLLABORATORS.md`](./CALL_FOR_COLLABORATORS.md) — 計算資源のある協力者向け
