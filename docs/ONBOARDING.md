# LLMオンボーディングサマリー

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** CAPA（Depth Completion as Parameter-Efficient Test-Time Adaptation）。単眼深度推定モデルを、疎な深度観測に合わせてテスト時適応する研究コード。
- **最終成果物:** RGB画像またはシーンと疎な深度条件から、適応後の密な深度推定結果（`.pt`）と可視化画像を生成する実行環境・コード。
- **ビジネス背景・価値:** 大規模な再学習を行わず、LoRAまたはVisual Prompt Tuning（VPT）で推論時にモデルを軽量適応する。基盤モデルごとの環境分離が必要。
- **現時点の進捗サマリ:** MoGe-2用uv環境 `venv/moge-2_env` を構築済み。MoGe-2 VPTでサンプル `corridor_02.pt` を実行し、予測・可視化・評価指標の出力を確認済み。

## 2. クリティカルな要求・制約

- Pythonは`>=3.10`を使用する。システムPythonではなく、uvで管理した環境を使用する。
- 基盤モデル間の依存関係に衝突があるため、VGGT・MoGe-2・UniDepth-v2は環境を分離する。
- MoGe-2環境ではCUDA対応PyTorchを使用する。現在はPyTorch `2.6.0+cu124`。
- CAPA本体および`third_party/`内のコード・モデルライセンスを変更・再配布する際は、各LICENSEの条件を確認する。
- `assets/vpt_init_tokens/*.pt`はGit LFS管理対象。ポインタファイルのままでは実行できないため、実体の取得状態を確認する。
- サンプル実行では入力テンソルのキー（`rgb_nv3hw`、`depth_condition_nvhw`、`mask_condition_nvhw`）を壊さない。

## 3. 参照すべき合意済み資料

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義書 | `README.md` | プロジェクト概要、セットアップ、サンプル実行、ライセンスの一次資料 |
| 要件定義書 | `pyproject.toml` | CAPAのPython要件と依存ライブラリ |
| 環境構築手順 | `scripts/setup_env.sh` | 基盤モデルごとの標準環境構築方針。uv環境では同等手順を`uv pip`で実施する |
| 実行入口 | `run.py` | 設定読込、入力処理、推論、評価、成果物保存 |
| プロトコル実装 | `capa/protocol.py` | LoRA/VPT注入、テスト時最適化、深度整合、推論の中核 |
| モデル設定 | `config/moge_vpt.yaml` | MoGe-2 VPTのチェックポイント、学習率、ステップ数、初期トークン設定 |
| テスト資産 | `input/sample_data/` | サンプル`.pt`入力。大容量シーンは容量を確認してから取得する |
| 実行成果物 | `output/moge-2_sample/` | サンプル予測、深度可視化、RGBとの比較画像 |
| 既知課題リスト | TBD | 現時点で正式な課題管理ファイルは未整備 |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク

- `capa/`、`config/`、`run.py`の調査・小規模な修正・テスト。
- uv環境の再構築、依存関係の検証、サンプル入力による動作確認。
- 出力テンソル、評価指標、可視化成果物の確認と記録。
- READMEや本オンボーディング文書の更新。

### 任せないタスク

- ユーザーの明示的な依頼なしに、モデル構造・最適化設定・ライセンス表記を大きく変更しない。
- `third_party/`のコードを、互換性確認なしに一括更新しない。
- 数GB級のサンプルデータを、容量確認や用途確認なしに全量取得しない。
- 外部サービスへの公開、GitHubへのpush、ライセンス変更を自動実行しない。

## 5. インタラクション方針

- **回答スタイル:** 日本語。結論を先に示し、必要なコマンドと検証結果を簡潔に記載する。
- **回答手順:** 前提・制約 → 実施内容 → 検証結果 → 残課題の順。
- **禁止事項・注意:** 未検証の実行結果を断定しない。依存関係・GPU・データ容量など、結果に影響する前提を省略しない。
- **秘匿情報の扱い:** HFトークン、SSH鍵、認証情報をファイルやログに記録しない。必要な認証情報は環境変数で扱う。

## 6. 試行タスク（オンボーディング演習）

1. `README.md`と`pyproject.toml`を読み、MoGe-2環境のPython要件・主要依存関係・実行コマンドを説明する。
2. `venv/moge-2_env`を有効化し、`torch.cuda.is_available()`とPyTorchのCUDAビルドを確認する。
3. `input/sample_data/ibims1_max-depth-5m_noise_10pct/corridor_02.pt`を`config/moge_vpt.yaml`で実行し、`output/moge-2_sample/`の3成果物と評価ログを確認する。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** 実行したコマンド、対象環境、入力データ、結果、未解決事項を具体的に記録する。
- **TBDの扱い:** 未確認の事項は推測で埋めずTBDとし、確認方法または確認担当を追記する。
- **レビュー/承認フロー:** `third_party/`、モデル設定、依存関係、ライセンスに関わる変更は、実行検証後に人間レビューを受ける。
- **その他の運用ルール:** 成果物は`output/`に保存し、入力データは`input/`に保存する。大容量データは必要なファイルだけ取得する。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:** `/home/kasm-user/Desktop/capa`
- **代表的なコマンド:**

  ```bash
  cd /home/kasm-user/Desktop/capa
  source venv/moge-2_env/bin/activate
  python run.py --config config/moge_vpt.yaml \
    --input input/sample_data/ibims1_max-depth-5m_noise_10pct/corridor_02.pt \
    --output output/moge-2_sample --save-vis --verbose
  ```

- **依存ライブラリ:** PyTorch/CUDA、torchvision、peft、omegaconf、matplotlib、numpy、Pillow、OpenCV、scipy、trimesh、gradio、huggingface_hub、utils3dなど。
- **モデルチェックポイント:** `Ruicheng/moge-2-vitl-normal`（初回実行時にHugging Face Hubから取得）。
- **サンプルデータ配布元:** <https://share.phys.ethz.ch/~pf/bingkedata/capa/sample_data/>
- **連絡先/責任者:** TBD

> この文書はバージョン管理し、環境・実行手順・既知課題に変更があった場合は更新する。
