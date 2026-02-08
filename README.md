# Gemma 3 AutoPilot Lab (GemmaCar / AutoDrive)

ブラウザ上の簡易 3D ドライビング環境（React + Vite + three.js）で、Ollama 経由の LLM（例: `gemma3:4b` / `gemma3:12b`）に**運転意思決定（throttle/steering）**をさせる研究・実験用プロトタイプです。

このリポジトリは「LLM 主導の制御」を中心に、**安全ゲート（preflight / runtime guard）**、**観測可能性（telemetry / decision log）**、**再現性のある AB 実験**、**レポート生成（HTML/PDF）**を揃えることを主目的にしています。

> 注意: これは実車制御ではありません（ブラウザ内シミュレーションです）。安全上の観点からも、実世界への転用を意図していません。

## Docs（研究レポート / 仕様 / 関連調査）

- 研究レポート（HTML）: `docs/gemma3_autodrive_research_report_paper_grade_2026-02-07.html`
- 研究レポート（PDF）: `docs/Gemma 3 AutoPilot Research Report (Paper-grade) - 2026-02-07.pdf`
- 関連研究サーベイ: `docs/gemma3_autodrive_related_work_survey_2026-02-07.html`
- 実装仕様（現行コード準拠）: `docs/gemma3_autodrive_implementation_spec_2026-02-07.md`
- ログ仕様（最小要件）: `docs/AI_DRIVER_LOGGING_SPEC.md`

## Research focus（この実装で検証したいこと）

この実装は、概ね次の問いを扱えるように設計しています（詳細は `docs/` を参照）。

- **モデル差**: 小型/大型モデルで、ターゲット捕捉・安全性・遅延（staleness）・探索挙動がどう変わるか
- **センサー条件差**: 固定レンジ vs 動的レンジなど、入力条件の違いが行動品質にどう影響するか
- **ガード設計**: LLM の不安定さ（JSON破損、逸脱、遅延、危険操作）に対し、どの安全ゲートが効くか
- **観測と再現性**: drive log と telemetry から、後で比較・追試できる形に落とせるか

## System overview（ざっくり）

1. **センサー**（8方向 ray + 目標手がかり）と状態（速度/方位など）をスナップショット化  
2. **LLM にプロンプト** → **JSON（行動プラン + 理由）**を要求  
3. **JSON を厳格にパース**（修復/再試行あり）し、ガードで逸脱を抑制  
4. throttle/steering を適用しつつ、telemetry/decision log を保存  
5. 走行後に **メトリクス集計 + HTML レポート生成**（必要に応じて AI レビュー）

実装の責務分割（現行仕様より）:

- Orchestrator / 実験・状態機械: `src/App.jsx`
- 物理・センサー: `src/components/Car.jsx`
- シーン: `src/components/GameScene.jsx`
- LLM 呼び出し・JSON整形・戦略/ヒステリシス: `src/services/ollamaService.js`
- 探索メモリ（グリッド）: `src/services/explorationMemory.js`
- 解析・レポート生成: `src/services/analysisService.js`

## Experiments（AB実験の考え方）

- モデルやセンサー条件などを「条件行列」として定義し、繰り返し実行して比較できる形を目指しています。
- 実験設定にはスキーマID（例: `gemma-autodrive-experiment-config`）があり、設定/出力の整合性を取りやすくしています（詳細は `docs/gemma3_autodrive_implementation_spec_2026-02-07.md`）。

## Outputs & logging（何が取れるか）

最低限、次の 2 ストリームを想定しています（詳細は `docs/AI_DRIVER_LOGGING_SPEC.md`）。

- `telemetry`: UI tick 単位の時系列（速度、方位、障害物距離、ターゲット距離、AI遅延など）
- `decisionLog`: LLM 1 回の意思決定ごとの記録（入力、プロンプト、raw、parse、ガード、出力）

また、preflight 失敗時のゲートログや、実験サマリ、HTML レポートなどの成果物をダウンロードできる設計になっています（命名や一覧は実装仕様に記載）。

## Requirements

- Node.js（推奨: 20+）
- Ollama（既定: `http://localhost:11434`）

## Quick start

```bash
npm install
npm run dev
```

### Ollama models（例）

```bash
ollama pull gemma3:4b
ollama pull gemma3:12b
```

## Scripts

- `npm run dev` 開発サーバ
- `npm run build` 本番ビルド（`dist/`）
- `npm run preview` ビルド成果物のプレビュー
- `npm run lint` ESLint

## データ（ログ / 実験結果）の扱い

ログや実験結果は大容量になりやすいため、Git 管理しない方針です（`.gitignore` により除外）。

- `thelatestlogdata/`
- `auto_experiment_results/`
- `OLDDATA/`
- `youtube/`
- `preflight_failure_logs/`

共有が必要な場合は Git LFS / Releases 添付 / 外部ストレージなどを推奨します。

## How to cite（メモ）

研究メモや共有で参照する場合は、`docs/` 内の “Paper-grade” レポート（`2026-02-07`）と、このリポジトリのコミットSHAを併記すると追跡しやすいです。

## License

MIT License（`LICENSE`）

