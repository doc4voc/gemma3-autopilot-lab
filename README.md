# GemmaCar / Gemma 3 AutoDrive

ブラウザ上の簡易3Dドライビング環境（React + Vite + three.js）で、Ollama 経由の LLM（例: `gemma3:4b` / `gemma3:12b`）に運転意思決定をさせる研究・実験用プロトタイプです。

## Docs（研究レポート / 仕様）

- `docs/gemma3_autodrive_research_report_paper_grade_2026-02-07.html`
- `docs/Gemma 3 AutoPilot Research Report (Paper-grade) - 2026-02-07.pdf`
- `docs/gemma3_autodrive_related_work_survey_2026-02-07.html`
- `docs/gemma3_autodrive_implementation_spec_2026-02-07.md`
- `docs/AI_DRIVER_LOGGING_SPEC.md`

## Requirements

- Node.js（推奨: 20+）
- Ollama（既定: `http://localhost:11434`）

## Quick start

```bash
npm install
npm run dev
```

起動後に表示されるローカルURLをブラウザで開きます。

### Ollama models

アプリ側の既定候補は `gemma3:4b` / `gemma3:12b` です（ローカル環境のタグに合わせて準備してください）。

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

大容量になりやすいログや実験結果は Git 管理しない方針です（`.gitignore` により除外）。

- `thelatestlogdata/`
- `auto_experiment_results/`
- `OLDDATA/`
- `youtube/`
- `preflight_failure_logs/`

必要なら Git LFS や Release 添付、あるいは別ストレージで共有してください。

## Status

研究・実験用のプロトタイプです。UI/ログ形式/意思決定プロンプト等は今後変わる可能性があります。

