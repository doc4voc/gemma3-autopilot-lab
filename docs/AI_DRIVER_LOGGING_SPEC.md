# AI Driver Logging Spec

## Objective
This file defines the minimum data required to debug target capture quality, wall collisions, and exploration efficiency.

## Primary goals
- Keep the LLM as the main decision maker.
- Preserve safety by blocking obviously unsafe sectors.
- Make post-run analysis reproducible from saved logs.

## Required log streams
1. `telemetry` (high frequency, sampled every UI tick)
2. `decisionLog` (one record per LLM decision cycle)

## Telemetry required fields
- Core motion:
  - `time`, `worldX`, `worldZ`, `headingDeg`, `speed`, `throttle`, `steering`
- Target state:
  - `distanceToTarget`, `targetAngle`, `targetHits`, `targetHitCount`, `targetContact`
- Obstacle state:
  - `front`, `leftDiag`, `rightDiag`, `left`, `right`, `back`, `backLeft`, `backRight`, `minObstacleDist`
- AI state:
  - `aiLatencyMs`, `aiStrategyMode`, `aiStrategyTransition`, `aiStrategySector`, `aiStrategyConfidence`
  - `aiParseMethod`, `aiParseRecovered`, `aiModel`, `aiSource`, `decisionAgeMs`
- Memory/heatmap diagnostics:
  - `memoryNoGoRatio`, `memoryRevisitRate`, `memoryCurrentWeight`
  - `memorySelectedWeight`, `memorySelectedNoGo`, `memorySelectedSector`, `memorySelectionReason`
  - `memoryCandidateCount`, `memorySafeCandidateCount`, `memoryNoGoCandidateCount`

## DecisionLog required fields
- Input snapshots:
  - `sensor_snapshot`, `sensor_latest`
- Memory context:
  - `exploration`
  - `heatmap_diag`:
    - `currentCellWeight`, `selectedCellWeight`, `selectedCellNoGo`, `selectedCellNoGoReasons`
    - `topCandidates`, `topSafeCandidates`, `sectorSafety`
    - `noGoRatio`, `revisitRate`
    - `targetBearingDeg`, `targetDistance`, `targetHitCount`
- LLM trace:
  - `ai_prompt`, `ai_raw`, `ai_parsed`
  - `ai_parse_method`, `ai_parse_recovered`, `ai_model`
- Safety trace:
  - `safety_guard`
- Output:
  - `controls`

## Analysis checks (minimum)
- Safety:
  - High `memorySelectedNoGoRate` means the system keeps choosing dangerous cells.
  - High `safetyRiskRatio` plus high `memoryNoGoRatio` means safe sectors are not selected effectively.
- Targeting:
  - Low `TargetLock@Contact` with non-zero contact means lock transition is weak.
  - Negative `intentionality` with frequent `EXPLORE` means over-exploration.
- Delay robustness:
  - Compare `aiLatencyMs` and `decisionAgeMs` to detect stale actions.

## Workflow
1. Run autopilot and record telemetry.
2. Save JSON logs (`SAVE LOGS`).
3. Export HTML report.
4. Review report metrics plus `heatmap_diag` and `safety_guard` in JSON.
5. Tune one variable group at a time (strategy weights, safety thresholds, or prompt contract).

