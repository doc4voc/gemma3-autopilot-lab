# Gemma 3 AutoDrive Detailed Implementation Specification

Document ID: `gemma3-autodrive-impl-spec`  
Version: `1.0`  
Date: `2026-02-07`  
Scope: current implementation in `src/App.jsx`, `src/components/Car.jsx`, `src/components/GameScene.jsx`, `src/services/ollamaService.js`, `src/services/explorationMemory.js`, `src/services/analysisService.js`

## 1. Purpose and background

This document defines the current system specification for the Gemma 3 AutoDrive application.

The project has entered a phase where implementation complexity and runtime error frequency are high enough that code-level behavior must be fixed against a single written spec.

Primary intent:
- Preserve LLM-led driving decisions as the core design.
- Keep safety gates and runtime guards deterministic and auditable.
- Make experiment outputs reproducible and comparable.
- Prevent regressions caused by implicit behavior and initialization-order bugs.

## 2. System goals and non-goals

### 2.1 Goals
- Blue target capture with continuous autonomous decision-making.
- Wall/obstacle safety under uncertain model output.
- Reproducible AB experiments with downloadable artifacts.
- Full observability (telemetry, drive log, preflight gate logs, summary logs).

### 2.2 Non-goals
- Hardcoded path planning replacing LLM as main driver.
- Perfect real-time performance optimization over correctness and safety.
- Closed-loop control without model reasoning trace.

## 3. Runtime architecture

### 3.1 Module responsibilities

`src/App.jsx`
- Global orchestrator.
- State machine for manual, preflight, autodrive, recovery, experiment automation.
- Data export pipeline and UI panels.

`src/components/Car.jsx`
- Physics body and ray sensors (8-direction).
- Dynamic sensor range logic.
- Collision event emission with region classification.
- Per-frame control application from analog throttle/steering.

`src/components/GameScene.jsx`
- Scene composition: track, walls, inner obstacles, compass markers, target mesh.
- Defines world orientation markers:
  - `N = -Z`
  - `S = +Z`
  - `E = +X`
  - `W = -X`

`src/services/ollamaService.js`
- Prompt composition and LLM request execution (`/api/generate`).
- Strict JSON parsing + repair fallback + retry prompt.
- Strategy normalization, hysteresis, skill executor, memory safety guard.
- Final action plan and reason contract normalization.

`src/services/explorationMemory.js`
- World grid memory map (cell-based).
- Multi-ray free-space/hazard evidence integration.
- Frontier/risky sector context and heatmap diagnostics generation.

`src/services/analysisService.js`
- Post-run metrics computation.
- AI review request for report commentary.
- HTML report generation.

## 4. Coordinate and control conventions

### 4.1 Coordinate system
- World bounds (`App`): `x,z in [-19, +19]`.
- Track outer walls (`GameScene`): approximately `+/-20`.
- Heading convention:
  - `0 deg = +Z (South)`
  - `+90 deg = +X (East)`
  - `-90 deg = -X (West)`
  - `180 deg = -Z (North)`

### 4.2 Steering/throttle sign
- Steering sign is unified as:
  - `+1 = LEFT`
  - `-1 = RIGHT`
- Throttle sign:
  - `> 0 = forward`
  - `< 0 = reverse`

### 4.3 Sensor channels
8 rays:
- `left`, `leftDiag`, `front`, `rightDiag`, `right`, `backRight`, `back`, `backLeft`

Target detection:
- Rays can visually hit target (`cyan`) but logic distance is not treated as obstacle for safety stop.
- `targetHits` is used as target cue input.

## 5. Core state model

### 5.1 High-level run phases
- `IDLE` (manual/standby)
- `PREFLIGHT`
- `AI_THINKING`
- `AI_DRIVING`
- `RECOVERY` (`REVERSING`, `TURNING`)
- `EXPERIMENT_RUNNING` (with sub-phases)

### 5.2 Important refs and state holders
- `sensorRef`: latest calibrated sensor snapshot.
- `decisionLog`: decision-level records (prompt/raw/parsed/actions/outcome).
- `telemetry`: 10Hz time-series points.
- `explorationMemoryRef`: map memory engine.
- `directionCalibrationRef`: heading and steering alignment profile.
- `preflightReport`: all startup checks.
- `preflightBlockGuardRef`: temporary retry block window.
- `experimentRunner`: automation phase and progress.

## 6. Startup preflight gate specification

### 6.1 Gate policy
- `ALL_PASS_REQUIRED`
- Autodrive start is rejected unless every check is `PASS`.

### 6.2 Checks

CAR area:
- `CAR_SENSOR_STREAM`
- `CAR_POSE_BOUNDS`
- `CAR_PHYSICS_READY`
- `DIRECTION_MAPPING`
- `CALIBRATION_READY`

AI area:
- `AI_CONNECTIVITY`
- `AI_STRATEGY_ALIGNMENT`
- `AI_MODE_SKILL_DIVERSITY`

### 6.3 Physics-ready pass thresholds
From `App` constants:
- `RUN_START_SPAWN_TOLERANCE_M = 3.0`
- `RUN_START_MAX_SPEED = 0.8`
- `RUN_START_MIN_WORLD_Y = 0.2`
- `RUN_START_MAX_WORLD_Y = 2.2`
- `RUN_START_MAX_VERTICAL_SPEED = 1.5`

`CAR_PHYSICS_READY` fails if any of:
- `finitePose` false
- `nearSpawn` false
- `speedReady` false
- `grounded` false
- `gravityStable` false

### 6.4 Preflight AI scenario probe
Scenarios:
- `LOCK_OPEN_APPROACH`
- `ESCAPE_FRONT_BLOCKED`
- `EXPLORE_NO_TARGET`

Per-scenario retries:
- `PREFLIGHT_AI_RETRY_PER_SCENARIO = 2`

Connectivity pass:
- All scenarios parse/connect successfully.

Alignment pass:
- All scenarios satisfy expected mode/skill behavior.

### 6.5 Reset readiness gate before preflight
Before running full preflight:
1. Reset session state.
2. Wait for fresh sensor tick (`sensorTickRef` increment).
3. Wait for post-reset stable sensor (2 consecutive valid samples).
4. Retry reset gate up to 2 times.

Failure outcomes:
- `SENSOR_REFRESH_TIMEOUT`
- `POST_RESET_UNSTABLE`

On failure:
- Generate synthetic preflight report.
- Auto-download preflight failure JSON.
- Apply retry cooldown.

### 6.6 Retry cooldown and fail-fast
- `PREFLIGHT_BLOCK_RETRY_COOLDOWN_MS = 5000`
- Repeated `CAR_PHYSICS_READY` ground/gravity failure contributes to experiment fail-fast.
- `EXPERIMENT_PRECHECK_BLOCK_STREAK_ABORT_THRESHOLD = 2`

## 7. AI decision contract and execution

### 7.1 LLM request model
- Endpoint: `http://localhost:11434/api/generate`
- Timeout: `15000 ms`
- Primary `num_predict`: `420`
- Retry `num_predict`: `360`

### 7.2 Prompt contract
Prompt injects:
- Sensor rays and target geometry.
- Pose, heading, speed, blocked distance.
- Exploration memory digest.
- Collision pressure digest.
- Previous outcome/reflection hints.
- Explicit coordinate convention text.

Required output:
- Strict JSON only.
- Includes `strategy`, `skill`, `reflection`, `reason`, `control`, optional `actions`.

### 7.3 Parse and recovery flow
Parsing order:
1. Strict JSON parse.
2. Trailing comma repair.
3. Single-quote repair.
4. Balanced object extraction.
5. Loose payload recovery.
6. Retry prompt for malformed/truncated output.

Fallback on parse/API error:
- Return safe hold action (`throttle=0`, `steering=0`).
- Preserve structured reason and parse method metadata.

### 7.4 Action plan constraints
Service-level constraints:
- `ACTION_PLAN_MAX_STEPS = 5`
- `ACTION_PLAN_MAX_TOTAL_SEC = 1.2`

App-level per-step clamp:
- Duration clamped to `[0.08, 1.2]` sec.
- Queue execution runs each step sequentially.

### 7.5 Think-act policy
- While waiting for LLM, vehicle is explicitly stopped.
- After decision, runtime guards can override model output.
- After plan execution, controls are set back to zero until next cycle.

## 8. Runtime safety and validation layers

Safety evaluation order:
1. Front-arc runtime safety guard (`front/leftDiag/rightDiag`).
2. Corner trap recovery trigger.
3. Direction flip cooldown guard.
4. Per-step reason validation guard.
5. Queue-time front guard on later action steps.

### 8.1 Front safety threshold
- Critical threshold for stop/escape logic: `minFrontDist < 2.5m`.

### 8.2 Direction flip cooldown
- `DIRECTION_FLIP_COOLDOWN_MS = 1000`
- Prevent immediate forward/reverse oscillation by holding throttle at 0.

### 8.3 Reason validation
Each executed step must carry a normalized reason object.
Validation checks:
- Reason present and structurally valid.
- Expected sign vs actual sign consistency.
- Optional model-source requirement.
- Forward motion blocked if front risk is high.

On validation fail:
- Step converted to short neutral hold.
- Runtime override reason injected.
- Validation stats counters updated.

## 9. Exploration memory map specification

### 9.1 Grid model
- Default cell size: `2.0`
- Maintains per-cell:
  - visits, risk EMA, stuck count
  - target hit/miss counters and absence EMA
  - obstacleHits, outerWallHits, openHits

### 9.2 Update method
Each sensor update:
- Updates current cell risk and visit.
- Integrates all 8 rays:
  - traversed cells marked as open evidence
  - hit cells marked as obstacle/outer-wall hazard
- Applies bounds clamping and risk floor logic near outer boundaries.

### 9.3 Candidate evaluation and no-go logic
No-go reasons include:
- `outsideBounds`
- `barrierBlocked`
- `obstacleDominant`
- `recentObstacleHit`
- `obstacleRepeat`
- `outerWallRepeat`
- `repeatPenaltyHigh`
- `highRisk`

Preferred sector comes from sector score aggregation over nearby candidates.

### 9.4 Visualization
Memory map view includes:
- Center cell and recent path.
- Candidate cell colors by score/risk.
- Current vehicle marker and heading.
- Compass labels consistent with world axes.

## 10. Telemetry and logs specification

### 10.1 Telemetry stream
- Sampling cadence: every `100 ms` during autodrive.
- Includes:
  - pose/motion
  - sensor rays
  - target and progress signals
  - AI metadata (latency, mode, parse method, reason stats)
  - memory map diagnostics
  - collision summary counters
  - calibration status

### 10.2 Decision log stream
One record per AI decision cycle:
- `sensor_snapshot`, `sensor_latest`
- `exploration`, `heatmap_diag`
- `ai_prompt`, `ai_raw`, `ai_parsed`, parse metadata
- safety override details
- requested vs executed action plan
- reason validation summary/stats
- post-hoc `outcome` (attached on next cycle/stop)

### 10.3 Collision log stream
From `Car.onCollide`:
- timestamp, impact velocity, region, position, other body id
- region categories:
  - `OUTER_NORTH`, `OUTER_SOUTH`, `OUTER_EAST`, `OUTER_WEST`
  - `INNER_OBSTACLE`
  - `OUTSIDE_BOUNDS`
  - `UNKNOWN`

### 10.4 Export artifact types
- Drive log JSON (`*_drive_gemma_drive_log_*.json`)
- Telemetry JSON (`*_telemetry_*.json`)
- Meta JSON (`*_meta_*.json`)
- HTML report (`*_report_driver_limit_report_*.html`)
- Preflight failure JSON (`*preflight_failure_logs_preflight_gate_*.json`)
- Experiment summary JSON (`*_experiment_automation_summary_*.json`)
- Optional all-in-one run bundle (`*_all_logs_*.json`)

## 11. Experiment automation specification

### 11.1 Config schema
- `schema = gemma-autodrive-experiment-config`
- `version = 1`
- Core fields:
  - `repeats` (`1..6`)
  - `runSeconds` (`20..240`)
  - `selectedConditionIds`
  - `saveMode` (`single_bundle_end` or `split_per_run`)
  - `startAttemptsPerRun` (`1..5`)
  - `includeHtmlReport` (bool)
  - `includeAllLogsBundle` (bool)

### 11.2 Condition matrix
- `AB-1`: 4b / adaptive sensor (7..14)
- `AB-2`: 12b / adaptive sensor (7..14)
- `AB-3`: 4b / fixed 10m sensor
- `AB-4`: 4b / wide adaptive sensor (6..16)

### 11.3 Automation flow
For each run:
1. Apply model + physics patch.
2. Preflight start with retry attempts.
3. Honor preflight cooldown waits.
4. Validate run-start vehicle state.
5. Execute timed run.
6. Stop and export run artifacts.
7. Update summary results.

Abort behavior:
- User stop request sets `STOP_REQUESTED`.
- Fail-fast on repeated ground/gravity precheck failures.

## 12. UI functional specification

Panels:
- Main HUD (left): score, target metrics, action, AI thought, debug output.
- Memory map + physics tuning (right/top).
- Direction calibration panel.
- Preflight panel (toggleable).
- Experiment panel (toggleable).
- Log download panel:
  - Save all logs
  - Download drive
  - Download telemetry
  - Download meta
  - Download preflight failure
  - Download HTML report

## 13. Error taxonomy and expected behavior

### 13.1 AI/API errors
- Timeout or API error -> safe hold + retry on next cycle.
- Parse error -> safe hold with parse metadata for diagnosis.

### 13.2 Preflight errors
- Non-PASS check blocks start.
- Always emits preflight gate failure log payload.
- Repeated ground/gravity failures activate cooldown/fail-fast.

### 13.3 Runtime physics/startup anomalies
Typical signature:
- all sensor distances at range max
- `worldX/Y/Z` near zero
- heading zero, speed zero
- `CAR_PHYSICS_READY` fails (`nearSpawn`, `grounded`, or related)

Expected behavior:
- Do not start autodrive.
- Record failure reason and metrics.

### 13.4 Initialization-order errors (TDZ class)
Class of issue:
- `ReferenceError: Cannot access '<symbol>' before initialization`

Mandatory prevention rule:
- Any helper called by hooks/effects must be declared before first usage.
- Do not reference `const` function expressions before declaration.

## 14. Current known issues and priority

### P0
- Stabilize startup/reset physics readiness across repeated runs.
- Eliminate repeated `CAR_PHYSICS_READY` precheck blocks in automation.
- Guarantee one complete summary artifact per automation session.

### P1
- Improve route quality under memory map guidance (reduce redundant revisits).
- Improve target-lock consistency when contact cues exist.
- Increase robustness of long-run automation without manual intervention.

### P2
- UI readability refinements and panel ergonomics.
- Additional analytics dimensions for post-run diagnosis.

## 15. Verification checklist (minimum)

Preflight:
- All checks can pass on healthy start.
- Failure JSON emits for each blocked start.

Autodrive loop:
- Vehicle stops while model inference is pending.
- Action queue executes in order with per-step validation.
- Runtime safety override activates under front hazard.

Memory map:
- Hazard cells become risk-biased/no-go over repeated collisions.
- Open-space evidence updates from non-hit ray segments.

Experiment automation:
- Selected AB conditions only.
- Retry/cooldown behavior works and is logged.
- Final summary JSON always produced.

Exports:
- Drive, telemetry, meta, report, preflight logs downloadable independently.
- Filenames include session prefix and trigger tag.

## 16. Change control rules

Before modifying control logic:
1. Update this specification section first.
2. Implement code changes.
3. Run one manual start-stop cycle and one short automation cycle.
4. Confirm produced logs match declared schema.

Before modifying data schema:
1. Preserve backward compatibility where practical.
2. Version the schema when breaking fields.
3. Update analysis and report generators in the same change set.

---

End of specification.
