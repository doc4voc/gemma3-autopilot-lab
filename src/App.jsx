import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import GameScene from "./components/GameScene";
import AutoAnalyst from "./components/AutoAnalyst"; // New Module
import { getDrivingDecision } from "./services/ollamaService"; // Removed getAvailableModels
import { createExplorationMemory } from "./services/explorationMemory";
import { generateAIReview, buildHTMLReportContent } from "./services/analysisService";

const TRACK_WORLD_BOUNDS = Object.freeze({
  minX: -19,
  maxX: 19,
  minZ: -19,
  maxZ: 19,
  softMargin: 3.5
});
const DEFAULT_CAR_SPAWN = Object.freeze({
  position: [0, 0.65, -10],
  rotation: [0, 0, 0]
});
const DEFAULT_TARGET_POSITION = Object.freeze([0, 1, 15]);

const asFiniteNumber = (value, fallback = 0) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);
const clamp01 = (value) => Math.max(0, Math.min(1, asFiniteNumber(value, 0)));
const wrapAngleDeg = (deg) => {
  let value = asFiniteNumber(deg, 0);
  while (value <= -180) value += 360;
  while (value > 180) value -= 360;
  return value;
};
const angleDiffDeg = (targetDeg, sourceDeg) => wrapAngleDeg(asFiniteNumber(targetDeg, 0) - asFiniteNumber(sourceDeg, 0));
const headingFromDelta = (dx, dz) => wrapAngleDeg(Math.atan2(asFiniteNumber(dx, 0), asFiniteNumber(dz, 0)) * (180 / Math.PI));
const circularMeanDeg = (angles) => {
  if (!Array.isArray(angles) || angles.length === 0) return 0;
  let sumSin = 0;
  let sumCos = 0;
  angles.forEach((deg) => {
    const rad = asFiniteNumber(deg, 0) * (Math.PI / 180);
    sumSin += Math.sin(rad);
    sumCos += Math.cos(rad);
  });
  if (Math.abs(sumSin) < 1e-6 && Math.abs(sumCos) < 1e-6) return 0;
  return wrapAngleDeg(Math.atan2(sumSin, sumCos) * (180 / Math.PI));
};
const THROTTLE_SIGN_DEADZONE = 0.08;
const ACTION_REASON_SIGN_DEADZONE = 0.08;
const DIRECTION_FLIP_COOLDOWN_MS = 1000;
const CALIBRATION_UPDATE_INTERVAL_MS = 600;
const CALIBRATION_MIN_MOTION_SAMPLES = 20;
const CALIBRATION_MIN_STEER_SAMPLES = 10;
const CALIBRATION_MIN_MOTION_DIST = 0.06;
const CALIBRATION_MAX_HISTORY = 900;
const CALIBRATION_APPLY_CONFIDENCE = 0.46;
const PREFLIGHT_GATE_POLICY = "ALL_PASS_REQUIRED";
const PREFLIGHT_FAILURE_LOG_FOLDER = "preflight_failure_logs";
const PREFLIGHT_AI_RETRY_PER_SCENARIO = 2;
const PREFLIGHT_AI_SCENARIOS = Object.freeze([
  "LOCK_OPEN_APPROACH",
  "ESCAPE_FRONT_BLOCKED",
  "EXPLORE_NO_TARGET"
]);
const COLLISION_REGIONS = Object.freeze([
  "OUTER_NORTH",
  "OUTER_SOUTH",
  "OUTER_EAST",
  "OUTER_WEST",
  "INNER_OBSTACLE",
  "OUTSIDE_BOUNDS",
  "UNKNOWN"
]);
const MODEL_OPTIONS = Object.freeze(["gemma3:4b", "gemma3:12b"]);
const EXPERIMENT_SAVE_MODES = Object.freeze({
  SINGLE_BUNDLE_END: "single_bundle_end",
  SPLIT_PER_RUN: "split_per_run"
});
const EXPERIMENT_START_MAX_ATTEMPTS = 2;
const EXPERIMENT_CONFIG_SCHEMA_ID = "gemma-autodrive-experiment-config";
const EXPERIMENT_CONFIG_SCHEMA_VERSION = 1;
const PREFLIGHT_BLOCK_RETRY_COOLDOWN_MS = 5000;
const EXPERIMENT_PRECHECK_BLOCK_STREAK_ABORT_THRESHOLD = 2;
const RUN_START_SPAWN_TOLERANCE_M = 3.0;
const RUN_START_MAX_SPEED = 0.8;
const RUN_START_MIN_WORLD_Y = 0.2;
const RUN_START_MAX_WORLD_Y = 2.2;
const RUN_START_MAX_VERTICAL_SPEED = 1.5;
const EXPERIMENT_CONDITION_MATRIX = Object.freeze([
  {
    id: "AB-1",
    label: "4b / adaptive sensor",
    model: "gemma3:4b",
    physicsPatch: { sensorDynamic: true, sensorRangeMin: 7, sensorRangeMax: 14 }
  },
  {
    id: "AB-2",
    label: "12b / adaptive sensor",
    model: "gemma3:12b",
    physicsPatch: { sensorDynamic: true, sensorRangeMin: 7, sensorRangeMax: 14 }
  },
  {
    id: "AB-3",
    label: "4b / fixed 10m sensor",
    model: "gemma3:4b",
    physicsPatch: { sensorDynamic: false, sensorRangeMin: 10, sensorRangeMax: 10.5 }
  },
  {
    id: "AB-4",
    label: "4b / wide adaptive sensor",
    model: "gemma3:4b",
    physicsPatch: { sensorDynamic: true, sensorRangeMin: 6, sensorRangeMax: 16 }
  }
]);

const createInitialExperimentConfig = () => ({
  repeats: 1,
  runSeconds: 75,
  selectedConditionIds: EXPERIMENT_CONDITION_MATRIX.map((condition) => condition.id),
  saveMode: EXPERIMENT_SAVE_MODES.SINGLE_BUNDLE_END,
  startAttemptsPerRun: EXPERIMENT_START_MAX_ATTEMPTS,
  includeHtmlReport: false,
  includeAllLogsBundle: false
});

const createInitialExperimentRunnerState = () => ({
  running: false,
  phase: "IDLE",
  abortRequested: false,
  currentRun: 0,
  totalRuns: 0,
  activeConditionId: "-",
  activeConditionLabel: "-",
  startedAt: null,
  finishedAt: null,
  summary: "Experiment automation idle.",
  results: []
});

const buildExperimentPlan = (config = {}) => {
  const repeats = Math.max(1, Math.min(6, Math.round(asFiniteNumber(config.repeats, 1))));
  const runSeconds = Math.max(20, Math.min(240, Math.round(asFiniteNumber(config.runSeconds, 75))));
  const selectedConditionIds = Array.isArray(config.selectedConditionIds)
    ? config.selectedConditionIds.map((id) => String(id || "").trim()).filter(Boolean)
    : [];
  const selectedSet = new Set(selectedConditionIds);
  const activeConditions = EXPERIMENT_CONDITION_MATRIX.filter((condition) => selectedSet.has(condition.id));
  const runs = [];

  for (let repeat = 1; repeat <= repeats; repeat += 1) {
    for (const condition of activeConditions) {
      const safeId = String(condition.id || "EXP").toUpperCase().replace(/[^A-Z0-9_-]/g, "_");
      runs.push({
        ...condition,
        repeat,
        runSeconds,
        runMs: runSeconds * 1000,
        runTag: `exp_${safeId.toLowerCase()}_r${repeat}`
      });
    }
  }

  return runs;
};

const summarizeTelemetryForExperiment = (
  history = [],
  decisionCount = 0,
  collisionSnapshot = createInitialCollisionStats()
) => {
  const points = Array.isArray(history) ? history : [];
  if (points.length === 0) {
    return {
      telemetrySamples: 0,
      decisionCount: asFiniteNumber(decisionCount, 0),
      startDistance: null,
      endDistance: null,
      minDistance: null,
      avgLatencyMs: null,
      minObstacleDist: null,
      maxTargetHitCount: 0,
      collisionCount: collisionSnapshot.totalCount || 0,
      sameWallCollisionCount: collisionSnapshot.sameWallRepeatCount || 0
    };
  }

  let minDistance = Number.POSITIVE_INFINITY;
  let minObstacleDist = Number.POSITIVE_INFINITY;
  let maxTargetHitCount = 0;
  let latencySum = 0;
  let latencyCount = 0;

  points.forEach((point) => {
    const distance = asFiniteNumber(point?.distanceToTarget, Number.POSITIVE_INFINITY);
    if (distance < minDistance) minDistance = distance;
    const minObstacle = asFiniteNumber(point?.minObstacleDist, Number.POSITIVE_INFINITY);
    if (minObstacle < minObstacleDist) minObstacleDist = minObstacle;
    const hits = Math.max(
      asFiniteNumber(point?.targetHitCount, 0),
      asFiniteNumber(point?.targetsReached, 0)
    );
    if (hits > maxTargetHitCount) maxTargetHitCount = hits;
    const latency = asFiniteNumber(point?.aiLatencyMs, 0);
    if (latency > 0) {
      latencySum += latency;
      latencyCount += 1;
    }
  });

  const first = points[0] || {};
  const last = points[points.length - 1] || {};

  return {
    telemetrySamples: points.length,
    decisionCount: asFiniteNumber(decisionCount, 0),
    startDistance: asFiniteNumber(first.distanceToTarget, null),
    endDistance: asFiniteNumber(last.distanceToTarget, null),
    minDistance: Number.isFinite(minDistance) ? Number(minDistance.toFixed(3)) : null,
    avgLatencyMs: latencyCount > 0 ? Number((latencySum / latencyCount).toFixed(1)) : null,
    minObstacleDist: Number.isFinite(minObstacleDist) ? Number(minObstacleDist.toFixed(3)) : null,
    maxTargetHitCount,
    collisionCount: collisionSnapshot.totalCount || 0,
    sameWallCollisionCount: collisionSnapshot.sameWallRepeatCount || 0
  };
};

const createInitialCollisionStats = () => ({
  totalCount: 0,
  sameWallRepeatCount: 0,
  sameWallConsecutiveRepeatCount: 0,
  byRegion: COLLISION_REGIONS.reduce((acc, region) => ({ ...acc, [region]: 0 }), {}),
  lastRegion: "NONE",
  lastCollisionAt: 0
});

const createInitialDirectionCalibrationProfile = () => ({
  headingSign: 1,
  headingOffsetDeg: 0,
  steeringSign: 1,
  headingConfidence: 0,
  steeringConfidence: 0,
  motionSampleCount: 0,
  steeringSampleCount: 0,
  motionErrorDeg: 180,
  steeringAgreement: 0.5,
  applied: false,
  status: "IDLE",
  reason: "not_enough_data",
  latestMotionHeadingDeg: null,
  latestHeadingDeg: null
});

const createInitialPreflightState = () => ({
  gatePolicy: PREFLIGHT_GATE_POLICY,
  status: "IDLE",
  overall: "UNKNOWN",
  startedAt: null,
  finishedAt: null,
  summary: "Preflight not started.",
  checks: [
    { id: "CAR_SENSOR_STREAM", area: "CAR", label: "Sensor stream integrity", status: "PENDING", detail: "", metric: null, blocking: true },
    { id: "CAR_POSE_BOUNDS", area: "CAR", label: "Pose within world bounds", status: "PENDING", detail: "", metric: null, blocking: true },
    { id: "CAR_PHYSICS_READY", area: "CAR", label: "Vehicle ground/gravity readiness", status: "PENDING", detail: "", metric: null, blocking: true },
    { id: "DIRECTION_MAPPING", area: "CAR", label: "Direction mapping convention", status: "PENDING", detail: "", metric: null, blocking: true },
    { id: "CALIBRATION_READY", area: "CAR", label: "Direction calibration readiness", status: "PENDING", detail: "", metric: null, blocking: false },
    { id: "AI_CONNECTIVITY", area: "AI", label: "AI connectivity and parsing", status: "PENDING", detail: "", metric: null, blocking: true },
    { id: "AI_STRATEGY_ALIGNMENT", area: "AI", label: "Strategy pattern consistency", status: "PENDING", detail: "", metric: null, blocking: true },
    { id: "AI_MODE_SKILL_DIVERSITY", area: "AI", label: "Mode/skill diversity", status: "PENDING", detail: "", metric: null, blocking: false }
  ],
  aiScenarioResults: [],
  blockingFailures: 0,
  warningCount: 0,
  nonPassCount: 0
});

const applyDirectionCalibrationToSensor = (sensor, profile) => {
  const source = sensor && typeof sensor === "object" ? sensor : {};
  const result = { ...source };
  const headingRaw = asFiniteNumber(source.headingDeg, 0);
  const sign = profile?.applied ? asFiniteNumber(profile.headingSign, 1) : 1;
  const offset = profile?.applied ? asFiniteNumber(profile.headingOffsetDeg, 0) : 0;
  result.headingDeg = wrapAngleDeg((headingRaw * sign) + offset);
  return result;
};

const snapshotCollisionStats = (stats) => {
  const source = stats || createInitialCollisionStats();
  return {
    totalCount: source.totalCount || 0,
    sameWallRepeatCount: source.sameWallRepeatCount || 0,
    sameWallConsecutiveRepeatCount: source.sameWallConsecutiveRepeatCount || 0,
    byRegion: COLLISION_REGIONS.reduce((acc, region) => ({ ...acc, [region]: source.byRegion?.[region] || 0 }), {}),
    lastRegion: source.lastRegion || "NONE",
    lastCollisionAt: source.lastCollisionAt || 0
  };
};

const regionForCollisionCounting = (region) => {
  const normalized = typeof region === "string" ? region.trim().toUpperCase() : "UNKNOWN";
  return COLLISION_REGIONS.includes(normalized) ? normalized : "UNKNOWN";
};

const throttleToSign = (throttle) => {
  const t = asFiniteNumber(throttle, 0);
  if (t > THROTTLE_SIGN_DEADZONE) return 1;
  if (t < -THROTTLE_SIGN_DEADZONE) return -1;
  return 0;
};

const controlValueToSign = (value, deadzone = ACTION_REASON_SIGN_DEADZONE) => {
  const v = asFiniteNumber(value, 0);
  if (v > deadzone) return 1;
  if (v < -deadzone) return -1;
  return 0;
};

const normalizeReasonCode = (raw, fallback = "UNSPECIFIED_REASON") => {
  const token = typeof raw === "string"
    ? raw.trim().toUpperCase().replace(/[\s-]+/g, "_").replace(/[^A-Z0-9_]/g, "")
    : "";
  return token ? token.slice(0, 48) : fallback;
};

const normalizeReasonSign = (raw, fallback = null) => {
  if (raw === null || raw === undefined) return fallback;
  if (typeof raw === "number" && Number.isFinite(raw)) {
    if (raw > 0.2) return 1;
    if (raw < -0.2) return -1;
    return 0;
  }
  if (typeof raw === "string") {
    const token = raw.trim().toUpperCase();
    if (!token || token === "ANY" || token === "AUTO") return fallback;
    if (token === "1" || token === "+1" || token === "FORWARD" || token === "LEFT" || token === "POSITIVE") return 1;
    if (token === "-1" || token === "REVERSE" || token === "RIGHT" || token === "NEGATIVE") return -1;
    if (token === "0" || token === "HOLD" || token === "NEUTRAL" || token === "STRAIGHT" || token === "CENTER") return 0;
  }
  return fallback;
};

const inferFallbackReason = (step, fallbackReason = null) => {
  const throttleSign = controlValueToSign(step?.throttle);
  const steeringSign = controlValueToSign(step?.steering);
  const fallbackCode = normalizeReasonCode(fallbackReason?.code, "UNSPECIFIED_REASON");
  const fallbackSummary = typeof fallbackReason?.summary === "string" ? fallbackReason.summary.trim() : "";
  if (fallbackSummary) {
    return {
      code: fallbackCode,
      summary: fallbackSummary.slice(0, 180),
      expectedThrottleSign: normalizeReasonSign(fallbackReason?.expectedThrottleSign, throttleSign),
      expectedSteeringSign: normalizeReasonSign(fallbackReason?.expectedSteeringSign, steeringSign),
      source: fallbackReason?.source || "fallback"
    };
  }
  if (throttleSign > 0 && steeringSign === 0) {
    return {
      code: "FORWARD_PROBE",
      summary: "Advance straight toward target/frontier.",
      expectedThrottleSign: 1,
      expectedSteeringSign: 0,
      source: "fallback"
    };
  }
  if (throttleSign > 0 && steeringSign !== 0) {
    return {
      code: "FORWARD_TURN_APPROACH",
      summary: "Advance while turning toward safer direction.",
      expectedThrottleSign: 1,
      expectedSteeringSign: steeringSign,
      source: "fallback"
    };
  }
  if (throttleSign < 0) {
    return {
      code: "REVERSE_ESCAPE",
      summary: "Reverse to reduce obstacle pressure.",
      expectedThrottleSign: -1,
      expectedSteeringSign: steeringSign,
      source: "fallback"
    };
  }
  if (steeringSign !== 0) {
    return {
      code: "PIVOT_SCAN",
      summary: "Pivot in place to scan safer heading.",
      expectedThrottleSign: 0,
      expectedSteeringSign: steeringSign,
      source: "fallback"
    };
  }
  return {
    code: "HOLD_AND_REASSESS",
    summary: "Hold and request next decision.",
    expectedThrottleSign: 0,
    expectedSteeringSign: 0,
    source: "fallback"
  };
};

const normalizeReasonEnvelope = (rawReason, fallbackReason = null, step = {}) => {
  const fallback = inferFallbackReason(step, fallbackReason);
  const reasonObj = (typeof rawReason === "string")
    ? { summary: rawReason }
    : (rawReason && typeof rawReason === "object")
      ? rawReason
      : {};
  const hasModelCode = typeof reasonObj.code === "string" && reasonObj.code.trim().length > 0;
  const hasModelSummary = typeof reasonObj.summary === "string" && reasonObj.summary.trim().length > 0;
  const code = normalizeReasonCode(reasonObj.code || reasonObj.reasonCode || reasonObj.label || fallback.code, fallback.code);
  const summaryRaw = reasonObj.summary || reasonObj.reason || reasonObj.text || reasonObj.rationale || fallback.summary;
  const summary = typeof summaryRaw === "string" && summaryRaw.trim().length > 0
    ? summaryRaw.trim().replace(/\s+/g, " ").slice(0, 180)
    : fallback.summary;
  const expectedThrottleSign = normalizeReasonSign(
    reasonObj.expectedThrottleSign ?? reasonObj.throttleSign ?? reasonObj.expected?.throttleSign,
    fallback.expectedThrottleSign
  );
  const expectedSteeringSign = normalizeReasonSign(
    reasonObj.expectedSteeringSign ?? reasonObj.steeringSign ?? reasonObj.expected?.steeringSign,
    fallback.expectedSteeringSign
  );
  const sourceRaw = typeof reasonObj.source === "string" ? reasonObj.source.trim().toLowerCase() : "";
  const source = sourceRaw || ((hasModelCode || hasModelSummary) ? "model" : fallback.source || "fallback");
  return {
    code,
    summary,
    expectedThrottleSign,
    expectedSteeringSign,
    source
  };
};

const didAllPreflightChecksPass = (report) => {
  const checks = Array.isArray(report?.checks) ? report.checks : [];
  return checks.length > 0 && checks.every((check) => check.status === "PASS");
};

const getPreflightNonPassSummary = (report, maxItems = 3) => {
  const checks = Array.isArray(report?.checks) ? report.checks : [];
  const nonPass = checks.filter((check) => check.status !== "PASS");
  if (nonPass.length === 0) return "All checks PASS.";
  const compact = nonPass.slice(0, Math.max(1, maxItems)).map((check) => {
    const id = String(check?.id || "UNKNOWN");
    const detail = typeof check?.detail === "string" && check.detail.trim().length > 0
      ? check.detail.trim().replace(/\s+/g, " ").slice(0, 120)
      : String(check?.status || "non-pass");
    return `${id}: ${detail}`;
  });
  const remain = nonPass.length - compact.length;
  return remain > 0
    ? `${compact.join(" | ")} | +${remain} more`
    : compact.join(" | ");
};

const isGroundGravityPreflightFailure = (report) => {
  const checks = Array.isArray(report?.checks) ? report.checks : [];
  const physicsCheck = checks.find((check) => String(check?.id || "").trim().toUpperCase() === "CAR_PHYSICS_READY");
  if (!physicsCheck) return false;
  if (String(physicsCheck.status || "").toUpperCase() !== "FAIL") return false;
  const detail = String(physicsCheck.detail || "").toLowerCase();
  return detail.includes("grounded") || detail.includes("gravitystable") || detail.includes("worldy") || detail.includes("verticalspeed");
};

const createInitialReasonValidationStats = () => ({
  totalSteps: 0,
  passedSteps: 0,
  blockedSteps: 0,
  missingModelReasonSteps: 0,
  signMismatchSteps: 0
});

const validateReasonedActionStep = ({ step, reason, minFrontDist, requireModelReason = true, bypass = false }) => {
  const issues = [];
  const normalizedReason = normalizeReasonEnvelope(reason, null, step);
  const throttleSign = controlValueToSign(step?.throttle);
  const steeringSign = controlValueToSign(step?.steering);

  if (!bypass && requireModelReason && normalizedReason.source !== "model") {
    issues.push("REASON_NOT_FROM_MODEL");
  }
  if (!normalizedReason.code || !normalizedReason.summary) {
    issues.push("REASON_EMPTY");
  }
  if (!bypass && normalizedReason.expectedThrottleSign !== null && normalizedReason.expectedThrottleSign !== throttleSign) {
    issues.push("THROTTLE_SIGN_MISMATCH");
  }
  if (!bypass && normalizedReason.expectedSteeringSign !== null && normalizedReason.expectedSteeringSign !== steeringSign) {
    issues.push("STEERING_SIGN_MISMATCH");
  }
  if (!bypass && minFrontDist < 2.5 && throttleSign > 0) {
    issues.push("FORWARD_INTO_FRONT_RISK");
  }

  return {
    ok: issues.length === 0,
    issues,
    primaryIssue: issues[0] || "",
    throttleSign,
    steeringSign,
    reason: normalizedReason,
    bypassed: !!bypass
  };
};

const buildHeatmapDecisionDiagnostics = (explorationContext, strategy, sensorSnapshot, latestSensor) => {
  const diagnostics = explorationContext?.diagnostics || {};
  const chosenSector = strategy?.chosenSector || explorationContext?.preferredSector || "F";
  const topCandidates = Array.isArray(diagnostics.topCandidates) ? diagnostics.topCandidates : [];
  const topSafeCandidates = Array.isArray(diagnostics.topSafeCandidates) ? diagnostics.topSafeCandidates : [];

  const preferredSafe = topSafeCandidates.find((c) => c.sector === chosenSector) || null;
  const fallbackSafe = topSafeCandidates[0] || null;
  const fallbackBySector = topCandidates.find((c) => c.sector === chosenSector) || null;
  const selectedCandidate = preferredSafe || fallbackSafe || fallbackBySector || topCandidates[0] || null;

  const selectedSource = preferredSafe
    ? "chosen_sector_safe"
    : fallbackSafe
      ? "fallback_best_safe"
      : fallbackBySector
        ? "chosen_sector_nogo"
        : selectedCandidate
          ? "fallback_top_candidate"
          : "none";

  return {
    chosenSector,
    selectedSource,
    currentCellWeight: explorationContext?.currentCell?.weightScore ?? null,
    selectedCellWeight: selectedCandidate?.score ?? null,
    selectedCellNoGo: !!selectedCandidate?.isNoGo,
    selectedCellNoGoReasons: selectedCandidate?.noGoReasons || [],
    selectedCell: selectedCandidate ? {
      ix: selectedCandidate.ix,
      iz: selectedCandidate.iz,
      sector: selectedCandidate.sector,
      score: selectedCandidate.score,
      risk: selectedCandidate.risk
    } : null,
    noGoRatio: diagnostics.noGoRatio ?? null,
    revisitRate: diagnostics.revisitRate ?? explorationContext?.loopRate ?? null,
    topCandidates,
    topSafeCandidates,
    sectorSafety: diagnostics.sectorSafety || [],
    targetBearingDeg: asFiniteNumber(sensorSnapshot?.angleToTarget, 0),
    targetDistance: asFiniteNumber(sensorSnapshot?.distanceToTarget, 0),
    targetHitCount: Object.values(sensorSnapshot?.targetHits || {}).filter(Boolean).length,
    minObstacleDistNow: Math.min(
      asFiniteNumber(latestSensor?.front, 10),
      asFiniteNumber(latestSensor?.leftDiag, 10),
      asFiniteNumber(latestSensor?.rightDiag, 10),
      asFiniteNumber(latestSensor?.left, 10),
      asFiniteNumber(latestSensor?.right, 10),
      asFiniteNumber(latestSensor?.back, 10),
      asFiniteNumber(latestSensor?.backLeft, 10),
      asFiniteNumber(latestSensor?.backRight, 10)
    )
  };
};

const countTargetHits = (snapshot) => Object.values(snapshot?.targetHits || {}).filter(Boolean).length;

const minObstacleDistance = (snapshot) => Math.min(
  asFiniteNumber(snapshot?.front, 10),
  asFiniteNumber(snapshot?.leftDiag, 10),
  asFiniteNumber(snapshot?.rightDiag, 10),
  asFiniteNumber(snapshot?.left, 10),
  asFiniteNumber(snapshot?.right, 10),
  asFiniteNumber(snapshot?.back, 10),
  asFiniteNumber(snapshot?.backLeft, 10),
  asFiniteNumber(snapshot?.backRight, 10)
);

const buildDecisionOutcome = (entry, endSensor, endExploration, nowMs = Date.now()) => {
  if (!entry?.sensor_snapshot || !endSensor) return null;

  const startSensor = entry.sensor_snapshot;
  const startDistance = asFiniteNumber(startSensor.distanceToTarget, 0);
  const endDistance = asFiniteNumber(endSensor.distanceToTarget, startDistance);
  const progressDelta = startDistance - endDistance;
  const startMinObstacle = minObstacleDistance(startSensor);
  const endMinObstacle = minObstacleDistance(endSensor);
  const minObstacleDelta = endMinObstacle - startMinObstacle;
  const startLoopRate = asFiniteNumber(entry?.exploration?.loopRate, 0);
  const endLoopRate = asFiniteNumber(endExploration?.loopRate, startLoopRate);
  const loopRateDelta = endLoopRate - startLoopRate;
  const targetHitDelta = countTargetHits(endSensor) - countTargetHits(startSensor);
  const elapsedMs = Math.max(0, nowMs - asFiniteNumber(entry?.decision_started_at_ms, nowMs));
  const safetyOverride = !!entry?.direction_cooldown?.applied || !!entry?.runtime_safety_override || !!entry?.safety_guard?.guardApplied;

  let label = "MIXED";
  if (targetHitDelta > 0) label = "TARGET_REACQUIRED";
  else if (progressDelta > 0.55 && endMinObstacle >= 2.4) label = "GOOD_PROGRESS";
  else if (progressDelta < -0.55 && endMinObstacle < 2.5) label = "RISKY_REGRESSION";
  else if (Math.abs(progressDelta) < 0.25 && loopRateDelta > 0.08) label = "LOOP_RISK";
  else if (Math.abs(progressDelta) < 0.2 && endMinObstacle < 2.4) label = "CAUTIOUS_HOLD";

  const skillName = entry?.ai_skill?.name || "UNKNOWN";
  const summary = `${label}: skill=${skillName}, progress=${progressDelta.toFixed(2)}m, minObs=${endMinObstacle.toFixed(2)}m, loopDelta=${loopRateDelta.toFixed(3)}, hitsDelta=${targetHitDelta}`;

  return {
    label,
    summary,
    elapsedMs,
    progressDeltaM: Number(progressDelta.toFixed(3)),
    startDistanceM: Number(startDistance.toFixed(3)),
    endDistanceM: Number(endDistance.toFixed(3)),
    startMinObstacleM: Number(startMinObstacle.toFixed(3)),
    endMinObstacleM: Number(endMinObstacle.toFixed(3)),
    minObstacleDeltaM: Number(minObstacleDelta.toFixed(3)),
    targetHitDelta,
    loopRateStart: Number(startLoopRate.toFixed(3)),
    loopRateEnd: Number(endLoopRate.toFixed(3)),
    loopRateDelta: Number(loopRateDelta.toFixed(3)),
    safetyOverride
  };
};

export default function App() {
  const [sensorData, setSensorData] = useState({
    left: 0,
    leftDiag: 0,
    front: 0,
    rightDiag: 0,
    right: 0,
    worldX: 0,
    worldY: 0,
    worldZ: 0,
    headingDeg: 0,
    targetHits: {},
    angleToTarget: 0,
    distanceToTarget: 0,
    speed: 0,
    verticalSpeed: 0,
    grounded: false,
    back: 0,
    backLeft: 0,
    backRight: 0,
    isStuck: false,
    sensorRange: 10
  });

  const [selectedModel, setSelectedModel] = useState("gemma3:4b");

  const [controls, setControls] = useState({ throttle: 0, steering: 0 }); // Analog Control State (0-1, -1 to 1)
  const controlRef = useRef({ throttle: 0, steering: 0 });
  const [lastAction, setLastAction] = useState("BRAKE"); // Keeping for Legacy HUD display
  const [actionHistory, setActionHistory] = useState([]); // Keep last 3 actions
  const [isThinking, setIsThinking] = useState(false);
  const [autoDrive, setAutoDrive] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showAnalyticsPanel, setShowAnalyticsPanel] = useState(false);
  const [panelVisibility, setPanelVisibility] = useState({
    priority: true,
    physics: true,
    memory: true,
    calibration: true,
    preflight: true,
    experiment: true,
    hud: true,
    controls: true,
    logs: true,
    branding: true
  });
  const [errorMsg, setErrorMsg] = useState(null);
  const aiMetaRef = useRef({
    latency: 0,
    thought: "",
    analysis: "",
    reasonCode: "",
    reasonSummary: "",
    reasonSource: "none",
    reasonValidationLast: "NONE",
    reasonBlockedTotal: 0,
    reasonPassRate: 1,
    strategyMode: "UNKNOWN",
    strategyTransition: "HOLD",
    strategySector: "F",
    strategyConfidence: 0,
    skillName: "UNKNOWN",
    skillIntensity: 0,
    reflectionAdjustment: "",
    parseMethod: "",
    parseRecovered: false,
    model: "",
    source: "IDLE",
    isThinking: false,
    safetyOverride: false,
    overrideReason: "",
    lastDecisionAt: 0,
    memoryNoGoRatio: 0,
    memoryRevisitRate: 0,
    memoryCurrentWeight: 0,
    memorySelectedWeight: 0,
    memorySelectedNoGo: false,
    memorySelectedSector: "F",
    memorySelectionReason: "",
    directionCooldownApplied: false,
    directionCooldownRemainingMs: 0,
    directionLastSign: 0
  });
  const targetCaptureRef = useRef({ count: 0, lastCaptureAt: 0 });
  const sessionRef = useRef({ startAt: Date.now(), lastDistance: null });
  const collisionEventsRef = useRef([]);
  const collisionStatsRef = useRef(createInitialCollisionStats());
  const directionCalibrationRef = useRef({
    enabled: true,
    autoApply: true,
    profile: createInitialDirectionCalibrationProfile(),
    samples: [],
    lastUpdateAt: 0
  });
  const [directionCalibView, setDirectionCalibView] = useState(createInitialDirectionCalibrationProfile());
  const explorationMemoryRef = useRef(createExplorationMemory({
    cellSize: 2.0,
    sensorRange: 10.0,
    worldBounds: TRACK_WORLD_BOUNDS
  }));
  const [memoryViz, setMemoryViz] = useState(null);
  const [isPreflightRunning, setIsPreflightRunning] = useState(false);
  const [preflightReport, setPreflightReport] = useState(createInitialPreflightState());
  const preflightGateLogsRef = useRef([]);
  const reasonValidationStatsRef = useRef(createInitialReasonValidationStats());
  const sensorTickRef = useRef(0);
  const [experimentConfig, setExperimentConfig] = useState(createInitialExperimentConfig());
  const [experimentRunner, setExperimentRunner] = useState(createInitialExperimentRunnerState());
  const experimentConfigFileInputRef = useRef(null);
  const [experimentConfigJsonStatus, setExperimentConfigJsonStatus] = useState("No JSON config loaded.");
  const experimentAbortRef = useRef(false);
  const autoDriveRef = useRef(false);
  const isAnalyzingRef = useRef(false);
  const isPreflightRunningRef = useRef(false);
  const telemetryRef = useRef([]);
  const selectedModelRef = useRef("gemma3:4b");
  const errorMsgRef = useRef(null);
  const preflightReportRef = useRef(createInitialPreflightState());
  const preflightBlockGuardRef = useRef({
    blockedUntil: 0,
    reason: "",
    repeatCount: 0
  });

  useEffect(() => {
    autoDriveRef.current = autoDrive;
  }, [autoDrive]);

  useEffect(() => {
    isAnalyzingRef.current = isAnalyzing;
  }, [isAnalyzing]);

  useEffect(() => {
    isPreflightRunningRef.current = isPreflightRunning;
  }, [isPreflightRunning]);

  useEffect(() => {
    selectedModelRef.current = selectedModel;
  }, [selectedModel]);

  useEffect(() => {
    errorMsgRef.current = errorMsg;
  }, [errorMsg]);

  useEffect(() => {
    preflightReportRef.current = preflightReport;
  }, [preflightReport]);

  const recordCollisionEvent = useCallback((evt) => {
    const region = regionForCollisionCounting(evt?.region);
    const stats = collisionStatsRef.current;
    const previousCount = stats.byRegion[region] || 0;
    stats.byRegion[region] = previousCount + 1;
    stats.totalCount += 1;
    if (previousCount > 0) stats.sameWallRepeatCount += 1;
    if (stats.lastRegion === region) stats.sameWallConsecutiveRepeatCount += 1;
    stats.lastRegion = region;
    stats.lastCollisionAt = Date.now();

    collisionEventsRef.current.push({
      time: evt?.time || new Date().toISOString(),
      region,
      impactVelocity: asFiniteNumber(evt?.impactVelocity, 0),
      worldX: asFiniteNumber(evt?.worldX, 0),
      worldZ: asFiniteNumber(evt?.worldZ, 0),
      otherBodyId: Number.isFinite(evt?.otherBodyId) ? evt.otherBodyId : null
    });
  }, []);

  const recomputeDirectionCalibration = useCallback(() => {
    const state = directionCalibrationRef.current;
    const samples = state.samples;
    if (!Array.isArray(samples) || samples.length < 3) return;

    const normalDiffs = [];
    const flippedDiffs = [];
    const steeringPairs = [];
    let latestMotionHeadingDeg = null;
    let latestHeadingDeg = null;

    for (let i = 1; i < samples.length; i += 1) {
      const prev = samples[i - 1];
      const curr = samples[i];
      const dx = asFiniteNumber(curr.x, 0) - asFiniteNumber(prev.x, 0);
      const dz = asFiniteNumber(curr.z, 0) - asFiniteNumber(prev.z, 0);
      const moveDist = Math.hypot(dx, dz);
      if (moveDist < CALIBRATION_MIN_MOTION_DIST) continue;

      const motionHeading = headingFromDelta(dx, dz);
      const headingNow = asFiniteNumber(curr.heading, 0);
      latestMotionHeadingDeg = motionHeading;
      latestHeadingDeg = headingNow;
      normalDiffs.push(angleDiffDeg(motionHeading, headingNow));
      flippedDiffs.push(angleDiffDeg(motionHeading, -headingNow));

      const steerMag = Math.abs(asFiniteNumber(prev.steering, 0));
      if (steerMag >= 0.2 && moveDist >= 0.08) {
        const headingDelta = angleDiffDeg(asFiniteNumber(curr.heading, 0), asFiniteNumber(prev.heading, 0));
        if (Math.abs(headingDelta) >= 2.0) {
          steeringPairs.push({
            steeringSign: Math.sign(asFiniteNumber(prev.steering, 0)),
            headingDeltaSign: Math.sign(headingDelta)
          });
        }
      }
    }

    const motionSampleCount = normalDiffs.length;
    const steeringSampleCount = steeringPairs.length;
    if (motionSampleCount < CALIBRATION_MIN_MOTION_SAMPLES) {
      const lowDataProfile = {
        ...state.profile,
        motionSampleCount,
        steeringSampleCount,
        status: "COLLECTING",
        reason: "insufficient_motion_samples",
        applied: false
      };
      state.profile = lowDataProfile;
      setDirectionCalibView({ ...lowDataProfile });
      return;
    }

    const normalMae = normalDiffs.reduce((sum, deg) => sum + Math.abs(deg), 0) / motionSampleCount;
    const flippedMae = flippedDiffs.reduce((sum, deg) => sum + Math.abs(deg), 0) / motionSampleCount;
    const useFlipped = flippedMae + 1.5 < normalMae;
    const headingSign = useFlipped ? -1 : 1;
    const chosenDiffs = useFlipped ? flippedDiffs : normalDiffs;
    const headingOffsetDeg = circularMeanDeg(chosenDiffs);
    const postErrors = chosenDiffs.map((deg) => Math.abs(angleDiffDeg(deg, headingOffsetDeg)));
    const motionErrorDeg = postErrors.reduce((sum, deg) => sum + deg, 0) / Math.max(1, postErrors.length);
    const sampleStrength = clamp01(motionSampleCount / 120);
    const headingAccuracy = clamp01((55 - motionErrorDeg) / 55);
    const headingConfidence = clamp01((sampleStrength * 0.55) + (headingAccuracy * 0.45));

    let steeringAgreement = 0.5;
    let steeringSign = 1;
    let steeringConfidence = 0;
    if (steeringSampleCount >= CALIBRATION_MIN_STEER_SAMPLES) {
      const positiveAgreement = steeringPairs.filter((p) => p.steeringSign !== 0 && p.headingDeltaSign !== 0 && p.steeringSign === p.headingDeltaSign).length;
      const negativeAgreement = steeringPairs.filter((p) => p.steeringSign !== 0 && p.headingDeltaSign !== 0 && p.steeringSign === -p.headingDeltaSign).length;
      const total = Math.max(1, positiveAgreement + negativeAgreement);
      steeringAgreement = positiveAgreement / total;
      const antiAgreement = negativeAgreement / total;
      steeringSign = antiAgreement > steeringAgreement ? -1 : 1;
      steeringConfidence = clamp01(Math.abs(steeringAgreement - antiAgreement) * clamp01(total / 60));
    }

    const shouldApply = state.enabled && state.autoApply && headingConfidence >= CALIBRATION_APPLY_CONFIDENCE;
    const nextProfile = {
      headingSign,
      headingOffsetDeg: Number(headingOffsetDeg.toFixed(2)),
      steeringSign,
      headingConfidence: Number(headingConfidence.toFixed(3)),
      steeringConfidence: Number(steeringConfidence.toFixed(3)),
      motionSampleCount,
      steeringSampleCount,
      motionErrorDeg: Number(motionErrorDeg.toFixed(2)),
      steeringAgreement: Number(steeringAgreement.toFixed(3)),
      applied: !!shouldApply,
      status: shouldApply ? "APPLIED" : "MONITORING",
      reason: shouldApply ? "auto_applied" : "low_confidence_or_manual",
      latestMotionHeadingDeg,
      latestHeadingDeg
    };
    state.profile = nextProfile;
    setDirectionCalibView({ ...nextProfile });
  }, []);

  const resetDirectionCalibration = useCallback(() => {
    directionCalibrationRef.current = {
      enabled: true,
      autoApply: true,
      profile: createInitialDirectionCalibrationProfile(),
      samples: [],
      lastUpdateAt: 0
    };
    setDirectionCalibView(createInitialDirectionCalibrationProfile());
  }, []);

  const setAction = (action) => {
    setLastAction(action);
    if (action !== "AI_CONTROL") {
      setActionHistory(prev => [...prev.slice(-2), action]);
    }
  };

  const applyControls = useCallback((nextControls) => {
    setControls(nextControls);
    controlRef.current = nextControls;

    // Keep legacy debug channels in sync for any remaining consumers.
    window.currentSteering = nextControls.steering;
    window.currentThrottle = nextControls.throttle;
  }, []);

  const prepareSessionForStart = useCallback(() => {
    setTelemetry([]);
    decisionLog.current = [];
    collisionEventsRef.current = [];
    collisionStatsRef.current = createInitialCollisionStats();
    targetCaptureRef.current = { count: 0, lastCaptureAt: 0 };
    sessionRef.current = { startAt: Date.now(), lastDistance: null };
    setTargetPosition([...DEFAULT_TARGET_POSITION]);
    setCarResetNonce((prev) => prev + 1);
    explorationMemoryRef.current.reset();
    directionFlipRef.current = { lastSign: 0, lastSignAt: 0 };
    smoothingRef.current = {
      lastSteering: 0,
      lastStrategyMode: "",
      noContactMs: 0,
      noContactCycles: 0,
      lastContactAt: 0,
      lastTickAt: 0,
      reacquireTurnDir: 1,
      lastReacquireFlipAt: 0,
      lastOutcomeSummary: "",
      lastOutcomeDetails: null,
      lastReflectionHint: "",
      lastSkillName: "",
      modeHoldRemaining: 0,
      targetLockHoldRemaining: 0
    };
    reasonValidationStatsRef.current = createInitialReasonValidationStats();
    setMemoryViz(null);
    aiMetaRef.current = {
      latency: 0,
      thought: "",
      analysis: "",
      reasonCode: "",
      reasonSummary: "",
      reasonSource: "none",
      reasonValidationLast: "NONE",
      reasonBlockedTotal: 0,
      reasonPassRate: 1,
      strategyMode: "UNKNOWN",
      strategyTransition: "HOLD",
      strategySector: "F",
      strategyConfidence: 0,
      skillName: "UNKNOWN",
      skillIntensity: 0,
      reflectionAdjustment: "",
      parseMethod: "",
      parseRecovered: false,
      model: "",
      source: "IDLE",
      isThinking: false,
      safetyOverride: false,
      overrideReason: "",
      lastDecisionAt: 0,
      memoryNoGoRatio: 0,
      memoryRevisitRate: 0,
      memoryCurrentWeight: 0,
      memorySelectedWeight: 0,
      memorySelectedNoGo: false,
      memorySelectedSector: "F",
      memorySelectionReason: "",
      directionCooldownApplied: false,
      directionCooldownRemainingMs: 0,
      directionLastSign: 0
    };
    setPreflightReport(createInitialPreflightState());
  }, []);

  const runStartupPreflight = useCallback(async (trigger = "manual", modelOverride = null) => {
    const preflightModel = (typeof modelOverride === "string" && modelOverride.trim())
      ? modelOverride.trim()
      : selectedModel;
    let report = {
      ...createInitialPreflightState(),
      status: "RUNNING",
      startedAt: new Date().toISOString(),
      summary: `Running startup integrity checks (${String(trigger || "manual")}, ${preflightModel})...`
    };
    setPreflightReport(report);

    const updateCheck = (id, status, detail, metric = null) => {
      report = {
        ...report,
        checks: report.checks.map((check) => check.id === id ? { ...check, status, detail, metric } : check)
      };
      setPreflightReport({ ...report });
    };

    const sensor = sensorRef.current || {};
    const requiredSensorKeys = [
      "left", "leftDiag", "front", "rightDiag", "right", "back", "backLeft", "backRight",
      "worldX", "worldY", "worldZ", "headingDeg", "distanceToTarget", "angleToTarget", "speed", "verticalSpeed", "sensorRange"
    ];
    const finiteSensorCount = requiredSensorKeys.filter((key) => Number.isFinite(sensor?.[key])).length;
    if (finiteSensorCount === requiredSensorKeys.length) {
      updateCheck("CAR_SENSOR_STREAM", "PASS", `All required sensor channels are finite (${finiteSensorCount}/${requiredSensorKeys.length}).`, { finiteSensorCount });
    } else {
      updateCheck("CAR_SENSOR_STREAM", "FAIL", `Missing/invalid sensor channels (${finiteSensorCount}/${requiredSensorKeys.length}).`, { finiteSensorCount });
    }

    const x = asFiniteNumber(sensor.worldX, 0);
    const z = asFiniteNumber(sensor.worldZ, 0);
    const inBounds = x >= TRACK_WORLD_BOUNDS.minX && x <= TRACK_WORLD_BOUNDS.maxX && z >= TRACK_WORLD_BOUNDS.minZ && z <= TRACK_WORLD_BOUNDS.maxZ;
    if (inBounds) {
      updateCheck("CAR_POSE_BOUNDS", "PASS", `Pose in bounds (x=${x.toFixed(1)}, z=${z.toFixed(1)}).`, { x, z });
    } else {
      updateCheck("CAR_POSE_BOUNDS", "FAIL", `Pose out of bounds (x=${x.toFixed(1)}, z=${z.toFixed(1)}).`, { x, z });
    }

    const y = asFiniteNumber(sensor.worldY, Number.NaN);
    const speedAbs = Math.abs(asFiniteNumber(sensor.speed, Number.NaN));
    const verticalSpeedAbs = Math.abs(asFiniteNumber(sensor.verticalSpeed, Number.NaN));
    const distanceFromSpawn = Number.isFinite(x) && Number.isFinite(z)
      ? Math.hypot(x - asFiniteNumber(DEFAULT_CAR_SPAWN.position?.[0], 0), z - asFiniteNumber(DEFAULT_CAR_SPAWN.position?.[2], -10))
      : Number.POSITIVE_INFINITY;
    const physicsChecks = {
      finitePose: Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z),
      nearSpawn: Number.isFinite(distanceFromSpawn) && distanceFromSpawn <= RUN_START_SPAWN_TOLERANCE_M,
      speedReady: Number.isFinite(speedAbs) && speedAbs <= RUN_START_MAX_SPEED,
      grounded: Number.isFinite(y) && y >= RUN_START_MIN_WORLD_Y && y <= RUN_START_MAX_WORLD_Y,
      gravityStable: Number.isFinite(verticalSpeedAbs) && verticalSpeedAbs <= RUN_START_MAX_VERTICAL_SPEED
    };
    const failedPhysicsChecks = Object.entries(physicsChecks)
      .filter(([, pass]) => !pass)
      .map(([name]) => name);
    const physicsMetrics = {
      worldX: Number.isFinite(x) ? Number(x.toFixed(3)) : null,
      worldY: Number.isFinite(y) ? Number(y.toFixed(3)) : null,
      worldZ: Number.isFinite(z) ? Number(z.toFixed(3)) : null,
      speed: Number.isFinite(speedAbs) ? Number(speedAbs.toFixed(3)) : null,
      verticalSpeed: Number.isFinite(verticalSpeedAbs) ? Number(verticalSpeedAbs.toFixed(3)) : null,
      distanceFromSpawn: Number.isFinite(distanceFromSpawn) ? Number(distanceFromSpawn.toFixed(3)) : null
    };
    if (failedPhysicsChecks.length === 0) {
      updateCheck("CAR_PHYSICS_READY", "PASS", "Vehicle pose/speed/gravity state is valid for run start.", physicsMetrics);
    } else {
      updateCheck("CAR_PHYSICS_READY", "FAIL", `Vehicle start state invalid: ${failedPhysicsChecks.join(", ")}.`, physicsMetrics);
    }

    const mappingOk = (
      Math.abs(headingFromDelta(0, 1) - 0) < 0.001
      && Math.abs(headingFromDelta(1, 0) - 90) < 0.001
      && Math.abs(headingFromDelta(0, -1) - 180) < 0.001
      && Math.abs(headingFromDelta(-1, 0) + 90) < 0.001
    );
    if (mappingOk) {
      updateCheck("DIRECTION_MAPPING", "PASS", "Coordinate and heading mapping constants are internally consistent.", null);
    } else {
      updateCheck("DIRECTION_MAPPING", "FAIL", "Coordinate-heading mapping failed self-test.", null);
    }

    const calibrationState = directionCalibrationRef.current || {};
    const calib = calibrationState.profile || createInitialDirectionCalibrationProfile();
    const calibrationConfigValid = Number.isFinite(calib.headingOffsetDeg) && Math.abs(asFiniteNumber(calib.headingSign, 0)) === 1 && Math.abs(asFiniteNumber(calib.steeringSign, 0)) === 1;
    const calibrationRuntimeReady = calib.motionSampleCount >= CALIBRATION_MIN_MOTION_SAMPLES && calib.headingConfidence >= CALIBRATION_APPLY_CONFIDENCE;
    if (!calibrationState.enabled || !calibrationConfigValid) {
      updateCheck("CALIBRATION_READY", "FAIL", "Calibration pipeline is disabled or invalid.", {
        enabled: !!calibrationState.enabled,
        headingSign: calib.headingSign,
        steeringSign: calib.steeringSign,
        headingOffsetDeg: calib.headingOffsetDeg
      });
    } else if (calibrationRuntimeReady) {
      updateCheck("CALIBRATION_READY", "PASS", `Calibration ready (conf=${(calib.headingConfidence * 100).toFixed(1)}%, samples=${calib.motionSampleCount}).`, {
        headingConfidence: calib.headingConfidence,
        motionSampleCount: calib.motionSampleCount
      });
    } else {
      updateCheck("CALIBRATION_READY", "PASS", `Calibration bootstrap valid; online refinement continues (conf=${(calib.headingConfidence * 100).toFixed(1)}%, samples=${calib.motionSampleCount}).`, {
        headingConfidence: calib.headingConfidence,
        motionSampleCount: calib.motionSampleCount,
        bootstrapMode: true
      });
    }

    const makePreflightSmoothingState = () => ({
      lastSteering: 0,
      lastStrategyMode: "",
      noContactMs: 0,
      noContactCycles: 0,
      lastContactAt: 0,
      lastTickAt: 0,
      reacquireTurnDir: 1,
      lastReacquireFlipAt: 0,
      lastOutcomeSummary: "",
      lastOutcomeDetails: null,
      lastReflectionHint: "",
      lastSkillName: "",
      modeHoldRemaining: 0,
      targetLockHoldRemaining: 0
    });
    const scenarioCatalog = {
      LOCK_OPEN_APPROACH: {
        name: "LOCK_OPEN_APPROACH",
        sensor: {
          left: 9, leftDiag: 9, front: 9, rightDiag: 9, right: 9, back: 9, backLeft: 9, backRight: 9,
          worldX: 0, worldZ: 0, headingDeg: 0, targetHits: {},
          angleToTarget: 0, distanceToTarget: 8, speed: 0.2, isStuck: false, moveDir: "IDLE", blockedDist: 99
        },
        expect: (decision) => decision?.strategy?.mode === "TARGET_LOCK" && (decision?.skill?.name === "APPROACH_TARGET")
      },
      ESCAPE_FRONT_BLOCKED: {
        name: "ESCAPE_FRONT_BLOCKED",
        sensor: {
          left: 2.2, leftDiag: 1.7, front: 1.4, rightDiag: 1.6, right: 2.1, back: 6.0, backLeft: 5.0, backRight: 5.2,
          worldX: 0, worldZ: 0, headingDeg: 0, targetHits: {},
          angleToTarget: 5, distanceToTarget: 10, speed: 0.1, isStuck: true, moveDir: "IDLE", blockedDist: 1.4
        },
        expect: (decision) => decision?.strategy?.mode === "ESCAPE_RECOVERY" || decision?.skill?.name === "BACKOFF_AND_TURN"
      },
      EXPLORE_NO_TARGET: {
        name: "EXPLORE_NO_TARGET",
        sensor: {
          left: 9.8, leftDiag: 9.8, front: 9.8, rightDiag: 9.8, right: 9.8, back: 9.8, backLeft: 9.8, backRight: 9.8,
          worldX: 0, worldZ: 0, headingDeg: 0, targetHits: {},
          angleToTarget: 155, distanceToTarget: 60, speed: 0.1, isStuck: false, moveDir: "IDLE", blockedDist: 99
        },
        expect: (decision) => decision?.strategy?.mode === "MEMORY_EXPLORE"
      }
    };
    const preflightScenarios = PREFLIGHT_AI_SCENARIOS
      .map((name) => scenarioCatalog[name])
      .filter(Boolean);
    const aiScenarioResults = [];
    let parseSuccessCount = 0;
    let alignmentCount = 0;
    const aiModes = new Set();
    const aiSkills = new Set();

    for (const scenario of preflightScenarios) {
      let scenarioParseOk = false;
      let scenarioAligned = false;
      let lastMode = "ERROR";
      let lastSkill = "ERROR";
      let lastParseMethod = "not_called";
      let lastLatency = 0;
      let lastError = "";
      let attemptsUsed = 0;

      for (let attempt = 1; attempt <= PREFLIGHT_AI_RETRY_PER_SCENARIO; attempt += 1) {
        attemptsUsed = attempt;
        try {
          const decision = await getDrivingDecision(
            scenario.sensor,
            ["PREFLIGHT"],
            preflightModel,
            makePreflightSmoothingState(),
            null
          );
          lastParseMethod = String(decision?.parseMethod || "");
          const parseOk = decision?.action !== "ERROR" && !lastParseMethod.startsWith("api_") && !lastParseMethod.startsWith("unparseable_");
          lastMode = decision?.strategy?.mode || "UNKNOWN";
          lastSkill = decision?.skill?.name || "UNKNOWN";
          lastLatency = asFiniteNumber(decision?.latency, 0);
          aiModes.add(String(lastMode));
          aiSkills.add(String(lastSkill));

          if (parseOk) scenarioParseOk = true;
          if (parseOk && scenario.expect(decision)) scenarioAligned = true;

          if (scenarioParseOk && scenarioAligned) break;
        } catch (err) {
          lastParseMethod = "exception";
          lastError = err?.message || "unknown";
        }
      }

      if (scenarioParseOk) parseSuccessCount += 1;
      if (scenarioAligned) alignmentCount += 1;
      aiScenarioResults.push({
        name: scenario.name,
        parseMethod: lastParseMethod,
        parseOk: scenarioParseOk,
        mode: lastMode,
        skill: lastSkill,
        aligned: scenarioAligned,
        latency: lastLatency,
        attemptsUsed,
        attemptsMax: PREFLIGHT_AI_RETRY_PER_SCENARIO,
        error: lastError || undefined
      });
      setPreflightReport((prev) => ({ ...prev, aiScenarioResults: [...aiScenarioResults] }));
    }

    if (parseSuccessCount === preflightScenarios.length) {
      updateCheck("AI_CONNECTIVITY", "PASS", `AI responded in ${parseSuccessCount}/${preflightScenarios.length} scenarios.`, { parseSuccessCount });
    } else {
      updateCheck("AI_CONNECTIVITY", "FAIL", `AI parse/connectivity unstable (${parseSuccessCount}/${preflightScenarios.length}).`, { parseSuccessCount });
    }

    if (alignmentCount === preflightScenarios.length) {
      updateCheck("AI_STRATEGY_ALIGNMENT", "PASS", `Scenario alignment ${alignmentCount}/${preflightScenarios.length}.`, { alignmentCount });
    } else {
      updateCheck("AI_STRATEGY_ALIGNMENT", "FAIL", `Scenario alignment too low (${alignmentCount}/${preflightScenarios.length}).`, { alignmentCount });
    }

    const modeDiversity = aiModes.size;
    const skillDiversity = aiSkills.size;
    if (modeDiversity >= 2 && skillDiversity >= 2) {
      updateCheck("AI_MODE_SKILL_DIVERSITY", "PASS", `Mode diversity=${modeDiversity}, skill diversity=${skillDiversity}.`, { modeDiversity, skillDiversity });
    } else if (modeDiversity >= 1 && skillDiversity >= 1) {
      updateCheck("AI_MODE_SKILL_DIVERSITY", "PASS", `Low diversity but valid (modes=${modeDiversity}, skills=${skillDiversity}).`, { modeDiversity, skillDiversity, monitorOnly: true });
    } else {
      updateCheck("AI_MODE_SKILL_DIVERSITY", "FAIL", "AI returned invalid mode/skill diversity.", { modeDiversity, skillDiversity });
    }

    const blockingFailures = report.checks.filter((check) => check.blocking && check.status === "FAIL").length;
    const warningCount = report.checks.filter((check) => check.status === "WARN").length;
    const nonPassCount = report.checks.filter((check) => check.status !== "PASS").length;
    const hasFail = report.checks.some((check) => check.status === "FAIL");
    const overall = nonPassCount === 0 ? "PASS" : (hasFail ? "FAIL" : "WARN");
    report = {
      ...report,
      status: "DONE",
      overall,
      finishedAt: new Date().toISOString(),
      aiScenarioResults,
      blockingFailures,
      warningCount,
      nonPassCount,
      summary: nonPassCount === 0
        ? "Preflight passed. All checks clear."
        : `Preflight not clear (${nonPassCount} check(s) are not PASS).`
    };
    setPreflightReport(report);
    return report;
  }, [selectedModel]);

  const startAutodriveWithPreflight = useCallback(async (trigger = "manual", modelOverride = null) => {
    if (isAnalyzing || isPreflightRunning || autoDrive) {
      return {
        started: false,
        preflightExecuted: false,
        preflightReport: null,
        reason: "Start rejected: drive/preflight/analyze is already active.",
        cooldownMsRemaining: 0
      };
    }
    const nowMs = Date.now();
    if (preflightBlockGuardRef.current.blockedUntil > nowMs) {
      const waitMs = preflightBlockGuardRef.current.blockedUntil - nowMs;
      const waitSec = Math.max(1, Math.ceil(waitMs / 1000));
      const cooldownReason = `Preflight retry paused (${waitSec}s cooldown): ${preflightBlockGuardRef.current.reason || "recent repeated failure"}`;
      setErrorMsg(cooldownReason);
      return {
        started: false,
        preflightExecuted: false,
        preflightReport: preflightReportRef.current || null,
        reason: cooldownReason,
        cooldownMsRemaining: waitMs
      };
    }
    applyControls({ throttle: 0, steering: 0 });
    setErrorMsg(null);
    const waitForFreshSensorTick = async (sensorTickBeforeReset, timeoutMs = 3000) => {
      const started = Date.now();
      while ((Date.now() - started) < timeoutMs) {
        if (sensorTickRef.current > sensorTickBeforeReset) return true;
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
      return sensorTickRef.current > sensorTickBeforeReset;
    };
    const waitForPostResetStableSensor = async (timeoutMs = 4200, pollMs = 80) => {
      const started = Date.now();
      let lastFailedChecks = ["sensor_stream"];
      let stablePassCount = 0;
      while ((Date.now() - started) < timeoutMs) {
        const sensor = sensorRef.current || {};
        const x = asFiniteNumber(sensor.worldX, Number.NaN);
        const y = asFiniteNumber(sensor.worldY, Number.NaN);
        const z = asFiniteNumber(sensor.worldZ, Number.NaN);
        const speedAbs = Math.abs(asFiniteNumber(sensor.speed, Number.NaN));
        const verticalSpeedAbs = Math.abs(asFiniteNumber(sensor.verticalSpeed, Number.NaN));
        const spawnX = asFiniteNumber(DEFAULT_CAR_SPAWN.position?.[0], 0);
        const spawnZ = asFiniteNumber(DEFAULT_CAR_SPAWN.position?.[2], -10);
        const distFromSpawn = Number.isFinite(x) && Number.isFinite(z)
          ? Math.hypot(x - spawnX, z - spawnZ)
          : Number.POSITIVE_INFINITY;

        const failedChecks = [];
        if (!(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z))) failedChecks.push("sensor_stream");
        if (!(Number.isFinite(distFromSpawn) && distFromSpawn <= RUN_START_SPAWN_TOLERANCE_M)) failedChecks.push("nearSpawn");
        if (!(Number.isFinite(y) && y >= RUN_START_MIN_WORLD_Y && y <= RUN_START_MAX_WORLD_Y)) failedChecks.push("grounded");
        if (!(Number.isFinite(verticalSpeedAbs) && verticalSpeedAbs <= RUN_START_MAX_VERTICAL_SPEED)) failedChecks.push("gravityStable");
        if (!(Number.isFinite(speedAbs) && speedAbs <= RUN_START_MAX_SPEED)) failedChecks.push("speedReady");

        if (failedChecks.length === 0) {
          stablePassCount += 1;
          if (stablePassCount < 2) {
            await new Promise((resolve) => setTimeout(resolve, pollMs));
            continue;
          }
          return {
            ok: true,
            waitedMs: Date.now() - started,
            failedChecks: [],
            sensorSnapshot: {
              worldX: Number(x.toFixed(3)),
              worldY: Number(y.toFixed(3)),
              worldZ: Number(z.toFixed(3)),
              speed: Number(speedAbs.toFixed(3)),
              verticalSpeed: Number(verticalSpeedAbs.toFixed(3)),
              distanceFromSpawn: Number(distFromSpawn.toFixed(3))
            }
          };
        }
        stablePassCount = 0;
        lastFailedChecks = failedChecks;
        await new Promise((resolve) => setTimeout(resolve, pollMs));
      }
      return {
        ok: false,
        waitedMs: timeoutMs,
        failedChecks: lastFailedChecks,
        sensorSnapshot: sensorRef.current || null
      };
    };

    const runResetReadinessGate = async (maxResetAttempts = 2) => {
      let lastGate = {
        ok: false,
        reasonCode: "sensor_refresh_timeout",
        failedChecks: ["sensor_stream"],
        sensorSnapshot: sensorRef.current || null
      };
      for (let resetAttempt = 1; resetAttempt <= maxResetAttempts; resetAttempt += 1) {
        const sensorTickBeforeReset = sensorTickRef.current;
        prepareSessionForStart();
        const sensorRefreshed = await waitForFreshSensorTick(sensorTickBeforeReset, 3000);
        if (!sensorRefreshed) {
          lastGate = {
            ok: false,
            reasonCode: "sensor_refresh_timeout",
            failedChecks: ["sensor_refresh_timeout"],
            sensorSnapshot: sensorRef.current || null
          };
        } else {
          const stableSensor = await waitForPostResetStableSensor(4200, 80);
          if (stableSensor.ok) {
            return {
              ok: true,
              reasonCode: "",
              failedChecks: [],
              sensorSnapshot: stableSensor.sensorSnapshot
            };
          }
          lastGate = {
            ok: false,
            reasonCode: "post_reset_unstable",
            failedChecks: stableSensor.failedChecks || ["post_reset_unstable"],
            sensorSnapshot: stableSensor.sensorSnapshot || sensorRef.current || null
          };
        }
        if (resetAttempt < maxResetAttempts) {
          await new Promise((resolve) => setTimeout(resolve, 160));
        }
      }
      return lastGate;
    };

    const resetGate = await runResetReadinessGate(2);
    if (!resetGate.ok) {
      const failedText = Array.isArray(resetGate.failedChecks) && resetGate.failedChecks.length > 0
        ? resetGate.failedChecks.join(", ")
        : "unknown";
      const reasonText = resetGate.reasonCode === "sensor_refresh_timeout"
        ? "Preflight blocked: sensor refresh timeout after run reset."
        : `Preflight blocked: post-reset vehicle state not stable (${failedText}).`;

      const reportTemplate = createInitialPreflightState();
      const syntheticReport = {
        ...reportTemplate,
        status: "DONE",
        overall: "FAIL",
        finishedAt: new Date().toISOString(),
        summary: "Preflight not clear (1 check(s) are not PASS).",
        checks: reportTemplate.checks.map((check) => {
          if (check.id === "CAR_PHYSICS_READY") {
            return {
              ...check,
              status: "FAIL",
              detail: resetGate.reasonCode === "sensor_refresh_timeout"
                ? "Sensor refresh timed out after reset."
                : `Vehicle start state invalid: ${failedText}.`,
              metric: resetGate.sensorSnapshot || null
            };
          }
          return {
            ...check,
            status: "PASS",
            detail: "Pre-check gate passed."
          };
        }),
        aiScenarioResults: [],
        blockingFailures: 1,
        warningCount: 0,
        nonPassCount: 1
      };
      setPreflightReport(syntheticReport);
      recordAndDownloadPreflightGateLog(
        syntheticReport,
        trigger,
        resetGate.reasonCode === "sensor_refresh_timeout" ? "SENSOR_REFRESH_TIMEOUT" : "POST_RESET_UNSTABLE"
      );

      preflightBlockGuardRef.current.blockedUntil = Date.now() + PREFLIGHT_BLOCK_RETRY_COOLDOWN_MS;
      preflightBlockGuardRef.current.repeatCount += 1;
      preflightBlockGuardRef.current.reason = resetGate.reasonCode === "sensor_refresh_timeout"
        ? "sensor refresh timeout"
        : `post-reset unstable (${failedText})`;
      setErrorMsg(reasonText);
      return {
        started: false,
        preflightExecuted: false,
        preflightReport: syntheticReport,
        reason: reasonText,
        cooldownMsRemaining: PREFLIGHT_BLOCK_RETRY_COOLDOWN_MS
      };
    }

    try {
      setIsPreflightRunning(true);
      const report = await runStartupPreflight(trigger, modelOverride);
      const checks = Array.isArray(report?.checks) ? report.checks : [];
      const allPass = checks.length > 0 && checks.every((check) => check.status === "PASS");
      if (!allPass) {
        const blockedLog = recordAndDownloadPreflightGateLog(report, trigger, "NOT_ALL_PASS");
        const detailSummary = getPreflightNonPassSummary(report, 2);
        const reasonText = `Preflight blocked: all checks must PASS (${blockedLog.nonPassCount} non-pass). ${detailSummary}`;
        const isGroundGravityFail = isGroundGravityPreflightFailure(report);
        if (isGroundGravityFail) {
          preflightBlockGuardRef.current.blockedUntil = Date.now() + PREFLIGHT_BLOCK_RETRY_COOLDOWN_MS;
          preflightBlockGuardRef.current.repeatCount += 1;
          preflightBlockGuardRef.current.reason = "CAR_PHYSICS_READY failed (ground/gravity).";
        } else {
          preflightBlockGuardRef.current.blockedUntil = 0;
          preflightBlockGuardRef.current.repeatCount = 0;
          preflightBlockGuardRef.current.reason = "";
        }
        setErrorMsg(reasonText);
        return {
          started: false,
          preflightExecuted: true,
          preflightReport: report,
          reason: reasonText,
          cooldownMsRemaining: isGroundGravityFail ? PREFLIGHT_BLOCK_RETRY_COOLDOWN_MS : 0
        };
      }
      preflightBlockGuardRef.current.blockedUntil = 0;
      preflightBlockGuardRef.current.repeatCount = 0;
      preflightBlockGuardRef.current.reason = "";
      setAutoDrive(true);
      return {
        started: true,
        preflightExecuted: true,
        preflightReport: report,
        reason: "",
        cooldownMsRemaining: 0
      };
    } catch (err) {
      const exceptionPayload = recordAndDownloadPreflightGateLog({
        ...createInitialPreflightState(),
        status: "ERROR",
        overall: "FAIL",
        summary: `Preflight exception: ${err?.message || "unknown error"}`,
        checks: []
      }, trigger, "PREFLIGHT_EXCEPTION");
      setErrorMsg(`Preflight exception: ${exceptionPayload.preflightSummary}`);
      return {
        started: false,
        preflightExecuted: true,
        preflightReport: null,
        reason: `Preflight exception: ${exceptionPayload.preflightSummary}`,
        cooldownMsRemaining: 0
      };
    } finally {
      setIsPreflightRunning(false);
    }
  }, [
    isAnalyzing,
    isPreflightRunning,
    autoDrive,
    applyControls,
    prepareSessionForStart,
    runStartupPreflight,
    getPreflightNonPassSummary,
    isGroundGravityPreflightFailure
  ]);


  // Game State
  const [score, setScore] = useState(0);
  const [targetPosition, setTargetPosition] = useState([...DEFAULT_TARGET_POSITION]); // Start target
  const [carResetNonce, setCarResetNonce] = useState(0);

  // Recovery State
  const [recoveryPhase, setRecoveryPhase] = useState(null); // null | "REVERSING" | "TURNING"
  const lastRecoveryTime = useRef(0); // Cooldown tracker
  const keysPressed = useRef({}); // Manual Control State

  // Randomize target
  const spawnNewTarget = useCallback(() => {
    // Simple random within -15 to 15 range
    const x = (Math.random() - 0.5) * 30;
    const z = (Math.random() - 0.5) * 30;
    setTargetPosition([x, 1, z]);
    setScore(s => s + 100);
  }, []);

  const startRecovery = useCallback(() => {
    // Check cooldown (e.g. 3 seconds)
    const now = Date.now();
    if (now - lastRecoveryTime.current < 3000) return;

    lastRecoveryTime.current = now;
    setRecoveryPhase("REVERSING");
    setAction("REVERSE");

    // Phase 1: Reverse for 1.5s
    setTimeout(() => {
      setRecoveryPhase("TURNING");
      setAction(Math.random() > 0.5 ? "LEFT" : "RIGHT");

      // Phase 2: Turn for 1.0s
      setTimeout(() => {
        setRecoveryPhase(null);
        // Update timestamp so loop knows we exited
        lastRecoveryTime.current = Date.now();
      }, 1000);
    }, 1500);
  }, []);

  // Throttle sensor updates to avoid React render spam
  const sensorRef = useRef(sensorData);
  const updateSensorData = useCallback((data) => {
    const now = Date.now();
    sensorTickRef.current += 1;
    const calibrationState = directionCalibrationRef.current;
    calibrationState.samples.push({
      t: now,
      x: asFiniteNumber(data?.worldX, 0),
      z: asFiniteNumber(data?.worldZ, 0),
      heading: asFiniteNumber(data?.headingDeg, 0),
      steering: asFiniteNumber(controlRef.current?.steering, 0),
      throttle: asFiniteNumber(controlRef.current?.throttle, 0),
      speed: asFiniteNumber(data?.speed, 0)
    });
    if (calibrationState.samples.length > CALIBRATION_MAX_HISTORY) {
      calibrationState.samples.splice(0, calibrationState.samples.length - CALIBRATION_MAX_HISTORY);
    }
    if (calibrationState.enabled && (now - calibrationState.lastUpdateAt) >= CALIBRATION_UPDATE_INTERVAL_MS) {
      calibrationState.lastUpdateAt = now;
      recomputeDirectionCalibration();
    }

    const calibratedData = applyDirectionCalibrationToSensor(data, calibrationState.profile);
    sensorRef.current = calibratedData;
    explorationMemoryRef.current.update(calibratedData);

    // Check if reached target
    if (calibratedData.distanceToTarget < 2.5) {
      if (now - targetCaptureRef.current.lastCaptureAt > 800) {
        targetCaptureRef.current.count += 1;
        targetCaptureRef.current.lastCaptureAt = now;
        spawnNewTarget();
      }
    }

    // Stuck check is handled in effect
  }, [spawnNewTarget, recomputeDirectionCalibration]);

  // Recovery Trigger Logic (Separate Effect to monitor sensorRef)
  useEffect(() => {
    if (!autoDrive || recoveryPhase) return;

    const checkStuckInterval = setInterval(() => {
      // While waiting for AI response, intentional stop should not trigger recovery.
      if (aiMetaRef.current.isThinking) return;
      if (sensorRef.current.isStuck) {
        startRecovery();
      }
    }, 500); // Check every 500ms

    return () => clearInterval(checkStuckInterval);
  }, [autoDrive, recoveryPhase, startRecovery]);

  // Realtime exploration-memory map snapshot.
  useEffect(() => {
    const interval = setInterval(() => {
      const snapshot = explorationMemoryRef.current.getVisualization(sensorRef.current, {
        radiusCells: 8,
        maxFrontier: 8,
        maxRisky: 6
      });
      setMemoryViz(snapshot);
    }, 250);

    return () => clearInterval(interval);
  }, []);

  // Telemetry History State
  const [telemetry, setTelemetry] = useState([]);

  useEffect(() => {
    telemetryRef.current = telemetry;
  }, [telemetry]);

  // UI Update Loop
  useEffect(() => {
    if (!autoDrive) return;

    sessionRef.current.startAt = Date.now();
    sessionRef.current.lastDistance = null;

    const interval = setInterval(() => {
      setSensorData(() => {
        const current = sensorRef.current;
        const controlsNow = controlRef.current;
        const hits = current.targetHits || {};
        const targetHitCount = Object.values(hits).filter(Boolean).length;
        const targetContact = targetHitCount > 0;
        const explorationContext = explorationMemoryRef.current.getContext(current);

        // --- TELEMETRY RECORDING ---
        // 1) Goal alignment signal
        const intentionality = Math.cos(current.angleToTarget * Math.PI / 180);
        const previousDistance = sessionRef.current.lastDistance ?? current.distanceToTarget ?? 0;
        const currentDistance = current.distanceToTarget ?? previousDistance;
        const progressDelta = previousDistance - currentDistance;
        sessionRef.current.lastDistance = currentDistance;

        // 2) Environment pressure summary
        const obstacleDistances = [
          current.front,
          current.leftDiag,
          current.rightDiag,
          current.left,
          current.right,
          current.back,
          current.backLeft,
          current.backRight
        ].map(v => typeof v === "number" ? v : 10);
        const minObstacleDist = Math.min(...obstacleDistances);
        const now = Date.now();
        const collisionSnapshot = snapshotCollisionStats(collisionStatsRef.current);
        const calibrationSnapshot = directionCalibrationRef.current.profile;
        const reasonStatsSnapshot = reasonValidationStatsRef.current || createInitialReasonValidationStats();
        const reasonPassRate = reasonStatsSnapshot.totalSteps > 0
          ? reasonStatsSnapshot.passedSteps / reasonStatsSnapshot.totalSteps
          : 1;

        // 3) Full telemetry point
        const newPoint = {
          time: now,
          worldX: current.worldX ?? 0,
          worldY: current.worldY ?? 0,
          worldZ: current.worldZ ?? 0,
          headingDeg: current.headingDeg ?? 0,
          targetAngle: current.angleToTarget ?? 0,
          distanceToTarget: currentDistance,
          progressDelta,
          steering: controlsNow.steering ?? 0,
          throttle: controlsNow.throttle ?? 0,
          intentionality,
          speed: current.speed ?? 0,
          verticalSpeed: current.verticalSpeed ?? 0,
          grounded: !!current.grounded,
          sensorRange: current.sensorRange ?? 10,
          isStuck: !!current.isStuck,
          moveDir: current.moveDir || "IDLE",
          blockedDist: current.blockedDist ?? 99,
          front: current.front ?? 10,
          leftDiag: current.leftDiag ?? 10,
          rightDiag: current.rightDiag ?? 10,
          left: current.left ?? 10,
          right: current.right ?? 10,
          back: current.back ?? 10,
          backLeft: current.backLeft ?? 10,
          backRight: current.backRight ?? 10,
          minObstacleDist,
          targetHits: hits,
          targetHitCount,
          targetContact,
          targetsReached: targetCaptureRef.current.count,
          recoveryPhase: recoveryPhase || "NONE",
          aiLatencyMs: aiMetaRef.current.latency || 0,
          aiReasonCode: aiMetaRef.current.reasonCode || "",
          aiReasonSource: aiMetaRef.current.reasonSource || "none",
          aiReasonValidationLast: aiMetaRef.current.reasonValidationLast || "NONE",
          aiReasonBlockedTotal: aiMetaRef.current.reasonBlockedTotal ?? 0,
          aiReasonPassRate: aiMetaRef.current.reasonPassRate ?? reasonPassRate,
          aiStrategyMode: aiMetaRef.current.strategyMode || "UNKNOWN",
          aiStrategyTransition: aiMetaRef.current.strategyTransition || "HOLD",
          aiStrategySector: aiMetaRef.current.strategySector || "F",
          aiStrategyConfidence: aiMetaRef.current.strategyConfidence ?? 0,
          aiSkillName: aiMetaRef.current.skillName || "UNKNOWN",
          aiSkillIntensity: aiMetaRef.current.skillIntensity ?? 0,
          aiParseMethod: aiMetaRef.current.parseMethod || "",
          aiParseRecovered: !!aiMetaRef.current.parseRecovered,
          aiModel: aiMetaRef.current.model || "",
          aiThinking: !!aiMetaRef.current.isThinking,
          aiSource: aiMetaRef.current.source || "AI",
          safetyOverride: !!aiMetaRef.current.safetyOverride,
          overrideReason: aiMetaRef.current.overrideReason || "",
          decisionAgeMs: aiMetaRef.current.lastDecisionAt ? now - aiMetaRef.current.lastDecisionAt : null,
          memoryNoGoRatio: aiMetaRef.current.memoryNoGoRatio ?? null,
          memoryRevisitRate: aiMetaRef.current.memoryRevisitRate ?? null,
          memoryCurrentWeight: aiMetaRef.current.memoryCurrentWeight ?? null,
          memorySelectedWeight: aiMetaRef.current.memorySelectedWeight ?? null,
          memorySelectedNoGo: !!aiMetaRef.current.memorySelectedNoGo,
          memorySelectedSector: aiMetaRef.current.memorySelectedSector || "F",
          memorySelectionReason: aiMetaRef.current.memorySelectionReason || "",
          directionCooldownApplied: !!aiMetaRef.current.directionCooldownApplied,
          directionCooldownRemainingMs: aiMetaRef.current.directionCooldownRemainingMs ?? 0,
          directionLastSign: aiMetaRef.current.directionLastSign ?? 0,
          collisionCount: collisionSnapshot.totalCount,
          sameWallCollisionCount: collisionSnapshot.sameWallRepeatCount,
          sameWallConsecutiveCollisionCount: collisionSnapshot.sameWallConsecutiveRepeatCount,
          collisionLastRegion: collisionSnapshot.lastRegion,
          collisionLastAt: collisionSnapshot.lastCollisionAt || null,
          collisionOuterNorthCount: collisionSnapshot.byRegion.OUTER_NORTH,
          collisionOuterSouthCount: collisionSnapshot.byRegion.OUTER_SOUTH,
          collisionOuterEastCount: collisionSnapshot.byRegion.OUTER_EAST,
          collisionOuterWestCount: collisionSnapshot.byRegion.OUTER_WEST,
          collisionInnerObstacleCount: collisionSnapshot.byRegion.INNER_OBSTACLE,
          collisionOutsideBoundsCount: collisionSnapshot.byRegion.OUTSIDE_BOUNDS,
          calibrationApplied: !!calibrationSnapshot.applied,
          calibrationHeadingSign: calibrationSnapshot.headingSign ?? 1,
          calibrationHeadingOffsetDeg: calibrationSnapshot.headingOffsetDeg ?? 0,
          calibrationHeadingConfidence: calibrationSnapshot.headingConfidence ?? 0,
          calibrationSteeringSign: calibrationSnapshot.steeringSign ?? 1,
          calibrationSteeringConfidence: calibrationSnapshot.steeringConfidence ?? 0,
          reasonTotalSteps: reasonStatsSnapshot.totalSteps,
          reasonPassedSteps: reasonStatsSnapshot.passedSteps,
          reasonBlockedSteps: reasonStatsSnapshot.blockedSteps,
          reasonMissingModelReasonSteps: reasonStatsSnapshot.missingModelReasonSteps,
          reasonSignMismatchSteps: reasonStatsSnapshot.signMismatchSteps
        };

        if (explorationContext) {
          newPoint.memoryLoopRate = explorationContext.loopRate ?? 0;
          newPoint.memoryRecommendedSector = explorationContext.preferredSector || "UNKNOWN";
          newPoint.memoryMappedCells = explorationContext.memoryStats?.mappedCells ?? 0;
          newPoint.memorySensorRange = explorationContext.memoryStats?.sensorRange ?? (current.sensorRange ?? 10);
          newPoint.memoryCurrentCellVisits = explorationContext.currentCell?.visits ?? 0;
          newPoint.memoryCandidateCount = explorationContext.diagnostics?.candidateCount ?? 0;
          newPoint.memorySafeCandidateCount = explorationContext.diagnostics?.safeCandidateCount ?? 0;
          newPoint.memoryNoGoCandidateCount = explorationContext.diagnostics?.noGoCandidateCount ?? 0;
        }

        setTelemetry(prev => [...prev, newPoint]);

        return { ...current };
      });
    }, 100);
    return () => clearInterval(interval);
  }, [autoDrive, recoveryPhase]);

  // Manual Control Loop (when AutoDrive is OFF)
  useEffect(() => {
    if (autoDrive) return;

    // keysPressed is now top-level ref


    const updateControls = () => {
      // Manual Control Overlay (Digital to Analog conversion)
      let throttle = 0;
      let steering = 0;
      let action = "IDLE";

      if (keysPressed.current["ArrowUp"] || keysPressed.current["w"]) { throttle = 1; action = "FORWARD"; }
      else if (keysPressed.current["ArrowDown"] || keysPressed.current["s"]) { throttle = -1; action = "REVERSE"; }

      if (keysPressed.current["ArrowLeft"] || keysPressed.current["a"]) { steering = 1; action = action === "IDLE" ? "LEFT" : action; }
      else if (keysPressed.current["ArrowRight"] || keysPressed.current["d"]) { steering = -1; action = action === "IDLE" ? "RIGHT" : action; }

      // console.log("Manual Analog:", throttle, steering);
      applyControls({ throttle, steering });
      setAction(action);
    };

    const handleKeyDown = (e) => {
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].indexOf(e.code) > -1) {
        e.preventDefault();
      }
      keysPressed.current[e.key] = true;
      updateControls();
    };

    const handleKeyUp = (e) => {
      keysPressed.current[e.key] = false;
      updateControls();
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [autoDrive, applyControls]);

  // Debug State
  const [aiStats, setAiStats] = useState({ latency: 0, raw: "", thought: "" });

  const decisionLog = useRef([]); // Full Session Log
  const smoothingRef = useRef({
    lastSteering: 0,
    lastStrategyMode: "",
    noContactMs: 0,
    noContactCycles: 0,
    lastContactAt: 0,
    lastTickAt: 0,
    reacquireTurnDir: 1,
    lastReacquireFlipAt: 0,
    lastOutcomeSummary: "",
    lastOutcomeDetails: null,
    lastReflectionHint: "",
    lastSkillName: "",
    modeHoldRemaining: 0,
    targetLockHoldRemaining: 0
  }); // Persist smoothing state
  const directionFlipRef = useRef({
    lastSign: 0,
    lastSignAt: 0
  });
  const finalizePendingDecisionOutcome = useCallback((sensorSnapshot, explorationContext, nowMs = Date.now()) => {
    const previousDecision = decisionLog.current.length > 0
      ? decisionLog.current[decisionLog.current.length - 1]
      : null;
    if (!previousDecision || previousDecision.outcome) return null;

    const outcome = buildDecisionOutcome(previousDecision, sensorSnapshot, explorationContext, nowMs);
    if (!outcome) return null;

    previousDecision.outcome = outcome;
    smoothingRef.current.lastOutcomeSummary = outcome.summary;
    smoothingRef.current.lastOutcomeDetails = outcome;
    return outcome;
  }, []);

  const formatStamp = (ts) => new Date(ts).toISOString().replace(/[:.]/g, "-");
  const buildSessionPrefix = useCallback(() => {
    const startAt = sessionRef.current?.startAt || Date.now();
    return `session_${formatStamp(startAt)}`;
  }, []);

  const triggerDownload = useCallback((content, fileName, mime = "application/octet-stream") => {
    const payload = typeof content === "string" ? content : JSON.stringify(content ?? {}, null, 2);
    const blob = new Blob([payload], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  const recordAndDownloadPreflightGateLog = useCallback((report, trigger = "manual", reason = "NOT_ALL_PASS") => {
    const safeTrigger = String(trigger || "manual").toLowerCase().replace(/[^a-z0-9_-]/g, "_");
    const nowMs = Date.now();
    const nowStamp = formatStamp(nowMs);
    const prefix = buildSessionPrefix();
    const checks = Array.isArray(report?.checks) ? report.checks : [];
    const nonPassChecks = checks
      .filter((check) => check.status !== "PASS")
      .map((check) => ({
        id: check.id,
        label: check.label,
        area: check.area,
        status: check.status,
        detail: check.detail || "",
        metric: check.metric ?? null,
        blocking: !!check.blocking
      }));

    const payload = {
      gatePolicy: PREFLIGHT_GATE_POLICY,
      suggestedFolder: PREFLIGHT_FAILURE_LOG_FOLDER,
      createdAt: new Date(nowMs).toISOString(),
      trigger: safeTrigger,
      reason,
      sessionPrefix: prefix,
      model: selectedModel,
      preflightSummary: report?.summary || "Preflight gate blocked.",
      preflightOverall: report?.overall || "FAIL",
      preflightStatus: report?.status || "DONE",
      nonPassCount: nonPassChecks.length,
      nonPassChecks,
      aiScenarioResults: Array.isArray(report?.aiScenarioResults) ? report.aiScenarioResults : [],
      latestSensor: sensorRef.current || null,
      directionCalibration: directionCalibrationRef.current?.profile || null
    };

    preflightGateLogsRef.current.push(payload);
    if (preflightGateLogsRef.current.length > 200) {
      preflightGateLogsRef.current = preflightGateLogsRef.current.slice(-200);
    }

    const fileName = `${prefix}_${PREFLIGHT_FAILURE_LOG_FOLDER}_preflight_gate_${nowStamp}_${safeTrigger}.json`;
    triggerDownload(JSON.stringify(payload, null, 2), fileName, "application/json");
    return payload;
  }, [buildSessionPrefix, selectedModel, triggerDownload]);

  const downloadLatestPreflightGateLog = useCallback((trigger = "manual") => {
    const latest = preflightGateLogsRef.current.length > 0
      ? preflightGateLogsRef.current[preflightGateLogsRef.current.length - 1]
      : null;
    if (!latest) {
      setErrorMsg("No preflight gate failure log is available yet.");
      return;
    }
    const safeTrigger = String(trigger || "manual").toLowerCase().replace(/[^a-z0-9_-]/g, "_");
    const nowStamp = formatStamp(Date.now());
    const prefix = buildSessionPrefix();
    triggerDownload(
      JSON.stringify(latest, null, 2),
      `${prefix}_${PREFLIGHT_FAILURE_LOG_FOLDER}_latest_${nowStamp}_${safeTrigger}.json`,
      "application/json"
    );
  }, [buildSessionPrefix, triggerDownload]);

  const downloadDriveLogOnly = useCallback((trigger = "manual") => {
    const safeTrigger = String(trigger || "manual").toLowerCase().replace(/[^a-z0-9_-]/g, "_");
    const nowStamp = formatStamp(Date.now());
    const prefix = buildSessionPrefix();
    const driveLogJson = JSON.stringify(decisionLog.current, null, 2);
    triggerDownload(
      driveLogJson,
      `${prefix}_drive_gemma_drive_log_${nowStamp}_${safeTrigger}.json`,
      "application/json"
    );
  }, [buildSessionPrefix, triggerDownload]);

  const downloadTelemetryOnly = useCallback((trigger = "manual") => {
    const safeTrigger = String(trigger || "manual").toLowerCase().replace(/[^a-z0-9_-]/g, "_");
    const nowStamp = formatStamp(Date.now());
    const prefix = buildSessionPrefix();
    const telemetryJson = JSON.stringify(telemetry || [], null, 2);
    triggerDownload(
      telemetryJson,
      `${prefix}_telemetry_${nowStamp}_${safeTrigger}.json`,
      "application/json"
    );
  }, [buildSessionPrefix, telemetry, triggerDownload]);

  const downloadMetaOnly = useCallback((trigger = "manual") => {
    const safeTrigger = String(trigger || "manual").toLowerCase().replace(/[^a-z0-9_-]/g, "_");
    const nowStamp = formatStamp(Date.now());
    const prefix = buildSessionPrefix();
    const history = telemetry || [];
    const collisionSnapshot = snapshotCollisionStats(collisionStatsRef.current);
    const meta = {
      sessionPrefix: prefix,
      exportedAt: new Date().toISOString(),
      trigger: safeTrigger,
      model: selectedModel,
      telemetrySamples: Array.isArray(history) ? history.length : 0,
      decisionLogRecords: decisionLog.current.length,
      collisionEvents: collisionEventsRef.current.length,
      collisionCount: collisionSnapshot.totalCount,
      sameWallCollisionCount: collisionSnapshot.sameWallRepeatCount,
      sameWallConsecutiveCollisionCount: collisionSnapshot.sameWallConsecutiveRepeatCount,
      collisionByRegion: collisionSnapshot.byRegion,
      preflight: preflightReport,
      preflightGatePolicy: PREFLIGHT_GATE_POLICY,
      preflightFailureLogFolder: PREFLIGHT_FAILURE_LOG_FOLDER,
      preflightFailureLogCount: preflightGateLogsRef.current.length,
      latestPreflightFailureLog: preflightGateLogsRef.current.length > 0
        ? preflightGateLogsRef.current[preflightGateLogsRef.current.length - 1]
        : null,
      directionCalibration: directionCalibrationRef.current?.profile || null
    };
    triggerDownload(
      JSON.stringify(meta, null, 2),
      `${prefix}_meta_${nowStamp}_${safeTrigger}.json`,
      "application/json"
    );
  }, [buildSessionPrefix, telemetry, selectedModel, triggerDownload, preflightReport]);

  const handleSaveLogs = useCallback(async (trigger = "manual") => {
    if (isAnalyzing) return;
    const history = telemetry || [];

    setIsAnalyzing(true);
    const nowStamp = formatStamp(Date.now());
    const prefix = buildSessionPrefix();
    const safeTrigger = String(trigger || "manual").toLowerCase().replace(/[^a-z0-9_-]/g, "_");

    try {
      let aiReview = "";
      let htmlContent = "";
      let reportStatus = "not_requested";

      if (Array.isArray(history) && history.length >= 10) {
        try {
          aiReview = await generateAIReview(history, selectedModel);
          htmlContent = buildHTMLReportContent(history, aiReview) || "";
          reportStatus = htmlContent ? "ok" : "empty";
        } catch (reportErr) {
          console.error("HTML report generation failed during bundle save:", reportErr);
          reportStatus = "error";
          htmlContent = "<html><body><h1>Report generation failed.</h1></body></html>";
        }
      } else {
        reportStatus = "insufficient_data";
        htmlContent = "<html><body><h1>Not enough telemetry to generate report.</h1></body></html>";
      }

      const allLogs = {
        sessionPrefix: prefix,
        exportedAt: new Date().toISOString(),
        model: selectedModel,
        telemetrySamples: Array.isArray(history) ? history.length : 0,
        decisionLogRecords: decisionLog.current.length,
        collisionEvents: collisionEventsRef.current.length,
        collisionSummary: snapshotCollisionStats(collisionStatsRef.current),
        preflight: preflightReport,
        preflightGatePolicy: PREFLIGHT_GATE_POLICY,
        preflightFailureLogFolder: PREFLIGHT_FAILURE_LOG_FOLDER,
        preflightFailureLogs: preflightGateLogsRef.current,
        directionCalibration: directionCalibrationRef.current?.profile || null,
        reportStatus,
        aiReview,
        files: {
          driveLog: decisionLog.current,
          telemetry: history,
          collisionEvents: collisionEventsRef.current,
          reportHtml: htmlContent
        }
      };

      triggerDownload(
        JSON.stringify(allLogs, null, 2),
        `${prefix}_all_logs_${nowStamp}_${safeTrigger}.json`,
        "application/json"
      );
    } catch (err) {
      console.error("Save logs error:", err);
      alert("Failed to save one or more log files.");
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing, telemetry, selectedModel, buildSessionPrefix, triggerDownload, preflightReport]);

  const handleDownloadHtmlReport = useCallback(async (trigger = "manual") => {
    if (isAnalyzing) return;
    const history = telemetry || [];
    const nowStamp = formatStamp(Date.now());
    const prefix = buildSessionPrefix();
    const safeTrigger = String(trigger || "manual").toLowerCase().replace(/[^a-z0-9_-]/g, "_");

    setIsAnalyzing(true);
    try {
      if (!Array.isArray(history) || history.length < 10) {
        const fallbackHtml = "<html><body><h1>Not enough telemetry to generate report.</h1></body></html>";
        triggerDownload(
          fallbackHtml,
          `${prefix}_report_driver_limit_report_${nowStamp}_${safeTrigger}_insufficient_data.html`,
          "text/html"
        );
      } else {
        const aiReview = await generateAIReview(history, selectedModel);
        const htmlContent = buildHTMLReportContent(history, aiReview) || "<html><body><h1>Report generation failed</h1></body></html>";
        triggerDownload(
          htmlContent,
          `${prefix}_report_driver_limit_report_${nowStamp}_${safeTrigger}.html`,
          "text/html"
        );
      }
    } catch (err) {
      console.error("HTML Report Export Error:", err);
      const fallbackHtml = "<html><body><h1>Report generation failed.</h1></body></html>";
      triggerDownload(
        fallbackHtml,
        `${prefix}_report_driver_limit_report_${nowStamp}_${safeTrigger}_error.html`,
        "text/html"
      );
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing, telemetry, selectedModel, buildSessionPrefix, triggerDownload]);

  // AI Loop
  useEffect(() => {
    if (!autoDrive) return;

    let isActive = true;

    const loop = async () => {
      if (!isActive) return;

      // 0. Recovery Override
      if (recoveryPhase) {
        // CRITICAL: Actually control the car during recovery!
        if (recoveryPhase === "REVERSING") {
          applyControls({ throttle: -0.8, steering: 0 }); // Reverse straight back
          aiMetaRef.current = {
            ...aiMetaRef.current,
            source: "RECOVERY",
            isThinking: false,
            safetyOverride: true,
            overrideReason: "RECOVERY_REVERSING",
            lastDecisionAt: Date.now()
          };
          setAiStats(prev => ({
            ...prev,
            raw: `[Recovery Phase 1/2] Reversing...`,
            thought: "EMERGENCY: Reversing 1.5s to escape trap!",
            timestamp: new Date().toLocaleTimeString()
          }));
        } else if (recoveryPhase === "TURNING") {
          // Turn toward random direction (set by startRecovery)
          const turnDir = lastAction === "LEFT" ? 1.0 : -1.0;
          applyControls({ throttle: 0.3, steering: turnDir }); // Slow forward + hard turn
          aiMetaRef.current = {
            ...aiMetaRef.current,
            source: "RECOVERY",
            isThinking: false,
            safetyOverride: true,
            overrideReason: "RECOVERY_TURNING",
            lastDecisionAt: Date.now()
          };
          setAiStats(prev => ({
            ...prev,
            raw: `[Recovery Phase 2/2] Turning ${lastAction}...`,
            thought: `EMERGENCY: Turning ${lastAction} 1.0s to reorient!`,
            timestamp: new Date().toLocaleTimeString()
          }));
        }
        setTimeout(loop, 500);
        return;
      }

      // 1. PHASE: OBSERVE (No Brake, continuous movement)
      // setAction("BRAKE"); // Removed to prevent stutter
      // await new Promise(r => setTimeout(r, 200));
      if (!isActive) return;

      // 2. PHASE: THINK
      // Freeze vehicle during AI inference so decision is based on near-static context.
      applyControls({ throttle: 0, steering: 0 });
      setIsThinking(true);
      setErrorMsg(null);
      aiMetaRef.current = {
        ...aiMetaRef.current,
        source: "AI",
        isThinking: true,
        safetyOverride: false,
        overrideReason: ""
      };

      try {
        // Pass recent history (last 3)
        const recentHistory = actionHistory.slice(-3);

        // CRITICAL: Capture sensor snapshot RIGHT NOW before AI thinks
        // This ensures we validate against the same data the AI saw
        const sensorSnapshot = { ...sensorRef.current };
        const explorationContext = explorationMemoryRef.current.getContext(sensorSnapshot);
        const collisionSnapshotForDecision = snapshotCollisionStats(collisionStatsRef.current);
        finalizePendingDecisionOutcome(sensorSnapshot, explorationContext, Date.now());

        // Get Decision from AI
        const decisionObj = await getDrivingDecision(
          sensorSnapshot,
          recentHistory,
          selectedModel,
          smoothingRef.current,
          explorationContext,
          { collisionSummary: collisionSnapshotForDecision }
        );
        console.log("AI Decision:", decisionObj); // Debug logic


        if (!isActive) return;

        if (!recoveryPhase) {
          // CRITICAL: Re-check sensors NOW (not 225ms ago!)
          // AI thought time can be 200-500ms, sensors update every 50ms
          const currentSensors = sensorRef.current;
          const minFrontDist = Math.min(
            currentSensors.front ?? 99,
            currentSensors.leftDiag ?? 99,
            currentSensors.rightDiag ?? 99
          );
          const heatmapDiag = buildHeatmapDecisionDiagnostics(
            explorationContext,
            decisionObj.strategy,
            sensorSnapshot,
            currentSensors
          );
          const decisionReason = normalizeReasonEnvelope(
            decisionObj.reason || null,
            null,
            {
              throttle: asFiniteNumber(decisionObj.throttle, 0),
              steering: asFiniteNumber(decisionObj.steering, 0),
              duration: asFiniteNumber(decisionObj.duration, 0.22)
            }
          );
          const reasonStatsBefore = reasonValidationStatsRef.current || createInitialReasonValidationStats();
          const reasonPassRateBefore = reasonStatsBefore.totalSteps > 0
            ? reasonStatsBefore.passedSteps / reasonStatsBefore.totalSteps
            : 1;

          // Update Debug Stats
          setAiStats({
            latency: decisionObj.latency,
            raw: decisionObj.raw,
            thought: `${decisionObj.thought || "Driving"}${Array.isArray(decisionObj.actionPlan) && decisionObj.actionPlan.length > 1 ? ` [Plan:${decisionObj.actionPlan.length}]` : ""}`,
            analysis: `[${Date.now() % 10000}ms] ${decisionObj.analysis}`,
            timestamp: new Date().toLocaleTimeString()
          });
          aiMetaRef.current = {
            ...aiMetaRef.current,
            latency: decisionObj.latency ?? 0,
            thought: decisionObj.thought || "",
            analysis: decisionObj.analysis || "",
            reasonCode: decisionReason.code,
            reasonSummary: decisionReason.summary,
            reasonSource: decisionReason.source || "unknown",
            reasonValidationLast: "PENDING",
            reasonBlockedTotal: reasonStatsBefore.blockedSteps ?? 0,
            reasonPassRate: reasonPassRateBefore,
            strategyMode: decisionObj.strategy?.mode || "UNKNOWN",
            strategyTransition: decisionObj.strategy?.transition || "HOLD",
            strategySector: decisionObj.strategy?.chosenSector || "F",
            strategyConfidence: decisionObj.strategy?.confidence ?? 0,
            skillName: decisionObj.skill?.name || "UNKNOWN",
            skillIntensity: decisionObj.skill?.intensity ?? 0,
            reflectionAdjustment: decisionObj.reflection?.adjustment || "",
            parseMethod: decisionObj.parseMethod || "",
            parseRecovered: !!decisionObj.parseRecovered,
            model: decisionObj.model || selectedModel,
            source: "AI",
            isThinking: false,
            safetyOverride: false,
            overrideReason: "",
            lastDecisionAt: Date.now(),
            memoryNoGoRatio: heatmapDiag.noGoRatio ?? 0,
            memoryRevisitRate: heatmapDiag.revisitRate ?? 0,
            memoryCurrentWeight: heatmapDiag.currentCellWeight ?? 0,
            memorySelectedWeight: heatmapDiag.selectedCellWeight ?? 0,
            memorySelectedNoGo: !!heatmapDiag.selectedCellNoGo,
            memorySelectedSector: heatmapDiag.chosenSector || "F",
            memorySelectionReason: heatmapDiag.selectedSource || "",
            directionCooldownApplied: false,
            directionCooldownRemainingMs: 0,
            directionLastSign: directionFlipRef.current.lastSign ?? 0
          };

          // 3. PHASE: ACT
          // Apply Analog Decisions (with fresh safety override).
          const requestedActionPlan = Array.isArray(decisionObj.actionPlan) && decisionObj.actionPlan.length > 0
            ? decisionObj.actionPlan
            : [{ throttle: decisionObj.throttle ?? 0, steering: decisionObj.steering ?? 0, duration: decisionObj.duration ?? 0.05 }];
          const normalizedActionPlan = requestedActionPlan
            .slice(0, 5)
            .map((step) => {
              const normalizedStep = {
                throttle: Math.max(-1, Math.min(1, asFiniteNumber(step?.throttle, 0))),
                steering: Math.max(-1, Math.min(1, asFiniteNumber(step?.steering, 0))),
                duration: Math.max(0.08, Math.min(1.2, asFiniteNumber(step?.duration, 0.22)))
              };
              return {
                ...normalizedStep,
                reason: normalizeReasonEnvelope(
                  step?.reason || null,
                  decisionReason,
                  normalizedStep
                )
              };
            });
          if (normalizedActionPlan.length === 0) {
            normalizedActionPlan.push({
              throttle: 0,
              steering: 0,
              duration: 0.22,
              reason: normalizeReasonEnvelope(
                null,
                decisionReason,
                { throttle: 0, steering: 0, duration: 0.22 }
              )
            });
          }

          let finalThrottle = normalizedActionPlan[0].throttle;

          // SAFETY OVERRIDE: If obstacle appeared during AI thinking, STOP NOW!
          let finalSteering = normalizedActionPlan[0].steering;
          let runtimeReasonOverride = null;
          const calibrationProfile = directionCalibrationRef.current.profile;
          if (calibrationProfile?.applied && calibrationProfile.steeringConfidence >= 0.45) {
            finalSteering *= asFiniteNumber(calibrationProfile.steeringSign, 1);
          }
          let bypassDirectionFlipCooldown = false;

          if (minFrontDist < 2.5 && finalThrottle > 0) {
            bypassDirectionFlipCooldown = true;
            // CRITICAL: Don't just stop - actively avoid!
            // If we're trying to go forward into a wall, we must turn away

            // Check which side is more open
            const leftSpace = Math.min(currentSensors.left ?? 99, currentSensors.leftDiag ?? 99);
            const rightSpace = Math.min(currentSensors.right ?? 99, currentSensors.rightDiag ?? 99);
            const backSpace = currentSensors.back ?? 99;

            // CORNER TRAP DETECTION: All sides blocked?
            const isCornerTrap = minFrontDist < 3.0 && leftSpace < 3.0 && rightSpace < 3.0;

            if (isCornerTrap && currentSensors.speed < 1.0) {
              // TRAPPED IN CORNER! Trigger full recovery maneuver
              console.log("🚨 CORNER TRAP DETECTED! Triggering recovery...");

              // Smart turn direction: favor the more open side + back space
              const turnTowardLeft = (leftSpace + backSpace) > (rightSpace + backSpace);

              // Temporarily store turn preference for recovery phase
              if (!recoveryPhase) {
                lastRecoveryTime.current = Date.now();
                setRecoveryPhase("REVERSING");
                setAction(turnTowardLeft ? "LEFT" : "RIGHT");

                // Schedule phase 2
                setTimeout(() => {
                  setRecoveryPhase("TURNING");
                }, 1500);

                // Schedule end of recovery
                setTimeout(() => {
                  setRecoveryPhase(null);
                  lastRecoveryTime.current = Date.now();
                }, 2500);
              }

              setAiStats(prev => ({
                ...prev,
                thought: `[CORNER TRAP] F:${minFrontDist.toFixed(1)}m L:${leftSpace.toFixed(1)}m R:${rightSpace.toFixed(1)}m - Recovery toward ${turnTowardLeft ? 'LEFT' : 'RIGHT'}!`
              }));
              aiMetaRef.current = {
                ...aiMetaRef.current,
                source: "RECOVERY",
                isThinking: false,
                safetyOverride: true,
                overrideReason: "CORNER_TRAP",
                lastDecisionAt: Date.now()
              };
              // Skip setting controls - recovery will handle it
              return;
            } else if (currentSensors.speed < 1.0) {
              // Nearly stopped but not fully trapped - quick reverse to open side
              if (backSpace > 3.0) {
                finalThrottle = -0.5; // REVERSE
                finalSteering = leftSpace > rightSpace ? 0.8 : -0.8; // Turn toward open side
                runtimeReasonOverride = normalizeReasonEnvelope(
                  {
                    code: "RUNTIME_WALL_ESCAPE_REVERSE",
                    summary: "Runtime safety override: reverse away from front wall pressure.",
                    expectedThrottleSign: -1,
                    expectedSteeringSign: controlValueToSign(finalSteering),
                    source: "runtime"
                  },
                  decisionReason,
                  { throttle: finalThrottle, steering: finalSteering, duration: normalizedActionPlan[0].duration }
                );
                aiMetaRef.current = {
                  ...aiMetaRef.current,
                  safetyOverride: true,
                  overrideReason: "WALL_ESCAPE_REVERSE",
                  lastDecisionAt: Date.now()
                };
                setAiStats(prev => ({
                  ...prev,
                  thought: `[EMERGENCY] Wall ${minFrontDist.toFixed(1)}m! Reversing ${leftSpace > rightSpace ? 'LEFT' : 'RIGHT'} (L:${leftSpace.toFixed(1)}m R:${rightSpace.toFixed(1)}m)`
                }));
              } else {
                finalThrottle = 0.18;
                finalSteering = leftSpace > rightSpace ? 0.88 : -0.88;
                runtimeReasonOverride = normalizeReasonEnvelope(
                  {
                    code: "RUNTIME_WALL_ESCAPE_PIVOT",
                    summary: "Runtime safety override: pivot to open side when rear is tight.",
                    expectedThrottleSign: 1,
                    expectedSteeringSign: controlValueToSign(finalSteering),
                    source: "runtime"
                  },
                  decisionReason,
                  { throttle: finalThrottle, steering: finalSteering, duration: normalizedActionPlan[0].duration }
                );
                aiMetaRef.current = {
                  ...aiMetaRef.current,
                  safetyOverride: true,
                  overrideReason: "WALL_ESCAPE_FORWARD_PIVOT",
                  lastDecisionAt: Date.now()
                };
                setAiStats(prev => ({
                  ...prev,
                  thought: `[EMERGENCY] Front tight but rear ${backSpace.toFixed(1)}m. Pivoting forward ${leftSpace > rightSpace ? 'LEFT' : 'RIGHT'}.`
                }));
              }
              } else {
                // Still moving - just stop for now
                finalThrottle = 0;
              runtimeReasonOverride = normalizeReasonEnvelope(
                {
                  code: "RUNTIME_WALL_STOP",
                  summary: "Runtime safety override: stop forward motion near wall.",
                  expectedThrottleSign: 0,
                  expectedSteeringSign: controlValueToSign(finalSteering),
                  source: "runtime"
                },
                decisionReason,
                { throttle: finalThrottle, steering: finalSteering, duration: normalizedActionPlan[0].duration }
              );
              aiMetaRef.current = {
                ...aiMetaRef.current,
                safetyOverride: true,
                overrideReason: "WALL_STOP",
                lastDecisionAt: Date.now()
              };
              setAiStats(prev => ({
                ...prev,
                thought: `[REALTIME SAFETY] Wall at ${minFrontDist.toFixed(1)}m! Stopping.`
              }));
            }
          }

          const applyDirectionFlipCooldownToStep = (candidateThrottle, bypass = false) => {
            const cooldown = {
              applied: false,
              reason: "",
              remainingMs: 0,
              previousSign: directionFlipRef.current.lastSign ?? 0,
              proposedSign: throttleToSign(candidateThrottle),
              bypassed: bypass
            };
            let throttleOut = candidateThrottle;

            if (!bypass && cooldown.proposedSign !== 0) {
              const nowMs = Date.now();
              const previousSign = directionFlipRef.current.lastSign ?? 0;
              const previousAt = directionFlipRef.current.lastSignAt ?? 0;

              if (previousSign !== 0 && previousSign !== cooldown.proposedSign) {
                const elapsed = nowMs - previousAt;
                if (elapsed < DIRECTION_FLIP_COOLDOWN_MS) {
                  const remaining = DIRECTION_FLIP_COOLDOWN_MS - elapsed;
                  cooldown.applied = true;
                  cooldown.remainingMs = remaining;
                  cooldown.reason = `DIRECTION_FLIP_COOLDOWN_${remaining}ms`;
                  throttleOut = 0;
                } else {
                  directionFlipRef.current.lastSign = cooldown.proposedSign;
                  directionFlipRef.current.lastSignAt = nowMs;
                }
              } else {
                directionFlipRef.current.lastSign = cooldown.proposedSign;
                if (previousSign === 0) directionFlipRef.current.lastSignAt = nowMs;
              }
            } else if (bypass && cooldown.proposedSign !== 0) {
              directionFlipRef.current.lastSign = cooldown.proposedSign;
              directionFlipRef.current.lastSignAt = Date.now();
            }

            return { throttleOut, cooldown };
          };

          normalizedActionPlan[0] = {
            ...normalizedActionPlan[0],
            throttle: finalThrottle,
            steering: finalSteering,
            reason: runtimeReasonOverride || normalizeReasonEnvelope(
              normalizedActionPlan[0]?.reason || null,
              decisionReason,
              {
                throttle: finalThrottle,
                steering: finalSteering,
                duration: normalizedActionPlan[0]?.duration
              }
            )
          };
          const firstStepCooldownResult = applyDirectionFlipCooldownToStep(finalThrottle, bypassDirectionFlipCooldown);
          finalThrottle = firstStepCooldownResult.throttleOut;
          normalizedActionPlan[0].throttle = finalThrottle;
          const directionCooldown = firstStepCooldownResult.cooldown;
          if (directionCooldown.applied) {
            normalizedActionPlan[0].reason = normalizeReasonEnvelope(
              {
                code: "DIRECTION_FLIP_COOLDOWN_HOLD",
                summary: "Direction flip cooldown applied; hold this step.",
                expectedThrottleSign: 0,
                expectedSteeringSign: controlValueToSign(normalizedActionPlan[0].steering),
                source: "runtime"
              },
              normalizedActionPlan[0].reason,
              {
                throttle: normalizedActionPlan[0].throttle,
                steering: normalizedActionPlan[0].steering,
                duration: normalizedActionPlan[0].duration
              }
            );
          } else {
            normalizedActionPlan[0].reason = normalizeReasonEnvelope(
              normalizedActionPlan[0].reason,
              decisionReason,
              {
                throttle: normalizedActionPlan[0].throttle,
                steering: normalizedActionPlan[0].steering,
                duration: normalizedActionPlan[0].duration
              }
            );
          }

          if (directionCooldown.applied) {
            aiMetaRef.current = {
              ...aiMetaRef.current,
              safetyOverride: true,
              overrideReason: directionCooldown.reason,
              directionCooldownApplied: true,
              directionCooldownRemainingMs: Math.max(0, Math.round(directionCooldown.remainingMs)),
              directionLastSign: directionFlipRef.current.lastSign ?? 0,
              lastDecisionAt: Date.now()
            };
            setAiStats(prev => ({
              ...prev,
              thought: `${prev.thought || decisionObj.thought || "Driving"} [Stability] Direction flip cooldown ${Math.round(directionCooldown.remainingMs)}ms`
            }));
          } else {
            aiMetaRef.current = {
              ...aiMetaRef.current,
              directionCooldownApplied: false,
              directionCooldownRemainingMs: 0,
              directionLastSign: directionFlipRef.current.lastSign ?? 0
            };
          }

          const executedActionPlan = [];
          const reasonValidationSummary = {
            decisionReason,
            blockedSteps: 0,
            missingModelReasonSteps: 0,
            signMismatchSteps: 0,
            stepResults: []
          };
          for (let stepIndex = 0; stepIndex < normalizedActionPlan.length; stepIndex += 1) {
            if (!isActive || recoveryPhase) break;

            const step = normalizedActionPlan[stepIndex];
            let stepThrottle = step.throttle;
            let stepSteering = step.steering;
            let stepDurationSec = step.duration;
            const stepSensors = sensorRef.current;
            const stepMinFront = Math.min(
              stepSensors.front ?? 99,
              stepSensors.leftDiag ?? 99,
              stepSensors.rightDiag ?? 99
            );
            let stepReason = normalizeReasonEnvelope(
              step?.reason || null,
              decisionReason,
              { throttle: stepThrottle, steering: stepSteering, duration: stepDurationSec }
            );
            let bypassReasonValidation = stepIndex === 0 && (runtimeReasonOverride !== null || directionCooldown.applied);

            if (stepIndex > 0) {
              const stepCalibrationProfile = directionCalibrationRef.current.profile;
              if (stepCalibrationProfile?.applied && stepCalibrationProfile.steeringConfidence >= 0.45) {
                stepSteering *= asFiniteNumber(stepCalibrationProfile.steeringSign, 1);
              }
              if (stepMinFront < 2.5 && stepThrottle > 0) {
                stepThrottle = 0;
                stepReason = normalizeReasonEnvelope(
                  {
                    code: "ACTION_QUEUE_FRONT_GUARD_HOLD",
                    summary: "Queued forward step blocked by front guard.",
                    expectedThrottleSign: 0,
                    expectedSteeringSign: controlValueToSign(stepSteering),
                    source: "runtime"
                  },
                  stepReason,
                  { throttle: stepThrottle, steering: stepSteering, duration: stepDurationSec }
                );
                aiMetaRef.current = {
                  ...aiMetaRef.current,
                  safetyOverride: true,
                  overrideReason: "ACTION_QUEUE_FRONT_GUARD",
                  lastDecisionAt: Date.now()
                };
                bypassReasonValidation = true;
              }
              const stepCooldownResult = applyDirectionFlipCooldownToStep(stepThrottle, false);
              stepThrottle = stepCooldownResult.throttleOut;
              if (stepCooldownResult.cooldown?.applied) {
                stepReason = normalizeReasonEnvelope(
                  {
                    code: "DIRECTION_FLIP_COOLDOWN_HOLD",
                    summary: "Queued step paused by direction cooldown.",
                    expectedThrottleSign: 0,
                    expectedSteeringSign: controlValueToSign(stepSteering),
                    source: "runtime"
                  },
                  stepReason,
                  { throttle: stepThrottle, steering: stepSteering, duration: stepDurationSec }
                );
                bypassReasonValidation = true;
              }
            }

            const stepValidation = validateReasonedActionStep({
              step: { throttle: stepThrottle, steering: stepSteering },
              reason: stepReason,
              minFrontDist: stepMinFront,
              requireModelReason: true,
              bypass: bypassReasonValidation || stepReason.source === "runtime"
            });
            const hasSignMismatch = stepValidation.issues.includes("THROTTLE_SIGN_MISMATCH")
              || stepValidation.issues.includes("STEERING_SIGN_MISMATCH");

            const reasonStats = reasonValidationStatsRef.current;
            reasonStats.totalSteps += 1;
            if (stepValidation.ok) {
              reasonStats.passedSteps += 1;
            } else {
              reasonStats.blockedSteps += 1;
              if (stepValidation.issues.includes("REASON_NOT_FROM_MODEL")) {
                reasonStats.missingModelReasonSteps += 1;
                reasonValidationSummary.missingModelReasonSteps += 1;
              }
              if (hasSignMismatch) {
                reasonStats.signMismatchSteps += 1;
                reasonValidationSummary.signMismatchSteps += 1;
              }
              reasonValidationSummary.blockedSteps += 1;
              stepThrottle = 0;
              stepSteering = 0;
              stepDurationSec = Math.min(stepDurationSec, 0.18);
              stepReason = normalizeReasonEnvelope(
                {
                  code: `REASON_BLOCK_${stepValidation.primaryIssue || "UNKNOWN"}`,
                  summary: `Blocked step due to reason-check failure: ${stepValidation.issues.join("|") || "UNKNOWN"}.`,
                  expectedThrottleSign: 0,
                  expectedSteeringSign: 0,
                  source: "runtime"
                },
                stepReason,
                { throttle: stepThrottle, steering: stepSteering, duration: stepDurationSec }
              );
              aiMetaRef.current = {
                ...aiMetaRef.current,
                safetyOverride: true,
                overrideReason: `REASON_VALIDATION_${stepValidation.primaryIssue || "BLOCK"}`,
                lastDecisionAt: Date.now()
              };
            }

            reasonValidationSummary.stepResults.push({
              index: stepIndex + 1,
              ok: stepValidation.ok,
              bypassed: stepValidation.bypassed,
              issues: stepValidation.issues,
              reason: stepReason,
              throttleSign: controlValueToSign(stepThrottle),
              steeringSign: controlValueToSign(stepSteering),
              minFrontDist: Number(stepMinFront.toFixed(3))
            });

            const stepDurationMs = Math.max(10, Math.min(3000, stepDurationSec * 1000));
            const stepControls = { throttle: stepThrottle, steering: stepSteering };
            applyControls(stepControls);
            executedActionPlan.push({
              index: stepIndex + 1,
              throttle: stepThrottle,
              steering: stepSteering,
              duration: Number(stepDurationSec.toFixed(3)),
              minFrontDist: Number(stepMinFront.toFixed(3)),
              reason: stepReason,
              reason_validation: {
                ok: stepValidation.ok,
                issues: stepValidation.issues,
                bypassed: stepValidation.bypassed
              }
            });
            await new Promise(r => setTimeout(r, stepDurationMs));
          }
          const reasonStatsAfter = reasonValidationStatsRef.current || createInitialReasonValidationStats();
          const reasonPassRateAfter = reasonStatsAfter.totalSteps > 0
            ? reasonStatsAfter.passedSteps / reasonStatsAfter.totalSteps
            : 1;
          aiMetaRef.current = {
            ...aiMetaRef.current,
            reasonValidationLast: reasonValidationSummary.blockedSteps > 0 ? "BLOCKED" : "PASS",
            reasonBlockedTotal: reasonStatsAfter.blockedSteps ?? 0,
            reasonPassRate: reasonPassRateAfter
          };

          // Log to Session History
          const firstExecutedControls = executedActionPlan[0]
            ? { throttle: executedActionPlan[0].throttle, steering: executedActionPlan[0].steering }
            : { throttle: 0, steering: 0 };
          decisionLog.current.push({
            time: new Date().toISOString(),
            sensor_snapshot: sensorSnapshot,
            sensor_latest: sensorRef.current,
            exploration: explorationContext,
            collision_summary: snapshotCollisionStats(collisionStatsRef.current),
            heatmap_diag: heatmapDiag,
            ai_prompt: decisionObj.prompt || "",
            ai_raw: decisionObj.raw,
            ai_parse_method: decisionObj.parseMethod || "",
            ai_parse_recovered: !!decisionObj.parseRecovered,
            ai_model: decisionObj.model || selectedModel,
            ai_skill: decisionObj.skill || null,
            ai_reason: decisionReason,
            ai_reflection: decisionObj.reflection || null,
            safety_guard: decisionObj.safetyGuard || null,
            direction_cooldown: directionCooldown,
            reason_validation: reasonValidationSummary,
            reason_stats: {
              ...reasonValidationStatsRef.current,
              passRate: aiMetaRef.current.reasonPassRate ?? 1
            },
            runtime_safety_override: !!aiMetaRef.current.safetyOverride,
            ai_parsed: decisionObj,
            controls: firstExecutedControls,
            action_plan_requested: normalizedActionPlan,
            action_plan_executed: executedActionPlan,
            runtime_diagnostics: { collisionSummary: collisionSnapshotForDecision },
            decision_started_at_ms: Date.now(),
            outcome: null
          });

          // Update History
          setAction("AI_CONTROL"); // Just for HUD text

          setIsThinking(false);
          aiMetaRef.current = {
            ...aiMetaRef.current,
            isThinking: false
          };

          // Stop controls while waiting for next think cycle.
          applyControls({ throttle: 0, steering: 0 });
        }
      } catch (err) {
        console.error("AutoPilot Error:", err);
        setAction("BRAKE");
        setErrorMsg(err.message || "AI Connection Failed");
        setIsThinking(false);
        aiMetaRef.current = {
          ...aiMetaRef.current,
          isThinking: false,
          source: "ERROR",
          safetyOverride: true,
          overrideReason: "AI_ERROR"
        };
        // If error, wait a bit before retrying
        await new Promise(r => setTimeout(r, 1000));
      }

      // 4. Loop
      if (isActive) {
        loop();
      }
    };

    loop();

    return () => { isActive = false; };
  }, [autoDrive, recoveryPhase, actionHistory, lastAction, selectedModel, applyControls, finalizePendingDecisionOutcome]);

  // Physics Tuning State
  const [physicsSettings, setPhysicsSettings] = useState({
    speedForce: 12000, // Boosted: High Speed enabled by Safety Stop
    turnTorque: 200,
    linearDamping: 0.2,
    interval: 1000,
    sensorDynamic: true,
    sensorRangeMin: 7,
    sensorRangeMax: 14
  });

  const correlationChart = useMemo(() => {
    const history = telemetry || [];
    if (history.length < 2) return { anglePoints: "", steeringPoints: "" };

    const step = Math.ceil(history.length / 800);

    const createPath = (getValueColor) => {
      return history.map((pt, i) => {
        if (i % step !== 0) return "";
        const x = (i / (history.length - 1)) * 100;

        let val = getValueColor(pt);
        val = Math.max(-1, Math.min(1, val));

        const y = 50 - (val * 50);
        return `${x},${y}`;
      }).filter(p => p).join(" ");
    };

    const anglePoints = createPath(pt => pt.targetAngle / 90);
    const steeringPoints = createPath(pt => pt.steering);

    return { anglePoints, steeringPoints };
  }, [telemetry]);

  const intentionalityChart = useMemo(() => {
    const history = telemetry || [];
    if (history.length < 2) return "";
    const step = Math.ceil(history.length / 800);

    const historyWithAvg = history.reduce((acc, item, index) => {
      const prevSum = index > 0 ? acc[index - 1].cumulativeSum : 0;
      const newSum = prevSum + item.intentionality;
      acc.push({ ...item, cumulativeAvg: newSum / (index + 1) });
      return acc;
    }, []);

    const points = historyWithAvg
      .map((pt, i) => {
        if (i % step !== 0) return "";
        const x = (i / (historyWithAvg.length - 1)) * 100;
        const y = 100 - ((pt.cumulativeAvg + 1) / 2 * 100);
        return `${x},${y}`;
      })
      .filter(p => p)
      .join(" ");

    return points;
  }, [telemetry]);

  const memoryMapView = useMemo(() => {
    if (!memoryViz) return null;

    const radius = memoryViz.radiusCells ?? 8;
    const gridSize = (radius * 2) + 1;
    const cellPx = Math.max(8, Math.floor(260 / gridSize));
    const mapPx = gridSize * cellPx;
    const clamp01 = (v) => Math.max(0, Math.min(1, v));

    const cells = (memoryViz.cells || []).map((cell) => {
      const scoreNorm = clamp01((cell.score + 1.2) / 2.4);
      const riskNorm = clamp01(cell.risk ?? 0);

      const r = Math.round((riskNorm * 200) + ((1 - scoreNorm) * 35));
      const g = Math.round((scoreNorm * 210) + ((1 - riskNorm) * 45));
      const b = cell.hasData ? 80 : 30;
      const fill = cell.outsideBounds
        ? "rgb(14, 16, 24)"
        : cell.barrierBlocked
          ? "rgb(96, 40, 40)"
          : cell.obstacleDominant
            ? "rgb(110, 54, 54)"
          : `rgb(${r}, ${g}, ${b})`;

      return {
        ...cell,
        x: (cell.dx + radius) * cellPx,
        y: (cell.dz + radius) * cellPx,
        fill
      };
    });

    const pathPoints = (memoryViz.recentPath || [])
      .filter((p) => Math.abs(p.dx) <= radius && Math.abs(p.dz) <= radius)
      .map((p, idx, arr) => ({
        x: ((p.dx + radius) * cellPx) + (cellPx / 2),
        y: ((p.dz + radius) * cellPx) + (cellPx / 2),
        alpha: (idx + 1) / Math.max(1, arr.length)
      }));
    const markerSize = Math.max(6, Math.floor(cellPx * 0.8));
    const worldHeadingDeg = asFiniteNumber(memoryViz.headingDeg, 0);
    const mapHeadingDeg = 180 - worldHeadingDeg;

    return {
      radius,
      cellPx,
      mapPx,
      cells,
      pathPoints,
      headingDeg: worldHeadingDeg,
      mapHeadingDeg,
      centerCell: memoryViz.center || { ix: 0, iz: 0 },
      worldPos: {
        x: asFiniteNumber(sensorData.worldX, 0),
        z: asFiniteNumber(sensorData.worldZ, 0)
      },
      vehicleMarker: {
        cx: mapPx / 2,
        cy: mapPx / 2,
        size: markerSize
      },
      mapCompass: {
        top: "N (-Z)",
        right: "E (+X)",
        bottom: "S (+Z)",
        left: "W (-X)"
      },
      loopRate: memoryViz.loopRate ?? 0,
      loopWarning: memoryViz.loopWarning || "LOW",
      preferredSector: memoryViz.preferredSector || "F",
      mappedCells: memoryViz.memoryStats?.mappedCells ?? 0,
      noGoRatio: memoryViz.diagnostics?.noGoRatio ?? 0,
      safeCandidates: memoryViz.diagnostics?.safeCandidateCount ?? 0,
      candidateCount: memoryViz.diagnostics?.candidateCount ?? 0,
      memorySensorRange: memoryViz.memoryStats?.sensorRange ?? asFiniteNumber(sensorData.sensorRange, 10)
    };
  }, [memoryViz, sensorData.worldX, sensorData.worldZ, sensorData.sensorRange]);

  const priorityStatus = useMemo(() => {
    const targetHitCount = Object.values(sensorData.targetHits || {}).filter(Boolean).length;
    const minObstacleDist = Math.min(
      asFiniteNumber(sensorData.front, 10),
      asFiniteNumber(sensorData.leftDiag, 10),
      asFiniteNumber(sensorData.rightDiag, 10),
      asFiniteNumber(sensorData.left, 10),
      asFiniteNumber(sensorData.right, 10),
      asFiniteNumber(sensorData.back, 10),
      asFiniteNumber(sensorData.backLeft, 10),
      asFiniteNumber(sensorData.backRight, 10)
    );

    let runState = "MANUAL";
    if (recoveryPhase) runState = `RECOVERY:${recoveryPhase}`;
    else if (autoDrive && isThinking) runState = "AI_THINKING";
    else if (autoDrive) runState = "AI_DRIVING";

    const dangerLevel = minObstacleDist < 2.5 ? "CRITICAL" : minObstacleDist < 4 ? "CAUTION" : "CLEAR";
    const riskTone = dangerLevel === "CRITICAL" ? "text-red-300" : dangerLevel === "CAUTION" ? "text-amber-300" : "text-emerald-300";

    return {
      runState,
      dangerLevel,
      riskTone,
      minObstacleDist,
      targetDistance: asFiniteNumber(sensorData.distanceToTarget, 0),
      targetAngle: asFiniteNumber(sensorData.angleToTarget, 0),
      targetHitCount,
      action: recoveryPhase || lastAction,
      aiMode: aiMetaRef.current.strategyMode || "UNKNOWN",
      aiConfidence: asFiniteNumber(aiMetaRef.current.strategyConfidence, 0),
      aiLatency: asFiniteNumber(aiMetaRef.current.latency, 0),
      preferredSector: memoryMapView?.preferredSector || "F",
      loopRate: asFiniteNumber(memoryMapView?.loopRate, 0) * 100
    };
  }, [sensorData, recoveryPhase, autoDrive, isThinking, lastAction, memoryMapView]);

  const handleStopAutodrive = useCallback((options = {}) => {
    const autoDownloadDrive = options?.autoDownloadDrive !== false;
    const stopTrigger = typeof options?.trigger === "string" ? options.trigger : "auto_stop";
    const latestSensor = { ...sensorRef.current };
    const latestExploration = explorationMemoryRef.current.getContext(latestSensor);
    finalizePendingDecisionOutcome(latestSensor, latestExploration, Date.now());
    if (autoDriveRef.current && autoDownloadDrive) {
      downloadDriveLogOnly(stopTrigger);
    }
    setAutoDrive(false);
    applyControls({ throttle: 0, steering: 0 });
    setErrorMsg(null);
    setAiStats({ latency: 0, raw: "", thought: "" });
    collisionEventsRef.current = [];
    collisionStatsRef.current = createInitialCollisionStats();
    smoothingRef.current = {
      lastSteering: 0,
      lastStrategyMode: "",
      noContactMs: 0,
      noContactCycles: 0,
      lastContactAt: 0,
      lastTickAt: 0,
      reacquireTurnDir: 1,
      lastReacquireFlipAt: 0,
      lastOutcomeSummary: "",
      lastOutcomeDetails: null,
      lastReflectionHint: "",
      lastSkillName: "",
      modeHoldRemaining: 0,
      targetLockHoldRemaining: 0
    };
    directionFlipRef.current = {
      lastSign: 0,
      lastSignAt: 0
    };
    reasonValidationStatsRef.current = createInitialReasonValidationStats();
    aiMetaRef.current = {
      latency: 0,
      thought: "",
      analysis: "",
      reasonCode: "",
      reasonSummary: "",
      reasonSource: "none",
      reasonValidationLast: "NONE",
      reasonBlockedTotal: 0,
      reasonPassRate: 1,
      strategyMode: "UNKNOWN",
      strategyTransition: "HOLD",
      strategySector: "F",
      strategyConfidence: 0,
      skillName: "UNKNOWN",
      skillIntensity: 0,
      reflectionAdjustment: "",
      parseMethod: "",
      parseRecovered: false,
      model: "",
      source: "IDLE",
      isThinking: false,
      safetyOverride: false,
      overrideReason: "",
      lastDecisionAt: 0,
      memoryNoGoRatio: 0,
      memoryRevisitRate: 0,
      memoryCurrentWeight: 0,
      memorySelectedWeight: 0,
      memorySelectedNoGo: false,
      memorySelectedSector: "F",
      memorySelectionReason: "",
      directionCooldownApplied: false,
      directionCooldownRemainingMs: 0,
      directionLastSign: 0
    };
  }, [applyControls, downloadDriveLogOnly, finalizePendingDecisionOutcome]);

  const delayMs = useCallback((ms) => new Promise((resolve) => setTimeout(resolve, ms)), []);

  const waitForCondition = useCallback(async (predicate, timeoutMs = 15000, pollMs = 120) => {
    const start = Date.now();
    while ((Date.now() - start) < timeoutMs) {
      if (predicate()) return true;
      await delayMs(pollMs);
    }
    return predicate();
  }, [delayMs]);

  const evaluateRunStartVehicleState = useCallback((sensorCandidate) => {
    const sensor = sensorCandidate && typeof sensorCandidate === "object" ? sensorCandidate : {};
    const x = asFiniteNumber(sensor.worldX, Number.NaN);
    const y = asFiniteNumber(sensor.worldY, Number.NaN);
    const z = asFiniteNumber(sensor.worldZ, Number.NaN);
    const speed = Math.abs(asFiniteNumber(sensor.speed, Number.NaN));
    const verticalSpeed = Math.abs(asFiniteNumber(sensor.verticalSpeed, Number.NaN));
    const headingDeg = asFiniteNumber(sensor.headingDeg, Number.NaN);
    const distanceToTarget = asFiniteNumber(sensor.distanceToTarget, Number.NaN);
    const sensorRange = asFiniteNumber(sensor.sensorRange, Number.NaN);
    const spawnX = asFiniteNumber(DEFAULT_CAR_SPAWN.position?.[0], 0);
    const spawnZ = asFiniteNumber(DEFAULT_CAR_SPAWN.position?.[2], -10);
    const distanceFromSpawn = Number.isFinite(x) && Number.isFinite(z)
      ? Math.hypot(x - spawnX, z - spawnZ)
      : Number.POSITIVE_INFINITY;

    const checks = {
      finitePose: Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z),
      finiteMotion: Number.isFinite(speed) && Number.isFinite(verticalSpeed),
      finiteNavigation: Number.isFinite(headingDeg) && Number.isFinite(distanceToTarget) && Number.isFinite(sensorRange),
      inBounds: Number.isFinite(x) && Number.isFinite(z)
        && x >= TRACK_WORLD_BOUNDS.minX
        && x <= TRACK_WORLD_BOUNDS.maxX
        && z >= TRACK_WORLD_BOUNDS.minZ
        && z <= TRACK_WORLD_BOUNDS.maxZ,
      nearSpawn: Number.isFinite(distanceFromSpawn) && distanceFromSpawn <= RUN_START_SPAWN_TOLERANCE_M,
      speedReady: Number.isFinite(speed) && speed <= RUN_START_MAX_SPEED,
      grounded: Number.isFinite(y) && y >= RUN_START_MIN_WORLD_Y && y <= RUN_START_MAX_WORLD_Y,
      gravityStable: Number.isFinite(verticalSpeed) && verticalSpeed <= RUN_START_MAX_VERTICAL_SPEED
    };
    const failedChecks = Object.entries(checks)
      .filter(([, pass]) => !pass)
      .map(([name]) => name);
    const summary = failedChecks.length === 0
      ? "run-start vehicle state is valid"
      : `run-start invalid (${failedChecks.join(", ")})`;

    return {
      ok: failedChecks.length === 0,
      summary,
      failedChecks,
      metrics: {
        worldX: Number.isFinite(x) ? Number(x.toFixed(3)) : null,
        worldY: Number.isFinite(y) ? Number(y.toFixed(3)) : null,
        worldZ: Number.isFinite(z) ? Number(z.toFixed(3)) : null,
        speed: Number.isFinite(speed) ? Number(speed.toFixed(3)) : null,
        verticalSpeed: Number.isFinite(verticalSpeed) ? Number(verticalSpeed.toFixed(3)) : null,
        headingDeg: Number.isFinite(headingDeg) ? Number(headingDeg.toFixed(3)) : null,
        distanceToTarget: Number.isFinite(distanceToTarget) ? Number(distanceToTarget.toFixed(3)) : null,
        sensorRange: Number.isFinite(sensorRange) ? Number(sensorRange.toFixed(3)) : null,
        distanceFromSpawn: Number.isFinite(distanceFromSpawn) ? Number(distanceFromSpawn.toFixed(3)) : null
      }
    };
  }, []);

  const waitForRunStartVehicleState = useCallback(async (timeoutMs = 3500, pollMs = 120) => {
    const startedAt = Date.now();
    let latest = evaluateRunStartVehicleState(sensorRef.current);
    while ((Date.now() - startedAt) < timeoutMs) {
      if (latest.ok) return latest;
      await delayMs(pollMs);
      latest = evaluateRunStartVehicleState(sensorRef.current);
    }
    return latest;
  }, [delayMs, evaluateRunStartVehicleState]);

  const applyExperimentCondition = useCallback(async (condition) => {
    const modelCandidate = typeof condition?.model === "string" ? condition.model.trim() : "";
    if (modelCandidate && MODEL_OPTIONS.includes(modelCandidate)) {
      setSelectedModel(modelCandidate);
      await waitForCondition(() => selectedModelRef.current === modelCandidate, 1600, 60);
    }

    if (condition?.physicsPatch && typeof condition.physicsPatch === "object") {
      const patch = condition.physicsPatch;
      setPhysicsSettings((prev) => {
        const nextMin = Number.isFinite(patch.sensorRangeMin) ? patch.sensorRangeMin : prev.sensorRangeMin;
        const nextMaxRaw = Number.isFinite(patch.sensorRangeMax) ? patch.sensorRangeMax : prev.sensorRangeMax;
        const nextMax = Math.max(nextMin + 0.5, nextMaxRaw);
        return {
          ...prev,
          ...patch,
          sensorRangeMin: nextMin,
          sensorRangeMax: nextMax
        };
      });
    }
    await delayMs(180);
  }, [delayMs, waitForCondition]);

  const stopExperimentAutomation = useCallback(() => {
    experimentAbortRef.current = true;
    setExperimentRunner((prev) => ({
      ...prev,
      abortRequested: true,
      phase: prev.running ? "STOP_REQUESTED" : prev.phase,
      summary: prev.running ? "Stop requested. Finishing current run safely." : prev.summary
    }));
  }, []);

  const waitForExperimentRunStart = useCallback(async (timeoutMs = 22000) => {
    const startedAt = Date.now();
    let preflightObserved = false;
    while ((Date.now() - startedAt) < timeoutMs) {
      if (autoDriveRef.current) {
        return { started: true, blocked: false, reason: "" };
      }
      if (isPreflightRunningRef.current) preflightObserved = true;

      const err = typeof errorMsgRef.current === "string" ? errorMsgRef.current : "";
      const errLower = err.toLowerCase();
      const preflightBlocked = preflightObserved
        && !isPreflightRunningRef.current
        && !autoDriveRef.current
        && (errLower.includes("preflight blocked") || errLower.includes("preflight exception") || errLower.includes("sensor refresh timeout"));
      if (preflightBlocked) {
        return {
          started: false,
          blocked: true,
          reason: err || "Preflight blocked."
        };
      }
      await delayMs(120);
    }
    return {
      started: !!autoDriveRef.current,
      blocked: false,
      reason: typeof errorMsgRef.current === "string" && errorMsgRef.current.trim().length > 0
        ? errorMsgRef.current
        : "Autodrive start timeout."
    };
  }, [delayMs]);

  const runExperimentAutomation = useCallback(async () => {
    if (experimentRunner.running || autoDriveRef.current || isPreflightRunningRef.current || isAnalyzingRef.current) {
      setErrorMsg("Experiment automation cannot start while drive/preflight/analyze is active.");
      return;
    }

    const runs = buildExperimentPlan(experimentConfig);
    if (!Array.isArray(runs) || runs.length === 0) {
      setErrorMsg("Experiment plan is empty. Select at least one AB condition.");
      return;
    }

    experimentAbortRef.current = false;
    const startedAtIso = new Date().toISOString();
    const runResults = [];
    const runArtifacts = [];
    const saveMode = experimentConfig.saveMode === EXPERIMENT_SAVE_MODES.SPLIT_PER_RUN
      ? EXPERIMENT_SAVE_MODES.SPLIT_PER_RUN
      : EXPERIMENT_SAVE_MODES.SINGLE_BUNDLE_END;
    const maxStartAttempts = Math.max(
      1,
      Math.min(5, Math.round(asFiniteNumber(experimentConfig.startAttemptsPerRun, EXPERIMENT_START_MAX_ATTEMPTS)))
    );

    setErrorMsg(null);
    setExperimentRunner({
      running: true,
      phase: "PREPARING",
      abortRequested: false,
      currentRun: 0,
      totalRuns: runs.length,
      activeConditionId: "-",
      activeConditionLabel: "-",
      startedAt: startedAtIso,
      finishedAt: null,
      summary: `Experiment automation started: ${runs.length} runs.`,
      results: []
    });

    try {
      let consecutiveGroundGravityBlocks = 0;
      for (let index = 0; index < runs.length; index += 1) {
        if (experimentAbortRef.current) break;

        const run = runs[index];
        const runTag = `${run.runTag}_${formatStamp(Date.now())}`;
        setExperimentRunner((prev) => ({
          ...prev,
          phase: "PREPARING",
          currentRun: index + 1,
          activeConditionId: run.id,
          activeConditionLabel: run.label,
          summary: `Preparing ${run.id} (${run.label}), repeat ${run.repeat}.`
        }));

        await applyExperimentCondition(run);
        if (experimentAbortRef.current) break;

        setExperimentRunner((prev) => ({
          ...prev,
          phase: "PREFLIGHT",
          summary: `Starting preflight for ${run.id}.`
        }));

        let startOutcome = { started: false, blocked: false, reason: "Autodrive not started." };
        let runPreflightReport = null;
        let startAttempt = 0;
        while (startAttempt < maxStartAttempts) {
          startAttempt += 1;
          const attemptTrigger = startAttempt === 1 ? runTag : `${runTag}_retry${startAttempt}`;
          const preflightAttempt = await startAutodriveWithPreflight(attemptTrigger, run.model || null);
          const cooldownMsRemaining = Math.max(0, Math.round(asFiniteNumber(preflightAttempt?.cooldownMsRemaining, 0)));
          if (cooldownMsRemaining > 0 && !experimentAbortRef.current) {
            setExperimentRunner((prev) => ({
              ...prev,
              phase: "PREFLIGHT_WAIT",
              summary: `Waiting preflight cooldown for ${run.id} (${Math.ceil(cooldownMsRemaining / 1000)}s)...`
            }));
            await delayMs(Math.min(7000, cooldownMsRemaining + 150));
            startAttempt -= 1;
            continue;
          }
          if (preflightAttempt?.preflightReport) {
            runPreflightReport = preflightAttempt.preflightReport;
          }

          if (preflightAttempt?.started) {
            startOutcome = await waitForExperimentRunStart(22000);
          } else {
            startOutcome = {
              started: false,
              blocked: true,
              reason: preflightAttempt?.reason || "Autodrive did not start after preflight."
            };
          }

          const latestPreflight = runPreflightReport || preflightReportRef.current;
          if (!didAllPreflightChecksPass(latestPreflight)) {
            startOutcome = {
              started: false,
              blocked: true,
              reason: startOutcome.reason || "Preflight blocked: all checks must PASS."
            };
          }

          if (startOutcome.started || experimentAbortRef.current) break;
          if (startAttempt < maxStartAttempts) {
            setExperimentRunner((prev) => ({
              ...prev,
              phase: "PREFLIGHT_RETRY",
              summary: `Retrying ${run.id} preflight (${startAttempt + 1}/${maxStartAttempts})...`
            }));
            await delayMs(350);
          }
        }

        if (!startOutcome.started) {
          const latestPreflight = runPreflightReport || preflightReportRef.current || createInitialPreflightState();
          const groundGravityBlocked = isGroundGravityPreflightFailure(latestPreflight);
          if (groundGravityBlocked) {
            consecutiveGroundGravityBlocks += 1;
          } else {
            consecutiveGroundGravityBlocks = 0;
          }
          runResults.push({
            conditionId: run.id,
            label: run.label,
            repeat: run.repeat,
            model: run.model,
            runSeconds: run.runSeconds,
            runTag,
            startedAt: new Date().toISOString(),
            status: "PRECHECK_BLOCKED",
            reason: startOutcome.reason || errorMsgRef.current || "Autodrive did not start after preflight.",
            preflight: latestPreflight,
            metrics: summarizeTelemetryForExperiment([], 0, createInitialCollisionStats())
          });

          if (groundGravityBlocked && consecutiveGroundGravityBlocks >= EXPERIMENT_PRECHECK_BLOCK_STREAK_ABORT_THRESHOLD) {
            experimentAbortRef.current = true;
            const failFastSummary = `Experiment fail-fast: repeated CAR_PHYSICS_READY ground/gravity failures (${consecutiveGroundGravityBlocks} in a row).`;
            setErrorMsg(failFastSummary);
            setExperimentRunner((prev) => ({
              ...prev,
              phase: "ABORTED",
              results: [...runResults],
              summary: failFastSummary
            }));
            break;
          }

          setExperimentRunner((prev) => ({
            ...prev,
            phase: "SKIPPED",
            results: [...runResults],
            summary: `Skipped ${run.id}: preflight did not clear.`
          }));
          continue;
        }

        const runStartValidation = await waitForRunStartVehicleState(4500, 120);
        if (!runStartValidation.ok) {
          if (autoDriveRef.current) {
            handleStopAutodrive({ autoDownloadDrive: false, trigger: `${runTag}_start_invalid` });
            await waitForCondition(() => !autoDriveRef.current, 6000, 80);
          }

          const failureReason = `Run-start vehicle validation failed: ${runStartValidation.summary}.`;
          setErrorMsg(failureReason);
          if (runStartValidation.failedChecks?.includes("grounded") || runStartValidation.failedChecks?.includes("gravityStable")) {
            consecutiveGroundGravityBlocks += 1;
          } else {
            consecutiveGroundGravityBlocks = 0;
          }
          runResults.push({
            conditionId: run.id,
            label: run.label,
            repeat: run.repeat,
            model: run.model,
            runSeconds: run.runSeconds,
            runTag,
            startedAt: new Date().toISOString(),
            status: "PRECHECK_BLOCKED",
            reason: failureReason,
            preflight: runPreflightReport || preflightReportRef.current || createInitialPreflightState(),
            runStartValidation,
            metrics: summarizeTelemetryForExperiment([], 0, createInitialCollisionStats())
          });

          if (consecutiveGroundGravityBlocks >= EXPERIMENT_PRECHECK_BLOCK_STREAK_ABORT_THRESHOLD) {
            experimentAbortRef.current = true;
            const failFastSummary = `Experiment fail-fast: repeated run-start ground/gravity validation failures (${consecutiveGroundGravityBlocks} in a row).`;
            setExperimentRunner((prev) => ({
              ...prev,
              phase: "ABORTED",
              results: [...runResults],
              summary: failFastSummary
            }));
            break;
          }

          setExperimentRunner((prev) => ({
            ...prev,
            phase: "SKIPPED",
            results: [...runResults],
            summary: `Skipped ${run.id}: run-start vehicle validation failed.`
          }));
          continue;
        }
        consecutiveGroundGravityBlocks = 0;

        setExperimentRunner((prev) => ({
          ...prev,
          phase: "RUNNING",
          summary: `Running ${run.id} for ${run.runSeconds}s.`
        }));

        const runStartMs = Date.now();
        let remainingMs = run.runMs;
        while (remainingMs > 0 && !experimentAbortRef.current) {
          const sleepMs = Math.min(250, remainingMs);
          await delayMs(sleepMs);
          remainingMs -= sleepMs;
        }

        const historySnapshot = Array.isArray(telemetryRef.current) ? [...telemetryRef.current] : [];
        const driveLogSnapshot = Array.isArray(decisionLog.current) ? [...decisionLog.current] : [];
        const collisionEventsSnapshot = Array.isArray(collisionEventsRef.current) ? [...collisionEventsRef.current] : [];
        const decisionCountSnapshot = driveLogSnapshot.length;
        const collisionSnapshot = snapshotCollisionStats(collisionStatsRef.current);
        const status = experimentAbortRef.current ? "ABORTED" : "DONE";
        const safeRunTag = String(runTag || "manual").toLowerCase().replace(/[^a-z0-9_-]/g, "_");
        const runStamp = formatStamp(Date.now());
        const runSessionPrefix = buildSessionPrefix();

        setExperimentRunner((prev) => ({
          ...prev,
          phase: "EXPORTING",
          summary: `Exporting logs for ${run.id}.`
        }));

        handleStopAutodrive({ autoDownloadDrive: false, trigger: runTag });
        await waitForCondition(() => !autoDriveRef.current, 6000, 80);

        let reportHtml = "";
        let reportAiReview = "";
        let reportStatus = "not_requested";
        if (experimentConfig.includeHtmlReport || experimentConfig.includeAllLogsBundle) {
          if (Array.isArray(historySnapshot) && historySnapshot.length >= 10) {
            try {
              reportAiReview = await generateAIReview(historySnapshot, run.model || selectedModelRef.current);
              reportHtml = buildHTMLReportContent(historySnapshot, reportAiReview) || "";
              reportStatus = reportHtml ? "ok" : "empty";
            } catch (reportErr) {
              console.error("Experiment run report generation failed:", reportErr);
              reportStatus = "error";
              reportHtml = "<html><body><h1>Report generation failed.</h1></body></html>";
            }
          } else {
            reportStatus = "insufficient_data";
            reportHtml = "<html><body><h1>Not enough telemetry to generate report.</h1></body></html>";
          }
        }

        const runMetaSnapshot = {
          sessionPrefix: runSessionPrefix,
          exportedAt: new Date().toISOString(),
          trigger: safeRunTag,
          model: run.model || selectedModelRef.current,
          telemetrySamples: Array.isArray(historySnapshot) ? historySnapshot.length : 0,
          decisionLogRecords: driveLogSnapshot.length,
          collisionEvents: collisionEventsSnapshot.length,
          collisionCount: collisionSnapshot.totalCount,
          sameWallCollisionCount: collisionSnapshot.sameWallRepeatCount,
          sameWallConsecutiveCollisionCount: collisionSnapshot.sameWallConsecutiveRepeatCount,
          collisionByRegion: collisionSnapshot.byRegion,
          preflight: runPreflightReport || preflightReportRef.current || createInitialPreflightState(),
          preflightGatePolicy: PREFLIGHT_GATE_POLICY,
          preflightFailureLogFolder: PREFLIGHT_FAILURE_LOG_FOLDER,
          preflightFailureLogCount: preflightGateLogsRef.current.length,
          latestPreflightFailureLog: preflightGateLogsRef.current.length > 0
            ? preflightGateLogsRef.current[preflightGateLogsRef.current.length - 1]
            : null,
          directionCalibration: directionCalibrationRef.current?.profile || null
        };

        if (saveMode === EXPERIMENT_SAVE_MODES.SPLIT_PER_RUN) {
          triggerDownload(
            JSON.stringify(driveLogSnapshot, null, 2),
            `${runSessionPrefix}_drive_gemma_drive_log_${runStamp}_${safeRunTag}.json`,
            "application/json"
          );
          triggerDownload(
            JSON.stringify(historySnapshot, null, 2),
            `${runSessionPrefix}_telemetry_${runStamp}_${safeRunTag}.json`,
            "application/json"
          );
          triggerDownload(
            JSON.stringify(runMetaSnapshot, null, 2),
            `${runSessionPrefix}_meta_${runStamp}_${safeRunTag}.json`,
            "application/json"
          );

          if (experimentConfig.includeHtmlReport) {
            triggerDownload(
              reportHtml || "<html><body><h1>Report generation skipped.</h1></body></html>",
              `${runSessionPrefix}_report_driver_limit_report_${runStamp}_${safeRunTag}.html`,
              "text/html"
            );
          }

          if (experimentConfig.includeAllLogsBundle) {
            const allLogsPayload = {
              sessionPrefix: runSessionPrefix,
              exportedAt: new Date().toISOString(),
              model: run.model || selectedModelRef.current,
              telemetrySamples: Array.isArray(historySnapshot) ? historySnapshot.length : 0,
              decisionLogRecords: driveLogSnapshot.length,
              collisionEvents: collisionEventsSnapshot.length,
              collisionSummary: collisionSnapshot,
              preflight: runPreflightReport || preflightReportRef.current || createInitialPreflightState(),
              preflightGatePolicy: PREFLIGHT_GATE_POLICY,
              preflightFailureLogFolder: PREFLIGHT_FAILURE_LOG_FOLDER,
              preflightFailureLogs: preflightGateLogsRef.current,
              directionCalibration: directionCalibrationRef.current?.profile || null,
              reportStatus,
              aiReview: reportAiReview,
              files: {
                driveLog: driveLogSnapshot,
                telemetry: historySnapshot,
                collisionEvents: collisionEventsSnapshot,
                reportHtml
              }
            };
            triggerDownload(
              JSON.stringify(allLogsPayload, null, 2),
              `${runSessionPrefix}_all_logs_${runStamp}_${safeRunTag}.json`,
              "application/json"
            );
          }
        } else {
          const runArtifact = {
            conditionId: run.id,
            label: run.label,
            repeat: run.repeat,
            runTag,
            exportedAt: new Date().toISOString(),
            fileNames: {
              driveLog: `${runSessionPrefix}_drive_gemma_drive_log_${runStamp}_${safeRunTag}.json`,
              telemetry: `${runSessionPrefix}_telemetry_${runStamp}_${safeRunTag}.json`,
              meta: `${runSessionPrefix}_meta_${runStamp}_${safeRunTag}.json`,
              reportHtml: experimentConfig.includeHtmlReport
                ? `${runSessionPrefix}_report_driver_limit_report_${runStamp}_${safeRunTag}.html`
                : null
            },
            files: {
              driveLog: driveLogSnapshot,
              telemetry: historySnapshot,
              meta: runMetaSnapshot,
              reportStatus,
              aiReview: reportAiReview,
              reportHtml: experimentConfig.includeHtmlReport ? reportHtml : null
            }
          };
          if (!experimentConfig.includeHtmlReport) {
            delete runArtifact.files.aiReview;
          }
          runArtifacts.push(runArtifact);
        }

        const metrics = summarizeTelemetryForExperiment(
          historySnapshot,
          decisionCountSnapshot,
          collisionSnapshot
        );
        runResults.push({
          conditionId: run.id,
          label: run.label,
          repeat: run.repeat,
          model: run.model,
          runSeconds: run.runSeconds,
          runTag,
          startedAt: new Date(runStartMs).toISOString(),
          finishedAt: new Date().toISOString(),
          status,
          reason: status === "ABORTED" ? "Abort requested by user." : "Completed",
          metrics
        });
        setExperimentRunner((prev) => ({
          ...prev,
          results: [...runResults],
          summary: `${run.id} ${status === "DONE" ? "completed" : "aborted"} (${runResults.length}/${runs.length}).`
        }));

        if (experimentAbortRef.current) break;
        await delayMs(240);
      }
    } catch (err) {
      console.error("Experiment automation failed:", err);
      setErrorMsg(`Experiment automation error: ${err?.message || "unknown error"}`);
      if (autoDriveRef.current) {
        handleStopAutodrive({ autoDownloadDrive: false, trigger: "experiment_exception" });
      }
    } finally {
      const nowStamp = formatStamp(Date.now());
      const sessionPrefix = buildSessionPrefix();
      const summaryPayload = {
        exportedAt: new Date().toISOString(),
        startedAt: startedAtIso,
        finishedAt: new Date().toISOString(),
        aborted: !!experimentAbortRef.current,
        config: experimentConfig,
        saveMode,
        selectedConditionIds: Array.isArray(experimentConfig.selectedConditionIds)
          ? experimentConfig.selectedConditionIds
          : [],
        conditionMatrix: EXPERIMENT_CONDITION_MATRIX,
        totalRunsPlanned: runs.length,
        totalRunsCompleted: runResults.length,
        results: runResults,
        runArtifacts: saveMode === EXPERIMENT_SAVE_MODES.SINGLE_BUNDLE_END ? runArtifacts : undefined
      };
      triggerDownload(
        JSON.stringify(summaryPayload, null, 2),
        `${sessionPrefix}_experiment_automation_summary_${nowStamp}.json`,
        "application/json"
      );
      setExperimentRunner({
        running: false,
        phase: experimentAbortRef.current ? "ABORTED" : "DONE",
        abortRequested: !!experimentAbortRef.current,
        currentRun: runResults.length,
        totalRuns: runs.length,
        activeConditionId: "-",
        activeConditionLabel: "-",
        startedAt: startedAtIso,
        finishedAt: new Date().toISOString(),
        summary: experimentAbortRef.current
          ? `Experiment automation stopped (${runResults.length}/${runs.length} run(s) exported).`
          : `Experiment automation complete (${runResults.length}/${runs.length} run(s) exported${saveMode === EXPERIMENT_SAVE_MODES.SINGLE_BUNDLE_END ? " into one bundle" : ""}).`,
        results: runResults
      });
      experimentAbortRef.current = false;
    }
  }, [
    applyExperimentCondition,
    buildSessionPrefix,
    delayMs,
    didAllPreflightChecksPass,
    experimentConfig,
    experimentRunner.running,
    handleStopAutodrive,
    isGroundGravityPreflightFailure,
    startAutodriveWithPreflight,
    triggerDownload,
    waitForExperimentRunStart,
    waitForRunStartVehicleState,
    waitForCondition
  ]);

  const selectedExperimentConditionIds = useMemo(() => {
    const selectedRaw = Array.isArray(experimentConfig.selectedConditionIds)
      ? experimentConfig.selectedConditionIds
      : [];
    const selectedSet = new Set(selectedRaw.map((id) => String(id || "").trim()).filter(Boolean));
    return EXPERIMENT_CONDITION_MATRIX
      .map((condition) => condition.id)
      .filter((id) => selectedSet.has(id));
  }, [experimentConfig.selectedConditionIds]);

  const selectedExperimentConditionCount = selectedExperimentConditionIds.length;

  const toggleExperimentConditionSelection = useCallback((conditionId) => {
    const safeId = String(conditionId || "").trim();
    if (!safeId) return;
    setExperimentConfig((prev) => {
      const prevIds = Array.isArray(prev.selectedConditionIds) ? prev.selectedConditionIds : [];
      const exists = prevIds.includes(safeId);
      const nextIds = exists
        ? prevIds.filter((id) => id !== safeId)
        : [...prevIds, safeId];
      return { ...prev, selectedConditionIds: nextIds };
    });
  }, []);

  const setExperimentConditionSelection = useCallback((conditionIds) => {
    const normalized = Array.isArray(conditionIds)
      ? conditionIds.map((id) => String(id || "").trim()).filter(Boolean)
      : [];
    const allowed = new Set(EXPERIMENT_CONDITION_MATRIX.map((condition) => condition.id));
    const nextIds = normalized.filter((id) => allowed.has(id));
    setExperimentConfig((prev) => ({ ...prev, selectedConditionIds: nextIds }));
  }, []);

  const applyExperimentConfigFromJson = useCallback((payload, sourceLabel = "json") => {
    if (experimentRunner.running) {
      setErrorMsg("Cannot load JSON config while experiment automation is running.");
      return false;
    }
    if (!payload || typeof payload !== "object") {
      setErrorMsg(`Invalid experiment config JSON (${sourceLabel}): root must be an object.`);
      setExperimentConfigJsonStatus(`JSON load failed (${sourceLabel}).`);
      return false;
    }

    const schemaId = typeof payload.schema === "string" ? payload.schema.trim() : "";
    if (schemaId && schemaId !== EXPERIMENT_CONFIG_SCHEMA_ID) {
      setErrorMsg(`Invalid config schema: ${schemaId}`);
      setExperimentConfigJsonStatus(`JSON schema mismatch (${sourceLabel}).`);
      return false;
    }
    const configNode = payload?.config && typeof payload.config === "object"
      ? payload.config
      : payload;

    const allowedConditionIds = new Set(EXPERIMENT_CONDITION_MATRIX.map((condition) => condition.id));
    const providedConditionIds = Array.isArray(configNode.selectedConditionIds)
      ? configNode.selectedConditionIds.map((id) => String(id || "").trim()).filter(Boolean)
      : null;
    const invalidConditionIds = Array.isArray(providedConditionIds)
      ? providedConditionIds.filter((id) => !allowedConditionIds.has(id))
      : [];

    setExperimentConfig((prev) => {
      const repeats = Number.isFinite(configNode.repeats)
        ? Math.max(1, Math.min(6, Math.round(configNode.repeats)))
        : prev.repeats;
      const runSeconds = Number.isFinite(configNode.runSeconds)
        ? Math.max(20, Math.min(240, Math.round(configNode.runSeconds)))
        : prev.runSeconds;
      const saveMode = (
        configNode.saveMode === EXPERIMENT_SAVE_MODES.SINGLE_BUNDLE_END
        || configNode.saveMode === EXPERIMENT_SAVE_MODES.SPLIT_PER_RUN
      ) ? configNode.saveMode : prev.saveMode;
      const startAttemptsPerRun = Number.isFinite(configNode.startAttemptsPerRun)
        ? Math.max(1, Math.min(5, Math.round(configNode.startAttemptsPerRun)))
        : prev.startAttemptsPerRun;
      const selectedConditionIds = Array.isArray(providedConditionIds)
        ? providedConditionIds.filter((id) => allowedConditionIds.has(id))
        : prev.selectedConditionIds;
      const includeHtmlReport = typeof configNode.includeHtmlReport === "boolean"
        ? configNode.includeHtmlReport
        : prev.includeHtmlReport;
      const includeAllLogsBundle = typeof configNode.includeAllLogsBundle === "boolean"
        ? configNode.includeAllLogsBundle
        : prev.includeAllLogsBundle;

      return {
        ...prev,
        repeats,
        runSeconds,
        saveMode,
        startAttemptsPerRun,
        selectedConditionIds,
        includeHtmlReport,
        includeAllLogsBundle
      };
    });

    const envNode = payload?.environment && typeof payload.environment === "object"
      ? payload.environment
      : {};
    const selectedModelCandidate = [configNode.selectedModel, envNode.selectedModel]
      .find((value) => typeof value === "string" && value.trim().length > 0);
    if (typeof selectedModelCandidate === "string") {
      const normalizedModel = selectedModelCandidate.trim();
      if (MODEL_OPTIONS.includes(normalizedModel)) {
        setSelectedModel(normalizedModel);
      }
    }

    const physicsNode = envNode.physicsSettings && typeof envNode.physicsSettings === "object"
      ? envNode.physicsSettings
      : (configNode.physicsSettings && typeof configNode.physicsSettings === "object" ? configNode.physicsSettings : null);
    if (physicsNode) {
      setPhysicsSettings((prev) => {
        const next = { ...prev };
        if (Number.isFinite(physicsNode.speedForce)) next.speedForce = Math.max(1000, Math.min(10000, Math.round(physicsNode.speedForce)));
        if (Number.isFinite(physicsNode.turnTorque)) next.turnTorque = Math.max(100, Math.min(2000, Math.round(physicsNode.turnTorque)));
        if (typeof physicsNode.sensorDynamic === "boolean") next.sensorDynamic = physicsNode.sensorDynamic;
        if (Number.isFinite(physicsNode.sensorRangeMin)) next.sensorRangeMin = Math.max(4, Math.min(12, Number(physicsNode.sensorRangeMin)));
        if (Number.isFinite(physicsNode.sensorRangeMax)) next.sensorRangeMax = Math.max(8, Math.min(20, Number(physicsNode.sensorRangeMax)));
        if (next.sensorRangeMax < next.sensorRangeMin + 0.5) {
          next.sensorRangeMax = next.sensorRangeMin + 0.5;
        }
        return next;
      });
    }

    const droppedText = invalidConditionIds.length > 0
      ? ` Dropped unknown IDs: ${invalidConditionIds.join(", ")}.`
      : "";
    const schemaText = schemaId
      ? ` schema=${schemaId}`
      : "";
    const versionText = Number.isFinite(payload.version)
      ? ` v${Math.round(payload.version)}`
      : "";
    setExperimentConfigJsonStatus(`Loaded from ${sourceLabel}${schemaText}${versionText}.${droppedText}`.trim());
    setErrorMsg(null);
    return true;
  }, [experimentRunner.running]);

  const onExperimentConfigJsonFileSelected = useCallback((event) => {
    const input = event?.target;
    const file = input?.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = String(reader.result || "");
        const parsed = JSON.parse(text);
        applyExperimentConfigFromJson(parsed, file.name);
      } catch (err) {
        setErrorMsg(`JSON parse error (${file.name}): ${err?.message || "invalid JSON"}`);
        setExperimentConfigJsonStatus(`JSON load failed (${file.name}).`);
      } finally {
        if (input) input.value = "";
      }
    };
    reader.onerror = () => {
      setErrorMsg(`Failed to read config file: ${file.name}`);
      setExperimentConfigJsonStatus(`JSON load failed (${file.name}).`);
      if (input) input.value = "";
    };
    reader.readAsText(file);
  }, [applyExperimentConfigFromJson]);

  const openExperimentConfigJsonPicker = useCallback(() => {
    if (experimentRunner.running) {
      setErrorMsg("Cannot load JSON config while experiment automation is running.");
      return;
    }
    experimentConfigFileInputRef.current?.click();
  }, [experimentRunner.running]);

  const downloadExperimentConfigJson = useCallback(() => {
    const now = Date.now();
    const safeConfig = {
      repeats: Math.max(1, Math.min(6, Math.round(asFiniteNumber(experimentConfig.repeats, 1)))),
      runSeconds: Math.max(20, Math.min(240, Math.round(asFiniteNumber(experimentConfig.runSeconds, 75)))),
      selectedConditionIds: selectedExperimentConditionIds,
      saveMode: experimentConfig.saveMode === EXPERIMENT_SAVE_MODES.SPLIT_PER_RUN
        ? EXPERIMENT_SAVE_MODES.SPLIT_PER_RUN
        : EXPERIMENT_SAVE_MODES.SINGLE_BUNDLE_END,
      startAttemptsPerRun: Math.max(
        1,
        Math.min(5, Math.round(asFiniteNumber(experimentConfig.startAttemptsPerRun, EXPERIMENT_START_MAX_ATTEMPTS)))
      ),
      includeHtmlReport: !!experimentConfig.includeHtmlReport,
      includeAllLogsBundle: !!experimentConfig.includeAllLogsBundle
    };
    const payload = {
      schema: EXPERIMENT_CONFIG_SCHEMA_ID,
      version: EXPERIMENT_CONFIG_SCHEMA_VERSION,
      exportedAt: new Date(now).toISOString(),
      config: safeConfig,
      environment: {
        selectedModel,
        physicsSettings: {
          speedForce: asFiniteNumber(physicsSettings.speedForce, 8000),
          turnTorque: asFiniteNumber(physicsSettings.turnTorque, 200),
          sensorDynamic: physicsSettings.sensorDynamic !== false,
          sensorRangeMin: asFiniteNumber(physicsSettings.sensorRangeMin, 7),
          sensorRangeMax: asFiniteNumber(physicsSettings.sensorRangeMax, 14)
        }
      },
      conditionCatalog: EXPERIMENT_CONDITION_MATRIX.map((condition) => ({
        id: condition.id,
        label: condition.label,
        model: condition.model
      }))
    };
    const prefix = buildSessionPrefix();
    const stamp = formatStamp(now);
    triggerDownload(
      JSON.stringify(payload, null, 2),
      `${prefix}_experiment_config_${stamp}.json`,
      "application/json"
    );
    setExperimentConfigJsonStatus(`Exported JSON config at ${new Date(now).toLocaleTimeString()}.`);
  }, [
    buildSessionPrefix,
    experimentConfig,
    physicsSettings,
    selectedExperimentConditionIds,
    selectedModel,
    triggerDownload
  ]);

  const togglePanel = (key) => {
    setPanelVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleCalibrationEnabled = () => {
    const state = directionCalibrationRef.current;
    state.enabled = !state.enabled;
    if (!state.enabled) {
      state.profile = {
        ...state.profile,
        applied: false,
        status: "DISABLED",
        reason: "manual_disable"
      };
    } else {
      state.profile = {
        ...state.profile,
        status: "COLLECTING",
        reason: "manual_enable"
      };
    }
    setDirectionCalibView({ ...state.profile });
  };

  const toggleCalibrationAutoApply = () => {
    const state = directionCalibrationRef.current;
    state.autoApply = !state.autoApply;
    if (!state.autoApply) {
      state.profile = {
        ...state.profile,
        applied: false,
        status: "MONITORING",
        reason: "manual_monitor_only"
      };
    }
    setDirectionCalibView({ ...state.profile });
  };

  const calibrationHeadingMapDeg = directionCalibView.latestHeadingDeg === null
    ? null
    : (180 - directionCalibView.latestHeadingDeg);
  const calibrationMotionMapDeg = directionCalibView.latestMotionHeadingDeg === null
    ? null
    : (180 - directionCalibView.latestMotionHeadingDeg);
  const headingMotionDeltaDeg = (directionCalibView.latestHeadingDeg === null || directionCalibView.latestMotionHeadingDeg === null)
    ? null
    : Math.abs(angleDiffDeg(directionCalibView.latestMotionHeadingDeg, directionCalibView.latestHeadingDeg));
  const preflightOverallTone = preflightReport.overall === "PASS"
    ? "text-emerald-300"
    : preflightReport.overall === "WARN"
      ? "text-amber-300"
      : preflightReport.overall === "FAIL"
        ? "text-red-300"
        : "text-slate-300";
  const preflightStatusBadgeClass = (status) => {
    if (status === "PASS") return "bg-emerald-700/70 border-emerald-400 text-emerald-100";
    if (status === "WARN") return "bg-amber-700/70 border-amber-400 text-amber-100";
    if (status === "FAIL") return "bg-red-700/70 border-red-400 text-red-100";
    if (status === "RUNNING") return "bg-blue-700/70 border-blue-400 text-blue-100";
    return "bg-slate-700/70 border-slate-400 text-slate-100";
  };
  const preflightStatusLabel = isPreflightRunning ? "RUNNING" : (preflightReport.status || "IDLE");

  return (
    <div className="relative w-screen h-screen">
      <GameScene
        onSensorUpdate={updateSensorData}
        onCollisionEvent={recordCollisionEvent}
        lastAction={lastAction}
        controls={controls}
        targetPosition={targetPosition}
        carResetNonce={carResetNonce}
        carSpawnPosition={DEFAULT_CAR_SPAWN.position}
        carSpawnRotation={DEFAULT_CAR_SPAWN.rotation}
        physicsSettings={physicsSettings}
        worldBounds={TRACK_WORLD_BOUNDS}
      />

      <div className="absolute top-4 left-4 pointer-events-auto z-50 p-3 rounded-xl border border-slate-500/50 bg-black/60 backdrop-blur-md font-mono text-white w-[260px]">
        <div className="text-[11px] uppercase tracking-wide text-slate-300 mb-2">Panel Visibility</div>
        <div className="grid grid-cols-2 gap-2">
          <button onClick={() => togglePanel("priority")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.priority ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Priority</button>
          <button onClick={() => togglePanel("physics")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.physics ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Physics</button>
          <button onClick={() => togglePanel("memory")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.memory ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Weight Map</button>
          <button onClick={() => togglePanel("calibration")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.calibration ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Calib</button>
          <button onClick={() => togglePanel("hud")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.hud ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Main HUD</button>
          <button onClick={() => togglePanel("controls")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.controls ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Drive UI</button>
          <button onClick={() => togglePanel("logs")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.logs ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Logs</button>
          <button onClick={() => togglePanel("branding")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.branding ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Branding</button>
          <button onClick={() => togglePanel("preflight")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.preflight ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Preflight</button>
          <button onClick={() => togglePanel("experiment")} className={`px-2 py-1 text-[10px] rounded border ${panelVisibility.experiment ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-800/70 border-slate-500"}`}>Experiment</button>
          <button onClick={() => setShowAnalyticsPanel(v => !v)} className={`px-2 py-1 text-[10px] rounded border ${showAnalyticsPanel ? "bg-blue-700/70 border-blue-300" : "bg-slate-800/70 border-slate-500"}`}>Analytics</button>
        </div>
      </div>

      {panelVisibility.priority && (
      <div className="absolute top-4 left-1/2 -translate-x-1/2 w-[min(860px,calc(100vw-2rem))] px-4 py-3 rounded-xl border border-cyan-400/30 bg-slate-950/78 backdrop-blur-md font-mono text-white pointer-events-none z-40">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-x-4 gap-y-2 text-[11px]">
          <div>
            <div className="text-gray-400">Run</div>
            <div className="font-bold text-cyan-300">{priorityStatus.runState}</div>
          </div>
          <div>
            <div className="text-gray-400">Target</div>
            <div className="font-bold text-cyan-200">{priorityStatus.targetDistance.toFixed(1)}m / {priorityStatus.targetAngle.toFixed(0)}deg / hit {priorityStatus.targetHitCount}</div>
          </div>
          <div>
            <div className="text-gray-400">Obstacle</div>
            <div className={`font-bold ${priorityStatus.riskTone}`}>{priorityStatus.minObstacleDist.toFixed(1)}m ({priorityStatus.dangerLevel})</div>
          </div>
          <div>
            <div className="text-gray-400">AI</div>
            <div className="font-bold text-violet-200">{priorityStatus.aiMode} ({priorityStatus.aiConfidence.toFixed(2)})</div>
          </div>
          <div>
            <div className="text-gray-400">Memory</div>
            <div className="font-bold text-emerald-300">{priorityStatus.preferredSector} / loop {priorityStatus.loopRate.toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-gray-400">Action</div>
            <div className="font-bold text-amber-300">{priorityStatus.action} | {priorityStatus.aiLatency}ms</div>
          </div>
        </div>
      </div>
      )}

      {/* Control Panel (Top Right) */}
      {panelVisibility.physics && (
      <div className="absolute top-4 right-4 p-4 bg-black/80 text-white rounded-lg font-mono w-[300px] pointer-events-auto">
        <h2 className="text-lg font-bold mb-2 text-yellow-400">Physics Tuning</h2>

        <div className="mb-3">
          <label className="text-xs text-gray-400 flex justify-between">
            <span>LLM Model</span>
            <span>{selectedModel}</span>
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={autoDrive || experimentRunner.running}
            className="w-full mt-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs text-slate-100 disabled:opacity-60"
          >
            {MODEL_OPTIONS.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>

        {/* Speed Force Slider */}
        <div className="mb-3">
          <label className="text-xs text-gray-400 flex justify-between">
            <span>Speed Force</span>
            <span>{physicsSettings.speedForce}</span>
          </label>
          <input
            type="range"
            min="1000"
            max="10000"
            step="100"
            value={physicsSettings.speedForce}
            onChange={(e) => setPhysicsSettings(prev => ({ ...prev, speedForce: parseInt(e.target.value) }))}
            className="w-full accent-yellow-500"
          />
        </div>

        {/* Turn Torque Slider */}
        <div className="mb-3">
          <label className="text-xs text-gray-400 flex justify-between">
            <span>Turn Sens.</span>
            <span>{physicsSettings.turnTorque}</span>
          </label>
          <input
            type="range"
            min="100"
            max="2000"
            step="50"
            value={physicsSettings.turnTorque}
            onChange={(e) => setPhysicsSettings(prev => ({ ...prev, turnTorque: parseInt(e.target.value) }))}
            className="w-full accent-yellow-500"
          />
        </div>

        <div className="mb-3">
          <label className="text-xs text-gray-400 flex justify-between">
            <span>Sensor Dynamic</span>
            <span>{physicsSettings.sensorDynamic ? "ON" : "OFF"}</span>
          </label>
          <button
            onClick={() => setPhysicsSettings(prev => ({ ...prev, sensorDynamic: !prev.sensorDynamic }))}
            className={`w-full mt-1 px-2 py-1 rounded border text-xs ${physicsSettings.sensorDynamic ? "bg-emerald-700/70 border-emerald-300" : "bg-slate-700/70 border-slate-400"}`}
          >
            {physicsSettings.sensorDynamic ? "Adaptive Range Enabled" : "Adaptive Range Disabled"}
          </button>
        </div>

        <div className="mb-3">
          <label className="text-xs text-gray-400 flex justify-between">
            <span>Sensor Min</span>
            <span>{physicsSettings.sensorRangeMin.toFixed(1)}m</span>
          </label>
          <input
            type="range"
            min="4"
            max="12"
            step="0.5"
            value={physicsSettings.sensorRangeMin}
            onChange={(e) => setPhysicsSettings(prev => {
              const minValue = parseFloat(e.target.value);
              return { ...prev, sensorRangeMin: minValue, sensorRangeMax: Math.max(minValue + 0.5, prev.sensorRangeMax) };
            })}
            className="w-full accent-cyan-500"
          />
        </div>

        <div className="mb-1">
          <label className="text-xs text-gray-400 flex justify-between">
            <span>Sensor Max</span>
            <span>{physicsSettings.sensorRangeMax.toFixed(1)}m</span>
          </label>
          <input
            type="range"
            min="8"
            max="20"
            step="0.5"
            value={physicsSettings.sensorRangeMax}
            onChange={(e) => setPhysicsSettings(prev => {
              const maxValue = parseFloat(e.target.value);
              return { ...prev, sensorRangeMax: Math.max(prev.sensorRangeMin + 0.5, maxValue) };
            })}
            className="w-full accent-cyan-500"
          />
        </div>
      </div>
      )}

      {panelVisibility.calibration && (
      <div className="absolute top-[210px] right-4 p-4 bg-black/85 text-white rounded-lg font-mono w-[300px] pointer-events-auto border border-cyan-500/40">
        <h2 className="text-sm font-bold mb-2 text-cyan-300 uppercase tracking-wide">Direction Calibration</h2>
        <div className="text-[11px] grid grid-cols-2 gap-x-2 gap-y-1 mb-3">
          <span>Status: <span className={directionCalibView.applied ? "text-emerald-300 font-bold" : "text-amber-300 font-bold"}>{directionCalibView.status}</span></span>
          <span>Reason: <span className="text-slate-300">{directionCalibView.reason}</span></span>
          <span>Heading Conf: <span className="text-cyan-200">{(directionCalibView.headingConfidence * 100).toFixed(1)}%</span></span>
          <span>Steer Conf: <span className="text-cyan-200">{(directionCalibView.steeringConfidence * 100).toFixed(1)}%</span></span>
          <span>Motion Samples: {directionCalibView.motionSampleCount}</span>
          <span>Steer Samples: {directionCalibView.steeringSampleCount}</span>
          <span>Heading Sign: {directionCalibView.headingSign >= 0 ? "+" : "-"}1</span>
          <span>Heading Offset: {directionCalibView.headingOffsetDeg.toFixed(1)}deg</span>
          <span>Steering Sign: {directionCalibView.steeringSign >= 0 ? "+" : "-"}1</span>
          <span>Motion Error: {directionCalibView.motionErrorDeg.toFixed(1)}deg</span>
        </div>

        <div className="mb-2">
          <div className="text-[10px] text-slate-300 mb-1">Heading Confidence</div>
          <div className="h-2 bg-slate-800 rounded">
            <div className="h-2 bg-cyan-500 rounded" style={{ width: `${Math.round(clamp01(directionCalibView.headingConfidence) * 100)}%` }} />
          </div>
        </div>
        <div className="mb-3">
          <div className="text-[10px] text-slate-300 mb-1">Steering Consistency</div>
          <div className="h-2 bg-slate-800 rounded">
            <div className="h-2 bg-emerald-500 rounded" style={{ width: `${Math.round(clamp01(directionCalibView.steeringAgreement) * 100)}%` }} />
          </div>
        </div>

        <div className="bg-slate-950/70 border border-slate-700 rounded p-2 mb-3">
          <div className="text-[10px] text-slate-300 mb-1">Live Alignment (white=heading, green=motion)</div>
          <svg width="100%" viewBox="0 0 120 80" className="overflow-visible">
            <circle cx="60" cy="40" r="22" fill="none" stroke="#334155" strokeWidth="1.2" />
            <line x1="60" y1="18" x2="60" y2="6" stroke="#334155" strokeWidth="1" />
            <line x1="82" y1="40" x2="94" y2="40" stroke="#334155" strokeWidth="1" />
            <line x1="60" y1="62" x2="60" y2="74" stroke="#334155" strokeWidth="1" />
            <line x1="38" y1="40" x2="26" y2="40" stroke="#334155" strokeWidth="1" />
            {calibrationHeadingMapDeg !== null && (
              <g transform={`translate(60 40) rotate(${calibrationHeadingMapDeg})`}>
                <polygon points="0,-26 4,-14 -4,-14" fill="#f8fafc" />
              </g>
            )}
            {calibrationMotionMapDeg !== null && (
              <g transform={`translate(60 40) rotate(${calibrationMotionMapDeg})`}>
                <polygon points="0,-22 3,-12 -3,-12" fill="#22c55e" />
              </g>
            )}
          </svg>
          <div className="text-[10px] text-slate-300">
            Delta: {headingMotionDeltaDeg === null ? "N/A" : `${headingMotionDeltaDeg.toFixed(1)}deg`}
          </div>
        </div>

        <div className="flex gap-2">
          <button onClick={toggleCalibrationEnabled} className={`px-2 py-1 text-[10px] rounded border ${directionCalibrationRef.current.enabled ? "bg-emerald-700/70 border-emerald-300" : "bg-slate-700/70 border-slate-400"}`}>
            {directionCalibrationRef.current.enabled ? "Calibration ON" : "Calibration OFF"}
          </button>
          <button onClick={toggleCalibrationAutoApply} className={`px-2 py-1 text-[10px] rounded border ${directionCalibrationRef.current.autoApply ? "bg-cyan-700/70 border-cyan-300" : "bg-slate-700/70 border-slate-400"}`}>
            {directionCalibrationRef.current.autoApply ? "Auto Apply" : "Monitor Only"}
          </button>
          <button onClick={resetDirectionCalibration} className="px-2 py-1 text-[10px] rounded border bg-slate-700/70 border-slate-400">
            Reset
          </button>
        </div>
      </div>
      )}

      {panelVisibility.preflight && (
      <div className="absolute top-[470px] right-4 p-4 bg-black/85 text-white rounded-lg font-mono w-[300px] pointer-events-auto border border-sky-500/40">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-bold text-sky-300 uppercase tracking-wide">Startup Preflight</h2>
          <span className={`px-2 py-0.5 text-[10px] rounded border ${preflightStatusBadgeClass(preflightStatusLabel)}`}>
            {preflightStatusLabel}
          </span>
        </div>
        <div className="text-[11px] grid grid-cols-2 gap-x-2 gap-y-1 mb-2">
          <span>Policy:</span>
          <span className="text-sky-200">{preflightReport.gatePolicy || PREFLIGHT_GATE_POLICY}</span>
          <span>Overall:</span>
          <span className={`font-bold ${preflightOverallTone}`}>{preflightReport.overall}</span>
          <span>Non-PASS:</span>
          <span>{preflightReport.nonPassCount ?? 0}</span>
          <span>Blocking Fail:</span>
          <span>{preflightReport.blockingFailures}</span>
          <span>Warnings:</span>
          <span>{preflightReport.warningCount}</span>
        </div>
        <div className="text-[10px] text-slate-300 mb-2 leading-tight">{preflightReport.summary}</div>
        <div className="text-[10px] text-slate-400 mb-2">
          Started: {preflightReport.startedAt ? new Date(preflightReport.startedAt).toLocaleTimeString() : "-"} | Ended: {preflightReport.finishedAt ? new Date(preflightReport.finishedAt).toLocaleTimeString() : "-"}
        </div>
        <div className="space-y-1 max-h-[170px] overflow-y-auto pr-1 mb-2">
          {(preflightReport.checks || []).map((check) => (
            <div key={check.id} className="border border-slate-700/60 rounded p-1.5 bg-slate-950/70">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] text-slate-200">{check.label}</span>
                <span className={`px-1.5 py-0.5 text-[9px] rounded border ${preflightStatusBadgeClass(check.status)}`}>{check.status}</span>
              </div>
              <div className="text-[9px] text-slate-400 mt-1 leading-tight">{check.detail || "-"}</div>
            </div>
          ))}
        </div>
        <div className="text-[10px] text-slate-300 border-t border-slate-700 pt-2">
          AI Scenario Probe ({preflightReport.aiScenarioResults?.length || 0}/{PREFLIGHT_AI_SCENARIOS.length})
        </div>
        <div className="space-y-1 max-h-[110px] overflow-y-auto pr-1 mt-1">
          {(preflightReport.aiScenarioResults || []).map((result) => (
            <div key={result.name} className="text-[9px] border border-slate-700/60 rounded p-1 bg-slate-950/70">
              <div className="flex justify-between">
                <span className="text-slate-200">{result.name}</span>
                <span className={result.aligned ? "text-emerald-300" : "text-amber-300"}>{result.aligned ? "aligned" : "off"}</span>
              </div>
              <div className="text-slate-400">mode={result.mode} skill={result.skill} parse={result.parseMethod}</div>
            </div>
          ))}
        </div>
      </div>
      )}

      {panelVisibility.experiment && (
      <div className="absolute top-[760px] right-4 p-4 bg-black/85 text-white rounded-lg font-mono w-[300px] pointer-events-auto border border-fuchsia-500/40 max-h-[320px] overflow-y-auto">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-bold text-fuchsia-300 uppercase tracking-wide">Experiment Automation</h2>
          <span className={`px-2 py-0.5 text-[10px] rounded border ${experimentRunner.running ? "bg-fuchsia-700/70 border-fuchsia-300 text-fuchsia-100" : "bg-slate-700/70 border-slate-400 text-slate-100"}`}>
            {experimentRunner.phase}
          </span>
        </div>

        <div className="text-[11px] grid grid-cols-2 gap-x-2 gap-y-1 mb-3">
          <span>Model:</span>
          <span className="text-fuchsia-200">{selectedModel}</span>
          <span>Runs:</span>
          <span>{experimentRunner.currentRun}/{experimentRunner.totalRuns}</span>
          <span>Selected:</span>
          <span>{selectedExperimentConditionCount}/{EXPERIMENT_CONDITION_MATRIX.length}</span>
          <span>Condition:</span>
          <span className="truncate">{experimentRunner.activeConditionId}</span>
        </div>

        <div className="grid grid-cols-2 gap-2 mb-3">
          <label className="text-[10px] text-slate-300">
            Repeats
            <input
              type="number"
              min="1"
              max="6"
              value={experimentConfig.repeats}
              disabled={experimentRunner.running}
              onChange={(e) => setExperimentConfig((prev) => ({ ...prev, repeats: Math.max(1, Math.min(6, parseInt(e.target.value || "1", 10) || 1)) }))}
              className="w-full mt-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs"
            />
          </label>
          <label className="text-[10px] text-slate-300">
            Run Seconds
            <input
              type="number"
              min="20"
              max="240"
              value={experimentConfig.runSeconds}
              disabled={experimentRunner.running}
              onChange={(e) => setExperimentConfig((prev) => ({ ...prev, runSeconds: Math.max(20, Math.min(240, parseInt(e.target.value || "75", 10) || 75)) }))}
              className="w-full mt-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs"
            />
          </label>
        </div>

        <div className="space-y-1 mb-3 text-[10px]">
          <label className="flex flex-col gap-1">
            <span className="text-slate-300">Save Mode</span>
            <select
              value={experimentConfig.saveMode || EXPERIMENT_SAVE_MODES.SINGLE_BUNDLE_END}
              disabled={experimentRunner.running}
              onChange={(e) => setExperimentConfig((prev) => ({ ...prev, saveMode: e.target.value }))}
              className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs text-slate-100 disabled:opacity-60"
            >
              <option value={EXPERIMENT_SAVE_MODES.SINGLE_BUNDLE_END}>One file at end (recommended)</option>
              <option value={EXPERIMENT_SAVE_MODES.SPLIT_PER_RUN}>Per-run separate files</option>
            </select>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={!!experimentConfig.includeHtmlReport}
              disabled={experimentRunner.running}
              onChange={(e) => setExperimentConfig((prev) => ({ ...prev, includeHtmlReport: e.target.checked }))}
              className="accent-fuchsia-500"
            />
            Include HTML report in experiment export
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={!!experimentConfig.includeAllLogsBundle}
              disabled={experimentRunner.running}
              onChange={(e) => setExperimentConfig((prev) => ({ ...prev, includeAllLogsBundle: e.target.checked }))}
              className="accent-fuchsia-500"
            />
            Include AI review bundle payload
          </label>
        </div>

        <div className="grid grid-cols-2 gap-1 mb-1">
          <button
            onClick={downloadExperimentConfigJson}
            disabled={experimentRunner.running}
            className="px-2 py-1 text-[9px] rounded border bg-indigo-700/70 border-indigo-400 disabled:opacity-50"
          >
            SAVE JSON CFG
          </button>
          <button
            onClick={openExperimentConfigJsonPicker}
            disabled={experimentRunner.running}
            className="px-2 py-1 text-[9px] rounded border bg-indigo-700/70 border-indigo-400 disabled:opacity-50"
          >
            LOAD JSON CFG
          </button>
        </div>
        <div className="text-[9px] text-slate-400 leading-tight mb-3">
          {experimentConfigJsonStatus}
        </div>
        <input
          ref={experimentConfigFileInputRef}
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={onExperimentConfigJsonFileSelected}
        />

        <div className="flex gap-2 mb-3">
          <button
            onClick={runExperimentAutomation}
            disabled={experimentRunner.running || autoDrive || isAnalyzing || isPreflightRunning || selectedExperimentConditionCount === 0}
            className="flex-1 px-3 py-2 rounded border bg-fuchsia-700/80 border-fuchsia-400 text-xs disabled:opacity-50 disabled:cursor-not-allowed"
          >
            START MATRIX
          </button>
          <button
            onClick={stopExperimentAutomation}
            disabled={!experimentRunner.running}
            className="flex-1 px-3 py-2 rounded border bg-rose-700/80 border-rose-400 text-xs disabled:opacity-50 disabled:cursor-not-allowed"
          >
            STOP
          </button>
        </div>

        <div className="text-[10px] text-slate-300 leading-tight border border-slate-700 rounded p-2 bg-slate-950/70 mb-2">
          {experimentRunner.summary}
        </div>

        <div className="text-[10px] text-slate-300 border-t border-slate-700 pt-2 mb-1">
          Condition Matrix ({selectedExperimentConditionCount} selected)
        </div>
        <div className="grid grid-cols-2 gap-1 mb-2">
          <button
            onClick={() => setExperimentConditionSelection(EXPERIMENT_CONDITION_MATRIX.map((condition) => condition.id))}
            disabled={experimentRunner.running}
            className="px-2 py-1 text-[9px] rounded border bg-slate-700/70 border-slate-400 disabled:opacity-50"
          >
            SELECT ALL
          </button>
          <button
            onClick={() => setExperimentConditionSelection([])}
            disabled={experimentRunner.running}
            className="px-2 py-1 text-[9px] rounded border bg-slate-700/70 border-slate-400 disabled:opacity-50"
          >
            CLEAR
          </button>
          <button
            onClick={() => setExperimentConditionSelection(["AB-1", "AB-4"])}
            disabled={experimentRunner.running}
            className="px-2 py-1 text-[9px] rounded border bg-fuchsia-700/70 border-fuchsia-400 disabled:opacity-50"
          >
            AB-1 + AB-4
          </button>
          <button
            onClick={() => setExperimentConditionSelection(["AB-1", "AB-2"])}
            disabled={experimentRunner.running}
            className="px-2 py-1 text-[9px] rounded border bg-fuchsia-700/70 border-fuchsia-400 disabled:opacity-50"
          >
            AB-1 + AB-2
          </button>
        </div>
        <div className="space-y-1 max-h-[120px] overflow-y-auto pr-1">
          {EXPERIMENT_CONDITION_MATRIX.map((condition) => (
            <div key={condition.id} className="text-[9px] border border-slate-700/60 rounded p-1 bg-slate-950/60">
              <div className="flex justify-between items-center">
                <label className="flex items-center gap-1 text-slate-200">
                  <input
                    type="checkbox"
                    checked={selectedExperimentConditionIds.includes(condition.id)}
                    disabled={experimentRunner.running}
                    onChange={() => toggleExperimentConditionSelection(condition.id)}
                    className="accent-fuchsia-500"
                  />
                  <span>{condition.id}</span>
                </label>
                <span className="text-fuchsia-200">{condition.model}</span>
              </div>
              <div className="text-slate-400">{condition.label}</div>
            </div>
          ))}
        </div>
      </div>
      )}

      {/* Exploration Memory Live Map */}
      {panelVisibility.memory && memoryMapView && (
        <div className="absolute top-4 right-[320px] p-3 bg-black/85 text-white rounded-lg font-mono w-[340px] pointer-events-none border border-cyan-600/40">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-sm font-bold text-cyan-300 uppercase tracking-wide">Exploration Weight Map</h2>
            <span className={`text-[10px] px-2 py-0.5 rounded ${memoryMapView.loopWarning === "HIGH_REVISIT_LOOP_RISK" ? "bg-red-700/80" : "bg-emerald-700/70"}`}>
              {memoryMapView.loopWarning}
            </span>
          </div>

          <div className="text-[11px] text-gray-300 mb-2 grid grid-cols-2 gap-x-2 gap-y-1">
            <span>Loop: {(memoryMapView.loopRate * 100).toFixed(1)}%</span>
            <span>Preferred: {memoryMapView.preferredSector}</span>
            <span>Mapped Cells: {memoryMapView.mappedCells}</span>
            <span>Cell Size: 2.0m</span>
            <span>Sensor Range: {asFiniteNumber(sensorData.sensorRange, 10).toFixed(1)}m</span>
            <span>Memory Range: {asFiniteNumber(memoryMapView.memorySensorRange, 10).toFixed(1)}m</span>
            <span>No-Go Cells: {(memoryMapView.noGoRatio * 100).toFixed(1)}%</span>
            <span>Safe Candidates: {memoryMapView.safeCandidates}/{memoryMapView.candidateCount}</span>
            <span>Pose: X {memoryMapView.worldPos.x.toFixed(1)} / Z {memoryMapView.worldPos.z.toFixed(1)}</span>
            <span>Heading: {memoryMapView.headingDeg.toFixed(0)}deg (map {memoryMapView.mapHeadingDeg.toFixed(0)}deg)</span>
          </div>

          <div className="bg-black/70 border border-gray-700 rounded overflow-hidden relative" style={{ width: `${memoryMapView.mapPx}px`, height: `${memoryMapView.mapPx}px` }}>
            <div className="absolute top-1 left-1/2 -translate-x-1/2 text-[9px] text-cyan-200 bg-black/65 px-1 rounded">{memoryMapView.mapCompass.top}</div>
            <div className="absolute right-1 top-1/2 -translate-y-1/2 text-[9px] text-cyan-200 bg-black/65 px-1 rounded">{memoryMapView.mapCompass.right}</div>
            <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[9px] text-cyan-200 bg-black/65 px-1 rounded">{memoryMapView.mapCompass.bottom}</div>
            <div className="absolute left-1 top-1/2 -translate-y-1/2 text-[9px] text-cyan-200 bg-black/65 px-1 rounded">{memoryMapView.mapCompass.left}</div>
            <svg width={memoryMapView.mapPx} height={memoryMapView.mapPx}>
              {memoryMapView.cells.map((cell) => (
                <rect
                  key={`${cell.ix},${cell.iz}`}
                  x={cell.x}
                  y={cell.y}
                  width={memoryMapView.cellPx}
                  height={memoryMapView.cellPx}
                  fill={cell.fill}
                  stroke={cell.isCurrent ? "#ffffff" : cell.isFrontier ? "#facc15" : cell.isRisky ? "#ef4444" : "rgba(17,24,39,0.4)"}
                  strokeWidth={cell.isCurrent ? 2 : 1}
                />
              ))}

              {memoryMapView.pathPoints.map((point, idx) => (
                <circle
                  key={`p-${idx}`}
                  cx={point.x}
                  cy={point.y}
                  r={2}
                  fill={`rgba(125,211,252,${point.alpha.toFixed(3)})`}
                />
              ))}

              <g transform={`translate(${memoryMapView.vehicleMarker.cx} ${memoryMapView.vehicleMarker.cy}) rotate(${memoryMapView.mapHeadingDeg})`}>
                <circle r={Math.max(3, Math.floor(memoryMapView.vehicleMarker.size * 0.26))} fill="#111827" stroke="#ffffff" strokeWidth="1" />
                <polygon
                  points={`0,${-memoryMapView.vehicleMarker.size} ${Math.round(memoryMapView.vehicleMarker.size * 0.64)},${Math.round(memoryMapView.vehicleMarker.size * 0.82)} ${-Math.round(memoryMapView.vehicleMarker.size * 0.64)},${Math.round(memoryMapView.vehicleMarker.size * 0.82)}`}
                  fill="#f8fafc"
                  stroke="#06b6d4"
                  strokeWidth="1.2"
                />
              </g>
            </svg>
          </div>

          <div className="mt-2 text-[10px] text-gray-400 leading-tight">
            Color: Green=explore-worthy, Red=risky/redundant, DarkRed=barrier, Navy=outside bounds, Yellow border=frontier, White border=current.
          </div>
          <div className="mt-1 text-[10px] text-cyan-200 leading-tight">
            Map axis is fixed to world coordinates. Current center cell: ({memoryMapView.centerCell.ix}, {memoryMapView.centerCell.iz}).
          </div>
          <div className="mt-1 text-[10px] text-cyan-300 leading-tight">
            Map is north-up. World heading: 0deg=South (+Z), 180deg=North (-Z).
          </div>
        </div>
      )}

      {/* HUD Overlay - 4K SCALE */}
      {panelVisibility.hud && (
      <div className="absolute top-8 left-8 p-8 bg-black/20 text-white rounded-2xl font-mono w-[900px] max-h-[calc(100vh-6rem)] overflow-y-auto pointer-events-none select-none backdrop-blur-xl border border-white/10">
        <h1 className="text-4xl font-bold mb-6 text-cyan-400 tracking-wider">Gemma 3 AutoPilot <span className="text-white opacity-50 text-2xl"> // Strategic</span></h1>

        <div className="space-y-4 mb-8">
          <div className="flex justify-between items-center bg-gray-800/80 p-4 rounded-xl mb-4 border border-gray-700">
            <span className="text-2xl text-gray-400">SCORE</span>
            <span className="text-5xl font-bold text-yellow-500">{score}</span>
          </div>

          <div className="flex justify-between text-2xl">
            <span>Target Dist: <span className="font-bold text-cyan-300">{sensorData.distanceToTarget?.toFixed(1)}m</span></span>
          </div>
          <div className="flex justify-between text-xl text-gray-400">
            <span>Target Angle: {sensorData.angleToTarget?.toFixed(0)}°</span>
          </div>

          <div className="h-px bg-gray-600 my-4"></div>

          <div className="flex justify-between text-xl font-mono tracking-tighter">
            <span className={sensorData.left < 5 ? "text-red-400" : ""}>L:{sensorData.left?.toFixed(0) ?? "?"}</span>
            <span className={sensorData.leftDiag < 5 ? "text-orange-400" : "text-gray-400"}>LD:{sensorData.leftDiag?.toFixed(0) ?? "?"}</span>
            <span className={sensorData.front < 8 ? "text-red-600 font-bold scale-110" : "text-green-400"}>
              F:{sensorData.front?.toFixed(0) ?? "?"}
            </span>
            <span className={sensorData.rightDiag < 5 ? "text-orange-400" : "text-gray-400"}>RD:{sensorData.rightDiag?.toFixed(0) ?? "?"}</span>
            <span className={sensorData.right < 5 ? "text-red-400" : ""}>R:{sensorData.right?.toFixed(0) ?? "?"}</span>
          </div>

          <div className="flex justify-between text-lg text-gray-400 mt-2">
            <span>Speed: {sensorData.speed?.toFixed(1)}</span>
            <span className={sensorData.isStuck ? "text-red-500 font-bold" : ""}>
              Stuck: {sensorData.isStuck ? "YES" : "NO"}
            </span>
          </div>
        </div>

        <div className="mb-8">
          <div className="text-xl text-gray-400 mb-2">Current Action:</div>
          <div className="text-6xl font-black text-yellow-400 tracking-widest uppercase filter drop-shadow-lg">
            {recoveryPhase ? `⚠ ${recoveryPhase} ⚠` : lastAction}
          </div>
          {errorMsg && (
            <div className="text-red-500 text-2xl mt-2 font-bold animate-pulse">
              ⚠ {errorMsg}
            </div>
          )}
        </div>

        {/* Thought Process - HERO SECTION */}
        <div className="mb-6 bg-blue-900/60 p-6 rounded-xl border-l-8 border-blue-500 shadow-lg">
          <div className="text-lg text-blue-300 font-bold mb-2 uppercase tracking-wide">✨ Gemma's Thought Process:</div>
          <div className="text-xl text-yellow-300 font-mono mb-2">
            "{aiStats.analysis || "Observing..."}"
          </div>
          <div className="text-3xl italic text-white leading-relaxed font-serif">
            "{aiStats.thought || "..."}"
          </div>
        </div>

        {/* Memory Display */}
        <div className="mb-6 flex items-center gap-4 text-gray-400 font-mono text-sm">
          <span>🧠 Memory:</span>
          <span className="text-cyan-300">{actionHistory.length > 0 ? actionHistory.join(" → ") : "Empty"}</span>
        </div>

        {/* AI Stats */}
        <div className="mb-6 bg-gray-900/80 p-4 rounded-lg border border-gray-700 min-h-[120px]">
          <div className="flex justify-between text-sm text-cyan-300 mb-2">
            <span>Latency: {aiStats.latency}ms</span>
            <span className="text-gray-500">{aiStats.timestamp}</span>
          </div>
          <div className="text-xs text-gray-400 font-mono break-all whitespace-pre-wrap leading-tight h-24 overflow-y-auto">
            {aiStats.raw || "Waiting for signal..."}
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className={`w-6 h-6 rounded-full ${isThinking ? "bg-green-500 animate-pulse shadow-[0_0_20px_#22c55e]" : "bg-gray-600"}`} />
          <span className="text-2xl">{isThinking ? "Thinking..." : "Manual Control"}</span>
        </div>

        {showAnalyticsPanel && (
        <div className="mb-6 bg-gray-900/95 p-6 rounded-xl border border-blue-500/50 shadow-2xl backdrop-blur-xl">
          <div className="flex justify-between items-end mb-4">
            <h3 className="text-blue-400 font-bold uppercase tracking-widest text-lg">📈 Tracking Correlation Analysis</h3>
            <div className="text-xs text-gray-500 font-mono">
              Target Angle (Cyan) vs Steering (Magenta) • {telemetry.length} Samples
            </div>
          </div>

          {/* MAIN CORRELATION GRAPH (Steering vs Angle) */}
          <div className="h-48 bg-black/60 rounded-lg border border-gray-700 relative overflow-hidden mb-4">
            <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
              {/* Center Line (0) */}
              <line x1="0" y1="50%" x2="100%" y2="50%" stroke="#4b5563" strokeWidth="1" />
              <g>
                <polyline points={correlationChart.anglePoints} fill="none" stroke="#22d3ee" strokeWidth="2" strokeOpacity="0.8" />
                <polyline points={correlationChart.steeringPoints} fill="none" stroke="#e879f9" strokeWidth="2" strokeOpacity="0.8" />
              </g>
            </svg>

            {/* Legend */}
            <div className="absolute top-2 right-2 flex flex-col gap-1 text-[10px] bg-black/50 p-2 rounded">
              <div className="flex items-center gap-2">
                <div className="w-3 h-0.5 bg-cyan-400"></div> <span className="text-cyan-400">Target Angle</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-0.5 bg-purple-400"></div> <span className="text-purple-400">Steering Output</span>
              </div>
            </div>
          </div>

          {/* SECONDARY METRIC: Intentionality Cumulative Mean */}
          <div className="h-24 bg-black/60 rounded-lg border border-gray-700 relative overflow-hidden">
            <div className="absolute top-2 left-2 text-[10px] text-green-400 font-bold">AVG INTENTIONALITY (Cumulative)</div>
            <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
              {/* 80% Threshold Line */}
              <line x1="0" y1="20%" x2="100%" y2="20%" stroke="#22c55e" strokeWidth="1" strokeDasharray="4 4" opacity="0.5" />
              <polyline points={intentionalityChart} fill="none" stroke="#4ade80" strokeWidth="2" />
            </svg>
          </div>
        </div>
        )}

        {!autoDrive && (
          <div className="mt-6 text-lg text-gray-400 border-t border-gray-600 pt-4">
            Manual Mode: Arrow Keys to Drive/Reverse.
          </div>
        )}
      </div>
      )}

      {panelVisibility.controls && (
      <div className="absolute bottom-16 left-1/2 -translate-x-1/2 pointer-events-auto z-50 flex flex-col items-center gap-4">

        {/* AUTO ANALYST MODULE */}
        <AutoAnalyst
          telemetry={telemetry}
          isRecording={autoDrive}
          onStart={() => {
            startAutodriveWithPreflight("auto_analyst");
          }}
          onStop={handleStopAutodrive}
          onAnalysisChange={setIsAnalyzing}
        />

        <button
          onClick={() => {
            if (isAnalyzing || isPreflightRunning) return;
            if (autoDrive) {
              handleStopAutodrive();
            } else {
              startAutodriveWithPreflight("engage_button");
            }
          }}
          disabled={isAnalyzing || isPreflightRunning}
          className={`px-12 py-6 rounded-full font-black text-3xl shadow-2xl transition-all transform hover:scale-110 active:scale-95 border-4 border-white/20 ${(isAnalyzing || isPreflightRunning) ? "bg-gray-600 opacity-50 cursor-not-allowed" :
            autoDrive
              ? "bg-red-600 hover:bg-red-700 shadow-red-900/50"
              : "bg-emerald-500 hover:bg-emerald-600 shadow-emerald-900/50"
            }`}
        >
          {isPreflightRunning ? "RUNNING PREFLIGHT..." : isAnalyzing ? "🤖 ANALYZING..." : autoDrive ? "🛑 STOP AUTOPILOT" : "🚀 ENGAGE AI DRIVER"}
        </button>

        <button
          onClick={() => setShowAnalyticsPanel(v => !v)}
          className="px-5 py-2 bg-blue-700/75 hover:bg-blue-600 text-white rounded-lg font-mono text-xs border border-blue-400 shadow-xl"
        >
          {showAnalyticsPanel ? "HIDE ANALYTICS PANEL" : "SHOW ANALYTICS PANEL"}
        </button>

      </div>
      )}

      {panelVisibility.logs && (
      <div className="absolute bottom-16 right-8 pointer-events-auto z-50 flex flex-col items-stretch gap-2 w-[280px] p-3 rounded-xl border border-slate-500/40 bg-black/55 backdrop-blur-md">
        <div className="text-[11px] font-mono uppercase tracking-wide text-slate-300">Log Downloads</div>
        <button
          onClick={handleSaveLogs}
          disabled={isAnalyzing}
          className="px-6 py-3 bg-gray-700/80 hover:bg-gray-600 text-white rounded-lg font-mono text-sm border border-gray-500 backdrop-blur-sm shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
        >
          SAVE ALL LOGS (1 FILE)
        </button>

        <button
          onClick={() => downloadDriveLogOnly("manual")}
          disabled={isAnalyzing}
          className="px-6 py-3 bg-slate-700/80 hover:bg-slate-600 text-white rounded-lg font-mono text-sm border border-slate-400 backdrop-blur-sm shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
        >
          DOWNLOAD DRIVE LOG
        </button>

        <button
          onClick={() => downloadTelemetryOnly("manual")}
          disabled={isAnalyzing}
          className="px-6 py-3 bg-slate-700/80 hover:bg-slate-600 text-white rounded-lg font-mono text-sm border border-slate-400 backdrop-blur-sm shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
        >
          DOWNLOAD TELEMETRY
        </button>

        <button
          onClick={() => downloadMetaOnly("manual")}
          disabled={isAnalyzing}
          className="px-6 py-3 bg-slate-700/80 hover:bg-slate-600 text-white rounded-lg font-mono text-sm border border-slate-400 backdrop-blur-sm shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
        >
          DOWNLOAD META
        </button>

        <button
          onClick={() => downloadLatestPreflightGateLog("manual")}
          disabled={isAnalyzing}
          className="px-6 py-3 bg-rose-700/80 hover:bg-rose-600 text-white rounded-lg font-mono text-sm border border-rose-400 backdrop-blur-sm shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
        >
          DOWNLOAD PREFLIGHT FAIL LOG
        </button>

        <button
          onClick={handleDownloadHtmlReport}
          disabled={isAnalyzing}
          className="px-6 py-3 bg-indigo-700/80 hover:bg-indigo-600 text-white rounded-lg font-mono text-sm border border-indigo-400 backdrop-blur-sm shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
        >
          DOWNLOAD HTML REPORT
        </button>
      </div>
      )}

      {panelVisibility.branding && (
      <div className="absolute bottom-8 right-8 text-xl text-gray-500 font-bold opacity-50">
        Powered by Gemma 3 & Ollama
      </div>
      )}
    </div>
  );
}
