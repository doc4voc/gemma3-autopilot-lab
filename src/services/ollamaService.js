export const OLLAMA_API_BASE = "http://localhost:11434/api";
export const OLLAMA_GENERATE = `${OLLAMA_API_BASE}/generate`;
export const OLLAMA_TAGS = `${OLLAMA_API_BASE}/tags`;
const OLLAMA_TIMEOUT_MS = 15000;
const MODE_SWITCH_COOLDOWN_MS = 2200;
const MODE_SWITCH_MIN_DWELL_MS = 1400;
const MODE_SWITCH_MIN_VOTES = 2;
const CRITICAL_FRONT_DIST = 2.15;
const TARGET_LOCK_PRIORITY_DISTANCE_M = 20;
const TARGET_LOCK_PRIORITY_ANGLE_DEG = 48;
const TARGET_LOCK_PRIORITY_PSEUDO = 0.5;
const TARGET_LOCK_HOLD_DISTANCE_M = 12;
const TARGET_LOCK_HOLD_ANGLE_DEG = 72;
const TARGET_LOCK_HOLD_PSEUDO = 0.42;
const TARGET_LOCK_HOLD_MIN_FRONT_DIST = 2.2;
const TARGET_LOCK_HOLD_MIN_CYCLES = 3;
const MODE_SWITCH_MIN_HOLD_CYCLES = 2;
const APPROACH_MIN_HOLD_CYCLES = 3;
const SKILL_BURST_SCAN_LIMIT = 2;
const SKILL_BURST_BACKOFF_LIMIT = 2;
const SKILL_BURST_RELEASE_FRONT_DIST = 3.2;
const MODEL_NUM_PREDICT_PRIMARY = 420;
const MODEL_NUM_PREDICT_RETRY = 360;
const ACTION_PLAN_MAX_STEPS = 5;
const ACTION_PLAN_MAX_TOTAL_SEC = 1.2;
const ACTION_SIGN_DEADZONE = 0.08;
const VALID_SKILLS = Object.freeze([
    "APPROACH_TARGET",
    "MOVE_TO_FRONTIER",
    "SCAN_SECTOR",
    "BACKOFF_AND_TURN",
    "HOLD_POSITION"
]);

const clampNumber = (value, min, max, fallback) => {
    const n = typeof value === "number" && Number.isFinite(value) ? value : fallback;
    if (!Number.isFinite(n)) return fallback;
    return Math.max(min, Math.min(max, n));
};

const asNum = (value, fallback = 0) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);
const clamp01 = (value) => Math.max(0, Math.min(1, asNum(value, 0)));

function signFromControlValue(value, deadzone = ACTION_SIGN_DEADZONE) {
    const n = asNum(value, 0);
    if (n > deadzone) return 1;
    if (n < -deadzone) return -1;
    return 0;
}

function normalizeReasonCode(raw, fallback = "UNSPECIFIED_REASON") {
    const source = typeof raw === "string" ? raw : "";
    const normalized = source
        .trim()
        .toUpperCase()
        .replace(/[\s-]+/g, "_")
        .replace(/[^A-Z0-9_]/g, "");
    if (!normalized) return fallback;
    return normalized.slice(0, 48);
}

function normalizeReasonSign(raw, fallback = null) {
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
}

function inferFallbackReasonShape(step, fallbackReason = {}) {
    const throttleSign = signFromControlValue(step?.throttle);
    const steeringSign = signFromControlValue(step?.steering);
    const fallbackCode = normalizeReasonCode(fallbackReason?.code, "UNSPECIFIED_REASON");
    const fallbackSummary = typeof fallbackReason?.summary === "string" ? fallbackReason.summary.trim() : "";
    if (fallbackSummary) {
        return {
            code: fallbackCode,
            summary: fallbackSummary.slice(0, 180),
            expectedThrottleSign: normalizeReasonSign(fallbackReason?.expectedThrottleSign, throttleSign),
            expectedSteeringSign: normalizeReasonSign(fallbackReason?.expectedSteeringSign, steeringSign)
        };
    }

    if (throttleSign > 0 && steeringSign === 0) {
        return {
            code: "FORWARD_PROBE",
            summary: "Advance straight toward safe frontier and target cue.",
            expectedThrottleSign: 1,
            expectedSteeringSign: 0
        };
    }
    if (throttleSign > 0 && steeringSign !== 0) {
        return {
            code: "FORWARD_TURN_APPROACH",
            summary: "Advance while turning to align with safer target direction.",
            expectedThrottleSign: 1,
            expectedSteeringSign: steeringSign
        };
    }
    if (throttleSign < 0) {
        return {
            code: "REVERSE_ESCAPE",
            summary: "Back off to avoid obstacle pressure and re-open path.",
            expectedThrottleSign: -1,
            expectedSteeringSign: steeringSign
        };
    }
    if (steeringSign !== 0) {
        return {
            code: "PIVOT_SCAN",
            summary: "Hold throttle and pivot to scan safer heading.",
            expectedThrottleSign: 0,
            expectedSteeringSign: steeringSign
        };
    }
    return {
        code: "HOLD_AND_REASSESS",
        summary: "Hold briefly to stabilize and request next decision.",
        expectedThrottleSign: 0,
        expectedSteeringSign: 0
    };
}

function normalizeActionReason(rawReason, fallbackReason = {}, step = {}) {
    const fallback = inferFallbackReasonShape(step, fallbackReason);
    const sourceObj = (typeof rawReason === "string")
        ? { summary: rawReason }
        : (rawReason && typeof rawReason === "object")
            ? rawReason
            : {};
    const hasModelCode = typeof sourceObj.code === "string" && sourceObj.code.trim().length > 0;
    const hasModelSummary = typeof sourceObj.summary === "string" && sourceObj.summary.trim().length > 0;

    const code = normalizeReasonCode(
        sourceObj.code || sourceObj.reasonCode || sourceObj.label || fallback.code,
        fallback.code
    );
    const summaryRaw = sourceObj.summary || sourceObj.reason || sourceObj.text || sourceObj.rationale || fallback.summary;
    const summary = (typeof summaryRaw === "string" && summaryRaw.trim().length > 0
        ? summaryRaw.trim().replace(/\s+/g, " ").slice(0, 180)
        : fallback.summary);
    const expectedThrottleSign = normalizeReasonSign(
        sourceObj.expectedThrottleSign ?? sourceObj.throttleSign ?? sourceObj.expected?.throttleSign,
        fallback.expectedThrottleSign
    );
    const expectedSteeringSign = normalizeReasonSign(
        sourceObj.expectedSteeringSign ?? sourceObj.steeringSign ?? sourceObj.expected?.steeringSign,
        fallback.expectedSteeringSign
    );
    const sourceRaw = typeof sourceObj.source === "string" ? sourceObj.source.trim().toLowerCase() : "";
    const source = sourceRaw || ((hasModelCode || hasModelSummary) ? "model" : "fallback");

    return {
        code,
        summary,
        expectedThrottleSign,
        expectedSteeringSign,
        source
    };
}

function buildDefaultDecisionReason({ strategyMode, skillName, strategySector, throttle, steering }) {
    const mode = normalizeStrategyMode(strategyMode, "MEMORY_EXPLORE");
    const skill = normalizeSkillName(skillName, "MOVE_TO_FRONTIER");
    const sector = normalizeSector(strategySector, "F");
    const code = normalizeReasonCode(`${mode}_${skill}`, "STRATEGY_EXECUTION");
    const summary = `${mode} with ${skill} toward sector ${sector}.`;
    const normalized = normalizeActionReason(
        { code, summary },
        {},
        { throttle, steering }
    );
    return {
        ...normalized,
        source: "fallback"
    };
}

function normalizeActionStep(raw, fallback = {}) {
    const throttle = clampNumber(raw?.throttle, -1.0, 1.0, asNum(fallback?.throttle, 0));
    const steering = clampNumber(raw?.steering, -1.0, 1.0, asNum(fallback?.steering, 0));
    const duration = clampNumber(raw?.duration, 0.1, 3.0, asNum(fallback?.duration, 0.3));
    const reasonRaw = raw?.reason ?? raw?.why ?? raw?.rationale ?? {
        code: raw?.reasonCode,
        summary: raw?.reasonText,
        expectedThrottleSign: raw?.expectedThrottleSign,
        expectedSteeringSign: raw?.expectedSteeringSign
    };
    const reason = normalizeActionReason(
        reasonRaw,
        fallback?.reason || {},
        { throttle, steering, duration }
    );
    return { throttle, steering, duration, reason };
}

function normalizeActionPlan(rawActions, fallbackStep) {
    if (!Array.isArray(rawActions) || rawActions.length === 0) return [];
    const plan = [];
    let totalDuration = 0;

    for (let i = 0; i < rawActions.length && plan.length < ACTION_PLAN_MAX_STEPS; i += 1) {
        const normalized = normalizeActionStep(rawActions[i], fallbackStep);
        const remaining = ACTION_PLAN_MAX_TOTAL_SEC - totalDuration;
        if (remaining < 0.1) break;
        const clampedDuration = Math.max(0.1, Math.min(normalized.duration, remaining));
        plan.push({
            throttle: normalized.throttle,
            steering: normalized.steering,
            duration: Number(clampedDuration.toFixed(3)),
            reason: normalized.reason
        });
        totalDuration += clampedDuration;
    }

    return plan;
}

function buildObstacleSummary(sensorData) {
    const entries = [
        ["L", sensorData?.left],
        ["LD", sensorData?.leftDiag],
        ["F", sensorData?.front],
        ["RD", sensorData?.rightDiag],
        ["R", sensorData?.right],
        ["BL", sensorData?.backLeft],
        ["B", sensorData?.back],
        ["BR", sensorData?.backRight]
    ];

    const seen = entries
        .map(([name, v]) => [name, asNum(v, 10)])
        .filter(([, v]) => v < 10)
        .map(([name, v]) => `${name}:${v.toFixed(1)}m`);

    if (seen.length === 0) return "Obstacle: None.";
    return `Obstacle: ${seen.join(" ")}.`;
}

function buildTargetSignalProfile(sensorData) {
    const hits = sensorData?.targetHits || {};
    const targetHitCount = Object.values(hits).filter(Boolean).length;
    const angleAbs = Math.abs(sensorData?.angleToTarget ?? 180);
    const distance = sensorData?.distanceToTarget ?? 99;

    const hasContact = targetHitCount > 0;
    const nearTarget = distance < 14;

    let mode = "WEAK";
    if (hasContact || (nearTarget && angleAbs <= 30)) {
        mode = "STRONG";
    } else if (nearTarget || angleAbs <= 70) {
        mode = "MEDIUM";
    }

    const explorationWeight = mode === "STRONG" ? 0.15 : mode === "MEDIUM" ? 0.45 : 0.85;
    const targetWeight = 1.0 - explorationWeight;

    return {
        mode,
        targetWeight,
        explorationWeight,
        targetHitCount,
        angleAbs,
        distance
    };
}

function buildCompactExplorationContext(explorationContext) {
    if (!explorationContext) return null;

    const diagnostics = explorationContext.diagnostics || {};

    return {
        loopWarning: explorationContext.loopWarning || "LOW",
        loopRate: explorationContext.loopRate ?? 0,
        preferredSector: explorationContext.preferredSector || "F",
        currentCell: explorationContext.currentCell || null,
        memoryStats: explorationContext.memoryStats || null,
        frontier: (explorationContext.frontier || []).slice(0, 3),
        risky: (explorationContext.risky || []).slice(0, 2),
        diagnostics: {
            candidateCount: diagnostics.candidateCount ?? 0,
            noGoRatio: diagnostics.noGoRatio ?? 0,
            revisitRate: diagnostics.revisitRate ?? explorationContext.loopRate ?? 0,
            targetColdCount: diagnostics.targetColdCount ?? 0,
            targetColdRatio: diagnostics.targetColdRatio ?? 0,
            topCandidates: (diagnostics.topCandidates || []).slice(0, 3),
            topSafeCandidates: (diagnostics.topSafeCandidates || []).slice(0, 3),
            preferredCandidate: diagnostics.preferredCandidate || null,
            sectorSafety: diagnostics.sectorSafety || []
        }
    };
}

function normalizeStrategyMode(mode, fallback = "MEMORY_EXPLORE") {
    const raw = typeof mode === "string" ? mode.trim().toUpperCase() : "";
    if (raw === "TARGET_LOCK" || raw === "MEMORY_EXPLORE" || raw === "ESCAPE_RECOVERY") {
        return raw;
    }
    return fallback;
}

function mapSignalToMode(targetSignalMode) {
    if (targetSignalMode === "STRONG") return "TARGET_LOCK";
    if (targetSignalMode === "MEDIUM") return "MEMORY_EXPLORE";
    return "MEMORY_EXPLORE";
}

function buildStrategicSnapshot(compactExplorationContext, targetSignal) {
    const frontier = compactExplorationContext?.frontier || [];
    const risky = compactExplorationContext?.risky || [];

    return {
        modeHintFromSignal: mapSignalToMode(targetSignal.mode),
        preferredSector: compactExplorationContext?.preferredSector || "F",
        loopWarning: compactExplorationContext?.loopWarning || "LOW",
        loopRate: compactExplorationContext?.loopRate ?? 0,
        bestFrontier: frontier[0] || null,
        secondFrontier: frontier[1] || null,
        highRiskCells: risky
    };
}

function candidateToPromptToken(candidate) {
    if (!candidate) return "none";
    const sector = normalizeSector(candidate.sector, "F");
    const score = asNum(candidate.score, 0).toFixed(2);
    const risk = asNum(candidate.risk, 0).toFixed(2);
    const visits = Math.max(0, Math.round(asNum(candidate.visits, 0)));
    const targetHits = Math.max(0, Math.round(asNum(candidate.targetHitCount, 0)));
    const targetMisses = Math.max(0, Math.round(asNum(candidate.targetMissCount, 0)));
    const targetAbsence = asNum(candidate.targetAbsence, asNum(candidate.targetAbsenceEMA, 0)).toFixed(2);
    return `${sector}(${candidate.ix},${candidate.iz}) s=${score} r=${risk} v=${visits} th=${targetHits} tm=${targetMisses} ta=${targetAbsence}`;
}

function buildExplorationPromptDigest(compactExplorationContext) {
    if (!compactExplorationContext) return "none";
    const diagnostics = compactExplorationContext.diagnostics || {};
    const currentCell = compactExplorationContext.currentCell || {};
    const memoryStats = compactExplorationContext.memoryStats || {};
    const top = (diagnostics.topCandidates || []).slice(0, 2).map(candidateToPromptToken).join(" | ");
    const safe = (diagnostics.topSafeCandidates || []).slice(0, 2).map(candidateToPromptToken).join(" | ");
    const sectorSafety = (diagnostics.sectorSafety || [])
        .map((s) => `${normalizeSector(s?.sector)}:safe${Math.round(asNum(s?.safeCount, 0))}/tot${Math.round(asNum(s?.totalCount, 0))}/ng${asNum(s?.noGoRatio, 0).toFixed(2)}`)
        .join(" ; ");
    return [
        `loop=${asNum(compactExplorationContext.loopRate, 0).toFixed(3)}`,
        `warn=${compactExplorationContext.loopWarning || "LOW"}`,
        `pref=${normalizeSector(compactExplorationContext.preferredSector, "F")}`,
        `cell=(${Math.round(asNum(currentCell.ix, 0))},${Math.round(asNum(currentCell.iz, 0))}) risk=${asNum(currentCell.risk, 0).toFixed(2)} visits=${Math.round(asNum(currentCell.visits, 0))} th=${Math.round(asNum(currentCell.targetHitCount, 0))} tm=${Math.round(asNum(currentCell.targetMissCount, 0))} ta=${asNum(currentCell.targetAbsence, 0).toFixed(2)}`,
        `mapped=${Math.round(asNum(memoryStats.mappedCells, 0))}`,
        `noGo=${asNum(diagnostics.noGoRatio, 0).toFixed(2)}`,
        `targetCold=${Math.round(asNum(diagnostics.targetColdCount, 0))}/${Math.round(asNum(diagnostics.candidateCount, 0))}(${asNum(diagnostics.targetColdRatio, 0).toFixed(2)})`,
        `top=${top || "none"}`,
        `safe=${safe || "none"}`,
        `sectorSafety=${sectorSafety || "none"}`
    ].join(" | ");
}

function buildStrategicSnapshotDigest(strategicSnapshot) {
    if (!strategicSnapshot) return "none";
    return [
        `hint=${strategicSnapshot.modeHintFromSignal || "MEMORY_EXPLORE"}`,
        `pref=${normalizeSector(strategicSnapshot.preferredSector, "F")}`,
        `loop=${strategicSnapshot.loopWarning || "LOW"}`,
        `best=${candidateToPromptToken(strategicSnapshot.bestFrontier)}`,
        `second=${candidateToPromptToken(strategicSnapshot.secondFrontier)}`
    ].join(" | ");
}

function buildCollisionPressureDigest(runtimeDiagnostics) {
    const summary = runtimeDiagnostics?.collisionSummary && typeof runtimeDiagnostics.collisionSummary === "object"
        ? runtimeDiagnostics.collisionSummary
        : {};
    const total = Math.max(0, Math.round(asNum(summary.totalCount, 0)));
    const repeat = Math.max(0, Math.round(asNum(summary.sameWallRepeatCount, 0)));
    const consecutive = Math.max(0, Math.round(asNum(summary.sameWallConsecutiveRepeatCount, 0)));
    const lastRegion = typeof summary.lastRegion === "string" ? summary.lastRegion : "NONE";
    if (total === 0) return "none";
    return `total=${total} repeat=${repeat} consecutive=${consecutive} last=${lastRegion}`;
}

function buildActionDescription(throttle, steering) {
    let actionDescription = "IDLE";
    if (throttle < -0.1) actionDescription = "REVERSE";
    else if (throttle > 0.1) actionDescription = "FORWARD";

    if (steering < -0.3) actionDescription += "_RIGHT";
    else if (steering > 0.3) actionDescription += "_LEFT";

    return actionDescription;
}

function normalizeSector(sector, fallback = "F") {
    const s = typeof sector === "string" ? sector.trim().toUpperCase() : "";
    return ["L", "F", "R", "B"].includes(s) ? s : fallback;
}

function normalizeSkillName(skill, fallback = "MOVE_TO_FRONTIER") {
    const raw = typeof skill === "string" ? skill.trim().toUpperCase() : "";
    return VALID_SKILLS.includes(raw) ? raw : fallback;
}

function getSectorClearance(sensorData) {
    return {
        L: Math.min(asNum(sensorData?.left, 99), asNum(sensorData?.leftDiag, 99)),
        F: Math.min(asNum(sensorData?.front, 99), asNum(sensorData?.leftDiag, 99), asNum(sensorData?.rightDiag, 99)),
        R: Math.min(asNum(sensorData?.right, 99), asNum(sensorData?.rightDiag, 99)),
        B: Math.min(asNum(sensorData?.back, 99), asNum(sensorData?.backLeft, 99), asNum(sensorData?.backRight, 99)),
    };
}

function sectorFromSteering(steering) {
    if (steering > 0.25) return "L";
    if (steering < -0.25) return "R";
    return "F";
}

function steeringForSector(sector, fallbackSteering = 0) {
    const s = normalizeSector(sector);
    if (s === "L") return 0.62;
    if (s === "R") return -0.62;
    if (s === "F") return 0;
    return fallbackSteering;
}

function worldRegionToSector(region, headingDeg = 0) {
    const key = typeof region === "string" ? region.trim().toUpperCase() : "";
    const vectorMap = {
        OUTER_NORTH: { x: 0, z: -1 },
        OUTER_SOUTH: { x: 0, z: 1 },
        OUTER_EAST: { x: 1, z: 0 },
        OUTER_WEST: { x: -1, z: 0 }
    };
    const targetVec = vectorMap[key];
    if (!targetVec) return null;

    const headingRad = (asNum(headingDeg, 0) * Math.PI) / 180;
    const fx = Math.sin(headingRad);
    const fz = Math.cos(headingRad);

    const dot = clampNumber((fx * targetVec.x) + (fz * targetVec.z), -1, 1, 0);
    const crossY = (fx * targetVec.z) - (fz * targetVec.x);
    const angle = Math.atan2(crossY, dot) * (180 / Math.PI);
    const abs = Math.abs(angle);

    if (abs <= 35) return "F";
    if (abs >= 145) return "B";
    return angle > 0 ? "R" : "L";
}

function countTargetHits(sensorData) {
    return Object.values(sensorData?.targetHits || {}).filter(Boolean).length;
}

function buildOutcomeSignalForPrompt(state) {
    const summary = (typeof state?.lastOutcomeSummary === "string" && state.lastOutcomeSummary.trim().length > 0)
        ? state.lastOutcomeSummary.trim()
        : "No previous outcome yet.";
    const details = (state?.lastOutcomeDetails && typeof state.lastOutcomeDetails === "object") ? state.lastOutcomeDetails : null;
    return {
        label: details?.label || "NONE",
        summary,
        progressDeltaM: asNum(details?.progressDeltaM, 0),
        minObstacleDeltaM: asNum(details?.minObstacleDeltaM, 0),
        targetHitDelta: asNum(details?.targetHitDelta, 0),
        loopRateDelta: asNum(details?.loopRateDelta, 0),
        safetyOverride: !!details?.safetyOverride
    };
}

function applySkillExecutor({
    skillName,
    strategySector,
    throttle,
    steering,
    duration,
    contextModeSignal,
    sensorData,
    skillIntensity = 0.5
}) {
    let nextThrottle = throttle;
    let nextSteering = steering;
    let nextDuration = duration;
    let executorNote = "";

    const frontDist = Math.min(
        asNum(sensorData?.front, 99),
        asNum(sensorData?.leftDiag, 99),
        asNum(sensorData?.rightDiag, 99)
    );
    const leftClear = Math.min(asNum(sensorData?.left, 99), asNum(sensorData?.leftDiag, 99));
    const rightClear = Math.min(asNum(sensorData?.right, 99), asNum(sensorData?.rightDiag, 99));
    const skillSector = normalizeSector(
        strategySector,
        Math.abs(steering) > 0.2 ? sectorFromSteering(steering) : "F"
    );
    const sectorSteer = steeringForSector(skillSector, steering);
    const intensity = clampNumber(skillIntensity, 0, 1, 0.5);

    if (skillName === "HOLD_POSITION") {
        nextThrottle = 0;
        nextSteering = 0;
        nextDuration = Math.max(nextDuration, 0.25);
        executorNote = "hold_position";
    } else if (skillName === "BACKOFF_AND_TURN") {
        nextThrottle = Math.min(nextThrottle, -(0.28 + (0.22 * intensity)));
        if (Math.abs(nextSteering) < 0.45) {
            nextSteering = leftClear >= rightClear ? 0.72 : -0.72;
        }
        nextDuration = Math.max(nextDuration, 0.45);
        executorNote = "backoff_turn_profile";
    } else if (skillName === "SCAN_SECTOR") {
        nextThrottle = Math.max(0.1, Math.min(nextThrottle, 0.2));
        if (skillSector === "F") {
            const sweepDir = contextModeSignal?.reacquireTurnDir > 0 ? 1 : -1;
            nextSteering = 0.45 * sweepDir;
        } else {
            nextSteering = steeringForSector(skillSector, nextSteering) * 0.72;
        }
        nextDuration = Math.max(nextDuration, 0.35);
        executorNote = "scan_sector_profile";
    } else if (skillName === "APPROACH_TARGET") {
        if (frontDist > 2.7) {
            nextThrottle = Math.max(nextThrottle, 0.24 + (0.28 * intensity));
        } else {
            nextThrottle = Math.min(nextThrottle, 0.18);
        }
        if (Math.abs(nextSteering) < 0.22) {
            nextSteering = skillSector === "F" ? nextSteering : (sectorSteer * 0.8);
        }
        nextDuration = Math.max(nextDuration, 0.3);
        executorNote = "approach_target_profile";
    } else {
        // MOVE_TO_FRONTIER (default)
        nextThrottle = Math.max(nextThrottle, 0.22 + (0.2 * intensity));
        if (Math.abs(nextSteering) < 0.2) {
            nextSteering = sectorSteer * 0.7;
        }
        nextDuration = Math.max(nextDuration, 0.35);
        executorNote = "move_frontier_profile";
    }

    return {
        throttle: clampNumber(nextThrottle, -1, 1, throttle),
        steering: clampNumber(nextSteering, -1, 1, steering),
        duration: clampNumber(nextDuration, 0.1, 3.0, duration),
        executorNote
    };
}

function resolveSkillWithContext({
    requestedSkill,
    strategyMode,
    strategySector,
    contextModeSignal,
    sensorData
}) {
    const minFrontDist = Math.min(
        asNum(sensorData?.front, 99),
        asNum(sensorData?.leftDiag, 99),
        asNum(sensorData?.rightDiag, 99)
    );
    const safeForward = minFrontDist > 2.55;
    const approachBias = (
        !contextModeSignal?.danger
        && safeForward
        && (
            contextModeSignal?.hasTargetContact
            || (
                asNum(contextModeSignal?.pseudoContactScore, 0) >= 0.56
                && asNum(contextModeSignal?.angleAbs, 180) <= 62
                && asNum(contextModeSignal?.distance, 99) <= 26
            )
        )
    );
    const strictScanCondition = (
        !!contextModeSignal?.reacquireActive
        && !!contextModeSignal?.weakTargetCue
        && !contextModeSignal?.hasTargetContact
        && asNum(contextModeSignal?.pseudoContactScore, 1) < 0.46
        && asNum(contextModeSignal?.distance, 99) > 18
        && asNum(contextModeSignal?.angleAbs, 0) > 18
    );

    let nextSkill = normalizeSkillName(requestedSkill, "MOVE_TO_FRONTIER");
    let overrideReason = "";

    if (strategyMode === "ESCAPE_RECOVERY") {
        if (nextSkill !== "BACKOFF_AND_TURN") {
            nextSkill = "BACKOFF_AND_TURN";
            overrideReason = "escape_forces_backoff";
        }
    } else if (strategyMode === "TARGET_LOCK") {
        if (!safeForward && nextSkill !== "BACKOFF_AND_TURN") {
            nextSkill = "BACKOFF_AND_TURN";
            overrideReason = "target_lock_front_blocked_escape";
        } else if (safeForward && nextSkill !== "APPROACH_TARGET") {
            nextSkill = "APPROACH_TARGET";
            overrideReason = "target_lock_forces_approach";
        }
    } else if (nextSkill === "SCAN_SECTOR" && !strictScanCondition) {
        if (approachBias) {
            nextSkill = "APPROACH_TARGET";
            overrideReason = "scan_demoted_to_approach";
        } else {
            nextSkill = "MOVE_TO_FRONTIER";
            overrideReason = "scan_demoted_to_frontier";
        }
    } else if (nextSkill === "SCAN_SECTOR" && strictScanCondition && safeForward && asNum(contextModeSignal?.pseudoContactScore, 0) >= 0.42) {
        nextSkill = "MOVE_TO_FRONTIER";
        overrideReason = "scan_demoted_due_to_mid_target_cue";
    } else if ((nextSkill === "MOVE_TO_FRONTIER" || nextSkill === "SCAN_SECTOR") && approachBias) {
        nextSkill = "APPROACH_TARGET";
        overrideReason = "approach_bias_from_target_cue";
    }

    if (nextSkill === "APPROACH_TARGET" && normalizeSector(strategySector) === "B" && !contextModeSignal?.hasTargetContact) {
        nextSkill = "MOVE_TO_FRONTIER";
        overrideReason = "avoid_backward_approach";
    }

    return {
        skill: nextSkill,
        overrideReason,
        approachBias,
        strictScanCondition
    };
}

function enforceSkillCadence({
    requestedSkill,
    strategyMode,
    contextModeSignal,
    sensorData,
    approachBias,
    smoothingState
}) {
    ensureSmoothingState(smoothingState);
    const minFrontDist = Math.min(
        asNum(sensorData?.front, 99),
        asNum(sensorData?.leftDiag, 99),
        asNum(sensorData?.rightDiag, 99)
    );
    const previousSkill = normalizeSkillName(smoothingState?.lastSkillName, "MOVE_TO_FRONTIER");
    let nextSkill = normalizeSkillName(requestedSkill, "MOVE_TO_FRONTIER");
    let overrideReason = "";

    const forwardApproachReady = (
        !contextModeSignal?.danger
        && minFrontDist > 2.65
        && (
            contextModeSignal?.hasTargetContact
            || (
                asNum(contextModeSignal?.pseudoContactScore, 0) >= 0.52
                && asNum(contextModeSignal?.distance, 99) <= 24
                && asNum(contextModeSignal?.angleAbs, 180) <= 65
            )
        )
    );

    if (
        nextSkill === "SCAN_SECTOR"
        && smoothingState.scanBurstCount >= SKILL_BURST_SCAN_LIMIT
        && !contextModeSignal?.danger
    ) {
        nextSkill = forwardApproachReady ? "APPROACH_TARGET" : "MOVE_TO_FRONTIER";
        overrideReason = "scan_burst_limited";
    } else if (
        nextSkill === "BACKOFF_AND_TURN"
        && smoothingState.backoffBurstCount >= SKILL_BURST_BACKOFF_LIMIT
        && strategyMode !== "ESCAPE_RECOVERY"
        && !contextModeSignal?.danger
        && minFrontDist > SKILL_BURST_RELEASE_FRONT_DIST
    ) {
        nextSkill = (forwardApproachReady || approachBias) ? "APPROACH_TARGET" : "MOVE_TO_FRONTIER";
        overrideReason = "backoff_burst_limited";
    }

    if (
        nextSkill !== "APPROACH_TARGET"
        && smoothingState.approachHoldRemaining > 0
        && strategyMode !== "ESCAPE_RECOVERY"
        && forwardApproachReady
    ) {
        nextSkill = "APPROACH_TARGET";
        smoothingState.approachHoldRemaining = Math.max(0, smoothingState.approachHoldRemaining - 1);
        overrideReason = overrideReason
            ? `${overrideReason}|approach_hold_lock`
            : "approach_hold_lock";
    }

    if (nextSkill === "APPROACH_TARGET") {
        if (previousSkill !== "APPROACH_TARGET") {
            smoothingState.approachHoldRemaining = Math.max(
                smoothingState.approachHoldRemaining,
                APPROACH_MIN_HOLD_CYCLES - 1
            );
        } else if (smoothingState.approachHoldRemaining > 0) {
            smoothingState.approachHoldRemaining -= 1;
        }
    } else if (
        !forwardApproachReady
        || strategyMode === "ESCAPE_RECOVERY"
        || contextModeSignal?.danger
    ) {
        smoothingState.approachHoldRemaining = 0;
    }

    smoothingState.scanBurstCount = nextSkill === "SCAN_SECTOR" ? (smoothingState.scanBurstCount + 1) : 0;
    smoothingState.backoffBurstCount = nextSkill === "BACKOFF_AND_TURN" ? (smoothingState.backoffBurstCount + 1) : 0;

    return {
        skill: nextSkill,
        overrideReason,
        approachHoldRemaining: smoothingState.approachHoldRemaining,
        scanBurstCount: smoothingState.scanBurstCount,
        backoffBurstCount: smoothingState.backoffBurstCount
    };
}

function applyMemorySafetyGuard({
    sensorData,
    explorationContext,
    strategySector,
    runtimeDiagnostics,
    throttle,
    steering,
    thought
}) {
    let nextThrottle = throttle;
    let nextSteering = steering;
    let nextThought = thought;
    let guardReason = "";
    let guardApplied = false;

    const clearance = getSectorClearance(sensorData);
    const sensorBlocked = new Set();
    if (clearance.F < 2.9) sensorBlocked.add("F");
    if (clearance.L < 2.25) sensorBlocked.add("L");
    if (clearance.R < 2.25) sensorBlocked.add("R");
    if (clearance.B < 2.3) sensorBlocked.add("B");
    const frontNearDanger = clearance.F < 2.6;

    const sectorSafety = explorationContext?.diagnostics?.sectorSafety || [];
    const memoryBlocked = new Set(
        sectorSafety
            .filter((s) => {
                const safeCount = asNum(s?.safeCount, 0);
                const noGoRatio = asNum(s?.noGoRatio, 0);
                return (safeCount <= 2 && noGoRatio >= 0.5) || noGoRatio >= 0.75;
            })
            .map((s) => normalizeSector(s?.sector))
    );
    const collisionSummary = runtimeDiagnostics?.collisionSummary && typeof runtimeDiagnostics.collisionSummary === "object"
        ? runtimeDiagnostics.collisionSummary
        : null;
    const repeatedWallCount = Math.max(0, Math.round(asNum(collisionSummary?.sameWallRepeatCount, 0)));
    const repeatedConsecutive = Math.max(0, Math.round(asNum(collisionSummary?.sameWallConsecutiveRepeatCount, 0)));
    const repeatedWallSector = worldRegionToSector(collisionSummary?.lastRegion, sensorData?.headingDeg);
    if (repeatedWallSector && (repeatedConsecutive >= 2 || repeatedWallCount >= 4)) {
        memoryBlocked.add(repeatedWallSector);
    }

    const intendedSector = Math.abs(steering) > 0.2
        ? sectorFromSteering(steering)
        : normalizeSector(strategySector);
    const shouldGuardForward = nextThrottle > 0.04;
    const frontThrottleRisk = intendedSector === "F" && frontNearDanger && nextThrottle > 0.18;
    const forbiddenForward = sensorBlocked.has(intendedSector) || memoryBlocked.has(intendedSector) || frontThrottleRisk;
    const targetHitCount = countTargetHits(sensorData);
    const closeTargetApproach = (
        targetHitCount > 0
        || (asNum(sensorData?.distanceToTarget, 99) < 3.2 && Math.abs(asNum(sensorData?.angleToTarget, 180)) < 18)
    );
    const allowFinalForwardApproach = closeTargetApproach && intendedSector === "F" && clearance.F > 2.2;

    if (shouldGuardForward && forbiddenForward && !allowFinalForwardApproach) {
        const forwardOptions = ["F", "L", "R"]
            .filter((s) => !sensorBlocked.has(s) && !memoryBlocked.has(s))
            .sort((a, b) => clearance[b] - clearance[a]);

        const bestForward = forwardOptions[0] || null;

        if (bestForward) {
            nextSteering = steeringForSector(bestForward, steering);
            const nearClamp = frontNearDanger ? 0.22 : 0.34;
            nextThrottle = Math.min(nextThrottle, bestForward === "F" ? 0.2 : nearClamp);
            guardReason = `forward_guard_${intendedSector}_to_${bestForward}`;
            nextThought = `[SafetyGuard] ${intendedSector} blocked, re-route to ${bestForward}.`;
            guardApplied = true;
        } else if (clearance.B > 2.8) {
            nextThrottle = -0.35;
            nextSteering = clearance.L >= clearance.R ? 0.58 : -0.58;
            guardReason = `forward_guard_${intendedSector}_reverse_escape`;
            nextThought = `[SafetyGuard] Forward sectors blocked, reverse escape.`;
            guardApplied = true;
        } else {
            nextThrottle = 0;
            nextSteering = 0;
            guardReason = `forward_guard_${intendedSector}_hold`;
            nextThought = `[SafetyGuard] No safe sector available, holding.`;
            guardApplied = true;
        }
    }

    // Additional hard clamp: do not push forward strongly when front arc is tight.
    if (nextThrottle > 0.1 && clearance.F < 2.35 && !allowFinalForwardApproach) {
        nextThrottle = 0.1;
        if (Math.abs(nextSteering) < 0.35) {
            nextSteering = clearance.L >= clearance.R ? 0.52 : -0.52;
        }
        guardReason = guardReason ? `${guardReason}|front_tight_clamp` : "front_tight_clamp";
        nextThought = "[SafetyGuard] Front arc tight, slowing and biasing escape turn.";
        guardApplied = true;
    }

    const reverseRequested = nextThrottle < -0.08;
    const rearTightForReverse = clearance.B < 3.05;
    const frontOpenForForwardEscape = clearance.F > 3.2;
    const lateralEscapeOpen = Math.max(clearance.L, clearance.R) > 2.45;
    if (reverseRequested && rearTightForReverse && frontOpenForForwardEscape && lateralEscapeOpen) {
        nextThrottle = 0.2;
        if (Math.abs(nextSteering) < 0.45) {
            nextSteering = clearance.L >= clearance.R ? 0.62 : -0.62;
        }
        guardReason = guardReason ? `${guardReason}|rear_tight_prefer_forward` : "rear_tight_prefer_forward";
        nextThought = "[SafetyGuard] Rear is tight while front is open, preferring forward escape.";
        guardApplied = true;
    }

    if (nextThrottle < -0.08 && sensorBlocked.has("B")) {
        nextThrottle = 0;
        guardReason = guardReason ? `${guardReason}|rear_blocked` : "rear_blocked";
        nextThought = "[SafetyGuard] Reverse path blocked, holding.";
        guardApplied = true;
    }

    return {
        throttle: nextThrottle,
        steering: nextSteering,
        thought: nextThought,
        guardApplied,
        guardReason,
        profile: {
            intendedSector,
            clearance,
            frontNearDanger,
            closeTargetApproach,
            allowFinalForwardApproach,
            targetHitCount,
            repeatedWallCount,
            repeatedConsecutive,
            repeatedWallSector,
            sensorBlockedSectors: Array.from(sensorBlocked),
            memoryBlockedSectors: Array.from(memoryBlocked)
        }
    };
}

function stripCodeFences(text) {
    if (typeof text !== "string") return "";
    return text.replace(/```json/gi, "").replace(/```/g, "").trim();
}

function isLikelyTruncatedJsonResponse(rawText) {
    if (typeof rawText !== "string") return true;
    const cleaned = stripCodeFences(rawText).trim();
    if (!cleaned) return true;
    if (!cleaned.endsWith("}")) return true;
    const openCount = (cleaned.match(/{/g) || []).length;
    const closeCount = (cleaned.match(/}/g) || []).length;
    return closeCount < openCount;
}

function extractFirstBalancedObject(text) {
    if (!text) return null;
    const start = text.indexOf("{");
    if (start < 0) return null;

    let depth = 0;
    let inString = false;
    let escaped = false;

    for (let i = start; i < text.length; i += 1) {
        const ch = text[i];

        if (inString) {
            if (escaped) {
                escaped = false;
            } else if (ch === "\\") {
                escaped = true;
            } else if (ch === "\"") {
                inString = false;
            }
            continue;
        }

        if (ch === "\"") {
            inString = true;
            continue;
        }
        if (ch === "{") depth += 1;
        if (ch === "}") {
            depth -= 1;
            if (depth === 0) return text.slice(start, i + 1);
        }
    }

    return null;
}

function tryParseJson(text) {
    try {
        return JSON.parse(text);
    } catch {
        return null;
    }
}

function tryParseJsonWithRepairs(text) {
    if (!text) return null;

    // 1) strict parse
    let parsed = tryParseJson(text);
    if (parsed) return { parsed, recovered: false, method: "strict" };

    // 2) remove trailing commas
    const noTrailingCommas = text.replace(/,\s*([}\]])/g, "$1");
    parsed = tryParseJson(noTrailingCommas);
    if (parsed) return { parsed, recovered: true, method: "trim_trailing_commas" };

    // 3) normalize single quoted keys/values (best-effort only)
    const singleQuoteRepaired = noTrailingCommas
        .replace(/([{,]\s*)'([^']+?)'\s*:/g, "$1\"$2\":")
        .replace(/:\s*'([^']*?)'(?=\s*[,}])/g, ": \"$1\"");
    parsed = tryParseJson(singleQuoteRepaired);
    if (parsed) return { parsed, recovered: true, method: "repair_single_quotes" };

    return null;
}

function recoverLoosePayload(text) {
    if (!text) return null;

    const numberFor = (label) => {
        const re = new RegExp(`${label}["'\\s:]+(-?\\d+(?:\\.\\d+)?)`, "i");
        const m = text.match(re);
        return m ? Number(m[1]) : null;
    };

    const stringFor = (label) => {
        const re = new RegExp(`${label}["'\\s:]+["']([^"'\\n\\r]+)["']`, "i");
        const m = text.match(re);
        return m ? m[1] : null;
    };

    const modeMatch = text.match(/TARGET_LOCK|MEMORY_EXPLORE|ESCAPE_RECOVERY/i);
    const transitionMatch = text.match(/\bSWITCH\b|\bHOLD\b/i);
    const sectorMatch = text.match(/chosenSector["'\s:]+([LFRB])/i) || text.match(/\bsector["'\s:]+([LFRB])/i);
    const skillMatch = text.match(/APPROACH_TARGET|MOVE_TO_FRONTIER|SCAN_SECTOR|BACKOFF_AND_TURN|HOLD_POSITION/i);

    const throttle = numberFor("throttle");
    const steering = numberFor("steering");
    const duration = numberFor("duration");
    const confidence = numberFor("confidence");
    const intensity = numberFor("intensity");
    const thought = stringFor("thought");
    const analysis = stringFor("analysis");
    const adjustment = stringFor("adjustment");

    if (throttle === null && steering === null && duration === null && !modeMatch) return null;

    return {
        strategy: {
            mode: modeMatch ? modeMatch[0].toUpperCase() : "MEMORY_EXPLORE",
            transition: transitionMatch ? transitionMatch[0].toUpperCase() : "HOLD",
            confidence: confidence === null ? 0.35 : confidence,
            chosenSector: sectorMatch ? sectorMatch[1].toUpperCase() : "F",
            targetCue: "",
            memoryCue: "",
            riskCue: "loose_recovery"
        },
        skill: {
            name: skillMatch ? normalizeSkillName(skillMatch[0], "MOVE_TO_FRONTIER") : "MOVE_TO_FRONTIER",
            intensity: intensity === null ? 0.45 : clampNumber(intensity, 0, 1, 0.45),
            rationale: "loose_recovery_default"
        },
        reflection: {
            lastOutcomeAssessment: "unknown",
            adjustment: adjustment || "Use safe exploratory movement and avoid blocked sectors."
        },
        thought: thought || "Recovered from malformed JSON",
        analysis: analysis || "",
        control: {
            throttle: throttle === null ? 0.2 : throttle,
            steering: steering === null ? 0 : steering,
            duration: duration === null ? 0.3 : duration
        }
    };
}

function parseModelResponseJson(rawText) {
    const cleaned = stripCodeFences(rawText);
    const candidates = [cleaned];
    const extracted = extractFirstBalancedObject(cleaned);
    if (extracted && extracted !== cleaned) candidates.push(extracted);

    for (const candidate of candidates) {
        const fixed = tryParseJsonWithRepairs(candidate);
        if (fixed?.parsed) return { data: fixed.parsed, recovered: fixed.recovered, method: fixed.method };
    }

    const loose = recoverLoosePayload(cleaned);
    if (loose) return { data: loose, recovered: true, method: "loose_recovery" };

    return null;
}

function ensureSmoothingState(state) {
    if (!state || typeof state !== "object") return;
    if (!Number.isFinite(state.lastSteering)) state.lastSteering = 0;
    if (typeof state.lastStrategyMode !== "string") state.lastStrategyMode = "";
    if (!Number.isFinite(state.lastModeSwitchAt)) state.lastModeSwitchAt = 0;
    if (typeof state.pendingStrategyMode !== "string") state.pendingStrategyMode = "";
    if (!Number.isFinite(state.pendingStrategySince)) state.pendingStrategySince = 0;
    if (!Number.isFinite(state.pendingStrategyVotes)) state.pendingStrategyVotes = 0;
    if (!Number.isFinite(state.noContactMs)) state.noContactMs = 0;
    if (!Number.isFinite(state.noContactCycles)) state.noContactCycles = 0;
    if (!Number.isFinite(state.lastContactAt)) state.lastContactAt = 0;
    if (!Number.isFinite(state.lastTickAt)) state.lastTickAt = Date.now();
    if (!Number.isFinite(state.reacquireTurnDir) || state.reacquireTurnDir === 0) state.reacquireTurnDir = 1;
    if (!Number.isFinite(state.lastReacquireFlipAt)) state.lastReacquireFlipAt = 0;
    if (typeof state.lastSkillName !== "string") state.lastSkillName = "";
    if (typeof state.lastOutcomeSummary !== "string") state.lastOutcomeSummary = "";
    if (!state.lastOutcomeDetails || typeof state.lastOutcomeDetails !== "object") state.lastOutcomeDetails = null;
    if (typeof state.lastReflectionHint !== "string") state.lastReflectionHint = "";
    if (!Number.isFinite(state.approachHoldRemaining)) state.approachHoldRemaining = 0;
    if (!Number.isFinite(state.scanBurstCount)) state.scanBurstCount = 0;
    if (!Number.isFinite(state.backoffBurstCount)) state.backoffBurstCount = 0;
    if (!Number.isFinite(state.modeHoldRemaining)) state.modeHoldRemaining = 0;
    if (!Number.isFinite(state.targetLockHoldRemaining)) state.targetLockHoldRemaining = 0;
}

function updateContactTracking(state, sensorData) {
    ensureSmoothingState(state);
    const now = Date.now();
    const dtMs = Math.max(0, Math.min(2000, now - state.lastTickAt));
    state.lastTickAt = now;

    const hits = sensorData?.targetHits || {};
    const hasTargetContact = Object.values(hits).some(Boolean);

    if (hasTargetContact) {
        state.noContactMs = 0;
        state.noContactCycles = 0;
        state.lastContactAt = now;
        state.lastReacquireFlipAt = 0;
    } else {
        state.noContactMs += dtMs;
        state.noContactCycles += 1;
    }

    const reacquireActive = !hasTargetContact && state.noContactMs >= 12000;
    if (reacquireActive) {
        if (!state.lastReacquireFlipAt) state.lastReacquireFlipAt = now;
        if (now - state.lastReacquireFlipAt > 3500) {
            state.reacquireTurnDir = state.reacquireTurnDir > 0 ? -1 : 1;
            state.lastReacquireFlipAt = now;
        }
    }

    return {
        hasTargetContact,
        noContactMs: state.noContactMs,
        noContactCycles: state.noContactCycles,
        reacquireActive,
        reacquireTurnDir: state.reacquireTurnDir
    };
}

function inferContextModeSignal(sensorData, contactState = null) {
    const hits = sensorData?.targetHits || {};
    const targetHitCount = Object.values(hits).filter(Boolean).length;
    const hasTargetContact = contactState?.hasTargetContact ?? (targetHitCount > 0);
    const angleAbs = Math.abs(asNum(sensorData?.angleToTarget, 180));
    const distance = asNum(sensorData?.distanceToTarget, 99);
    const noContactMs = asNum(contactState?.noContactMs, 0);
    const noContactCycles = asNum(contactState?.noContactCycles, 0);

    const minFrontDist = Math.min(
        asNum(sensorData?.front, 99),
        asNum(sensorData?.leftDiag, 99),
        asNum(sensorData?.rightDiag, 99)
    );
    const danger = !!sensorData?.isStuck || minFrontDist < CRITICAL_FRONT_DIST;
    const reacquireActive = !!contactState?.reacquireActive;
    const reacquireTurnDir = (contactState?.reacquireTurnDir ?? 1) > 0 ? 1 : -1;

    // Pseudo-contact enables lock-ready behavior even when direct ray hits are sparse.
    const distanceScore = clamp01((22 - distance) / 16);
    const angleScore = clamp01((75 - angleAbs) / 75);
    const frontClearScore = clamp01((minFrontDist - 2.0) / 5.5);
    const hitScore = clamp01(targetHitCount / 2);
    const pseudoContactScore = hasTargetContact
        ? 1
        : clamp01((distanceScore * 0.48) + (angleScore * 0.37) + (frontClearScore * 0.1) + (hitScore * 0.05));
    const reliableDirectionCue = distance < 28 && angleAbs <= 60 && minFrontDist > 2.5;
    const strongTargetCue = hasTargetContact || (!danger && pseudoContactScore >= 0.6 && reliableDirectionCue);
    const weakTargetCue = !hasTargetContact && (pseudoContactScore <= 0.22 || distance > 30 || angleAbs > 95 || (noContactMs >= 18000 && pseudoContactScore < 0.4));
    const lockPriority = (
        !danger
        && (
            hasTargetContact
            || (
                distance <= TARGET_LOCK_PRIORITY_DISTANCE_M
                && angleAbs <= TARGET_LOCK_PRIORITY_ANGLE_DEG
                && pseudoContactScore >= TARGET_LOCK_PRIORITY_PSEUDO
            )
        )
    );
    const lockHoldWindow = (
        !danger
        && minFrontDist > TARGET_LOCK_HOLD_MIN_FRONT_DIST
        && (
            hasTargetContact
            || (
                distance <= TARGET_LOCK_HOLD_DISTANCE_M
                && angleAbs <= TARGET_LOCK_HOLD_ANGLE_DEG
                && pseudoContactScore >= TARGET_LOCK_HOLD_PSEUDO
            )
        )
    );

    const expectedMode = danger
        ? "ESCAPE_RECOVERY"
        : (lockPriority || strongTargetCue)
            ? "TARGET_LOCK"
            : "MEMORY_EXPLORE";

    return {
        expectedMode,
        hasTargetContact,
        strongTargetCue,
        weakTargetCue,
        danger,
        distance,
        angleAbs,
        minFrontDist,
        pseudoContactScore,
        lockPriority,
        lockHoldWindow,
        reliableDirectionCue,
        noContactMs,
        noContactCycles,
        reacquireActive,
        reacquireTurnDir
    };
}

function resolveStrategyModeWithContext(rawMode, signal, previousMode = null) {
    let mode = rawMode;
    let corrected = false;
    let reason = "";
    const previous = normalizeStrategyMode(previousMode, null);

    if (signal.danger && mode !== "ESCAPE_RECOVERY") {
        mode = "ESCAPE_RECOVERY";
        corrected = true;
        reason = "danger_context_forced_escape";
    } else if (mode !== "TARGET_LOCK" && (signal.lockPriority || signal.lockHoldWindow)) {
        mode = "TARGET_LOCK";
        corrected = true;
        reason = signal.lockPriority ? "lock_priority_window_promote" : "lock_hold_window_promote";
    } else if (signal.reacquireActive && mode === "TARGET_LOCK" && !signal.strongTargetCue) {
        mode = "MEMORY_EXPLORE";
        corrected = true;
        reason = "prolonged_no_contact_reacquire";
    } else if (mode === "TARGET_LOCK" && signal.weakTargetCue && !signal.lockPriority && !signal.lockHoldWindow) {
        mode = "MEMORY_EXPLORE";
        corrected = true;
        reason = "weak_target_cue_demote_to_explore";
    } else if (mode !== "TARGET_LOCK" && !signal.danger && signal.strongTargetCue) {
        mode = "TARGET_LOCK";
        corrected = true;
        reason = "strong_target_cue_promote_to_lock";
    } else if (previous === "TARGET_LOCK" && signal.lockHoldWindow && mode !== "TARGET_LOCK") {
        mode = "TARGET_LOCK";
        corrected = true;
        reason = "previous_lock_hold_window";
    }

    return { mode, corrected, reason };
}

function stabilizeStrategyModeWithHysteresis(state, desiredMode, signal) {
    ensureSmoothingState(state);
    const now = Date.now();
    const previousMode = normalizeStrategyMode(state.lastStrategyMode, null);

    const clearPending = () => {
        state.pendingStrategyMode = "";
        state.pendingStrategySince = 0;
        state.pendingStrategyVotes = 0;
    };

    if (!previousMode || desiredMode === previousMode) {
        if (state.modeHoldRemaining > 0) state.modeHoldRemaining -= 1;
        if (previousMode === "TARGET_LOCK" && signal.lockHoldWindow && !signal.danger) {
            state.targetLockHoldRemaining = Math.max(state.targetLockHoldRemaining, TARGET_LOCK_HOLD_MIN_CYCLES);
        } else if (state.targetLockHoldRemaining > 0 && (signal.danger || !signal.lockHoldWindow)) {
            state.targetLockHoldRemaining -= 1;
        }
        clearPending();
        return { mode: desiredMode, held: false, forced: false, reason: "stable" };
    }

    const criticalOverride = signal.danger || signal.minFrontDist < CRITICAL_FRONT_DIST;
    const strongLockPromotion = (
        desiredMode === "TARGET_LOCK"
        && !signal.danger
        && signal.strongTargetCue
        && signal.pseudoContactScore >= 0.7
        && signal.minFrontDist > 2.6
    );
    const lockPriorityPromotion = (
        desiredMode === "TARGET_LOCK"
        && !signal.danger
        && !!signal.lockPriority
        && signal.minFrontDist > 2.3
    );
    const targetLockHoldActive = (
        previousMode === "TARGET_LOCK"
        && !signal.danger
        && signal.minFrontDist > TARGET_LOCK_HOLD_MIN_FRONT_DIST
        && (signal.lockHoldWindow || state.targetLockHoldRemaining > 0)
    );

    if (targetLockHoldActive && desiredMode !== "TARGET_LOCK") {
        clearPending();
        state.targetLockHoldRemaining = signal.lockHoldWindow
            ? TARGET_LOCK_HOLD_MIN_CYCLES
            : Math.max(0, state.targetLockHoldRemaining - 1);
        return { mode: "TARGET_LOCK", held: true, forced: true, reason: "target_lock_hold_window" };
    }

    if (criticalOverride || strongLockPromotion || lockPriorityPromotion) {
        clearPending();
        state.lastModeSwitchAt = now;
        state.modeHoldRemaining = MODE_SWITCH_MIN_HOLD_CYCLES;
        state.targetLockHoldRemaining = desiredMode === "TARGET_LOCK" ? TARGET_LOCK_HOLD_MIN_CYCLES : 0;
        return {
            mode: desiredMode,
            held: false,
            forced: true,
            reason: criticalOverride
                ? "critical_context_override"
                : (lockPriorityPromotion ? "lock_priority_promotion" : "strong_lock_promotion")
        };
    }

    const inCooldown = state.lastModeSwitchAt > 0 && (now - state.lastModeSwitchAt) < MODE_SWITCH_COOLDOWN_MS;
    if (inCooldown) {
        return { mode: previousMode, held: true, forced: false, reason: "switch_cooldown_hold" };
    }
    if (state.modeHoldRemaining > 0) {
        state.modeHoldRemaining -= 1;
        return { mode: previousMode, held: true, forced: false, reason: "minimum_mode_hold" };
    }

    if (state.pendingStrategyMode !== desiredMode) {
        state.pendingStrategyMode = desiredMode;
        state.pendingStrategySince = now;
        state.pendingStrategyVotes = 1;
        return { mode: previousMode, held: true, forced: false, reason: "pending_switch_vote" };
    }

    state.pendingStrategyVotes += 1;
    const pendingMs = now - state.pendingStrategySince;
    const confirmed = state.pendingStrategyVotes >= MODE_SWITCH_MIN_VOTES && pendingMs >= MODE_SWITCH_MIN_DWELL_MS;
    if (confirmed) {
        clearPending();
        state.lastModeSwitchAt = now;
        state.modeHoldRemaining = MODE_SWITCH_MIN_HOLD_CYCLES;
        state.targetLockHoldRemaining = desiredMode === "TARGET_LOCK" ? TARGET_LOCK_HOLD_MIN_CYCLES : 0;
        return { mode: desiredMode, held: false, forced: false, reason: "hysteresis_confirmed_switch" };
    }

    return { mode: previousMode, held: true, forced: false, reason: "pending_switch_wait" };
}

function deriveStrategyTransition(llmTransition, previousMode, currentMode, corrected) {
    const byModeChange = previousMode && previousMode !== currentMode ? "SWITCH" : "HOLD";
    const byLlm = (typeof llmTransition === "string" && llmTransition.toUpperCase() === "SWITCH") ? "SWITCH" : "HOLD";
    if (corrected) return "SWITCH";
    if (byModeChange === "SWITCH") return "SWITCH";
    return byLlm;
}

export async function getAvailableModels() {
    try {
        const response = await fetch(OLLAMA_TAGS);
        if (!response.ok) throw new Error("Failed to fetch models");
        const data = await response.json();
        return data.models.map((m) => m.name);
    } catch (err) {
        console.error("Ollama Model Fetch Error:", err);
        return [];
    }
}

export async function getDrivingDecision(
    sensorData,
    actionHistory = [],
    modelName = "gemma2:9b",
    smoothingState = { lastSteering: 0 },
    explorationContext = null,
    runtimeDiagnostics = null
) {
    ensureSmoothingState(smoothingState);
    const contactState = updateContactTracking(smoothingState, sensorData);
    const historyStr = actionHistory.length > 0 ? actionHistory.join(" -> ") : "None";
    const targetSignal = buildTargetSignalProfile(sensorData);
    const contextModeSignal = inferContextModeSignal(sensorData, contactState);
    const compactExplorationContext = buildCompactExplorationContext(explorationContext);
    const strategicSnapshot = buildStrategicSnapshot(compactExplorationContext, targetSignal);
    const previousOutcomeSignal = buildOutcomeSignalForPrompt(smoothingState);
    const collisionPressureDigest = buildCollisionPressureDigest(runtimeDiagnostics);
    const previousReflectionHint = (typeof smoothingState?.lastReflectionHint === "string" && smoothingState.lastReflectionHint.trim().length > 0)
        ? smoothingState.lastReflectionHint.trim()
        : "None";
    const previousSkill = normalizeSkillName(smoothingState?.lastSkillName, "MOVE_TO_FRONTIER");
    const explorationMemoryDigest = buildExplorationPromptDigest(compactExplorationContext);
    const strategicSnapshotDigest = buildStrategicSnapshotDigest(strategicSnapshot);
    const strategyPriorityDigest = "target_capture > safe_motion > anti_loop";

    const fallbackMode = mapSignalToMode(targetSignal.mode);
    const previousStrategyMode = normalizeStrategyMode(smoothingState?.lastStrategyMode, null) || null;

    console.log(`AI Driver using model: ${modelName}`);

    const prompt = `
You are the PRIMARY autonomous driving intelligence.
Goal: Capture blue targets accurately and repeatedly.
You are responsible for BOTH strategy and control each cycle.

Current target status:
- Distance: ${sensorData.distanceToTarget?.toFixed(1)}m
- Angle: ${sensorData.angleToTarget?.toFixed(0)} deg

Data:
- Sensors (8 rays): L:${sensorData.left?.toFixed(1)} LD:${sensorData.leftDiag?.toFixed(1)} F:${sensorData.front?.toFixed(1)} RD:${sensorData.rightDiag?.toFixed(1)} R:${sensorData.right?.toFixed(1)} BL:${sensorData.backLeft?.toFixed(1)} B:${sensorData.back?.toFixed(1)} BR:${sensorData.backRight?.toFixed(1)}
- Target Hits: ${JSON.stringify(sensorData.targetHits || {})}
- Pose: X:${sensorData.worldX?.toFixed?.(1) ?? sensorData.worldX}, Z:${sensorData.worldZ?.toFixed?.(1) ?? sensorData.worldZ}, Heading:${sensorData.headingDeg?.toFixed?.(0) ?? sensorData.headingDeg}deg
- Coordinate Convention: +X=EAST, -X=WEST, +Z=SOUTH, -Z=NORTH, Heading 0deg=+Z(SOUTH), +90deg=+X(EAST).
- Speed: ${sensorData.speed?.toFixed(1)}
- MoveDir: ${sensorData.moveDir?.toUpperCase()} (PathClear:${sensorData.blockedDist}m)
- Last Actions: ${historyStr}
- Stuck Status: ${sensorData.isStuck ? "STUCK" : "MOVING"}
- Target Signal Mode: ${targetSignal.mode} (ContactCount=${targetSignal.targetHitCount}, AngleAbs=${targetSignal.angleAbs.toFixed(0)}, Dist=${targetSignal.distance.toFixed(1)}m)
- Priority Weights: Target=${targetSignal.targetWeight.toFixed(2)}, ExplorationMemory=${targetSignal.explorationWeight.toFixed(2)}
- Exploration Memory Digest: ${explorationMemoryDigest}
- Strategic Snapshot Digest: ${strategicSnapshotDigest}
- Collision Pressure Digest: ${collisionPressureDigest}
- Strategy Priority: ${strategyPriorityDigest}
- Previous Outcome Signal: ${JSON.stringify(previousOutcomeSignal)}
- Previous Reflection Hint: ${previousReflectionHint}
- Previous Skill: ${previousSkill}
- Previous Strategy Mode: ${previousStrategyMode || "NONE"}
- Expected Mode From Context: ${contextModeSignal.expectedMode}
- LockHoldWindow: ${contextModeSignal.lockHoldWindow}
- PseudoContactScore: ${contextModeSignal.pseudoContactScore.toFixed(2)} (ReliableDirectionCue=${contextModeSignal.reliableDirectionCue})
- NoContactDurationMs: ${contextModeSignal.noContactMs}
- NoContactCycles: ${contextModeSignal.noContactCycles}
- ReacquireActive: ${contextModeSignal.reacquireActive}
- ReacquireTurnHint: ${contextModeSignal.reacquireTurnDir > 0 ? "LEFT" : "RIGHT"}

Rules:
1) Choose one strategy mode: TARGET_LOCK / MEMORY_EXPLORE / ESCAPE_RECOVERY.
2) Decide transition: HOLD or SWITCH. If mode changed from previous mode, transition must be SWITCH.
3) Choose sector: L/F/R/B.
4) Choose one skill: APPROACH_TARGET / MOVE_TO_FRONTIER / SCAN_SECTOR / BACKOFF_AND_TURN / HOLD_POSITION.
5) Use target cues + memory cues together (LLM-led fusion).
6) Avoid no-go memory cells (outsideBounds/barrierBlocked/obstacleDominant).
7) If obstacle in front arc is very close or stuck, prioritize safe escape.
7a) If LockHoldWindow=true and no critical danger, keep TARGET_LOCK.
8) If ReacquireActive=true and no strong target cue, use MEMORY_EXPLORE and perform explicit scan behavior.
9) Use Previous Outcome Signal to avoid repeating failed action.
10) PseudoContactScore>=0.68 with ReliableDirectionCue=true is lock-ready even if direct targetHits are zero.
11) Keep text concise: targetCue/memoryCue/riskCue/rationale/adjustment <= 12 words each.
12) Always provide top-level reason object with code/summary/expectedThrottleSign/expectedSteeringSign.
13) You may optionally provide an actions array (2-5 steps) for smoother trajectory.
14) Each action step must include throttle, steering, duration, and reason object.
15) Do not output markdown. JSON only.

Return JSON with this schema:
{
  "strategy": {
    "mode": "TARGET_LOCK|MEMORY_EXPLORE|ESCAPE_RECOVERY",
    "transition": "HOLD|SWITCH",
    "confidence": 0.0,
    "chosenSector": "L|F|R|B",
    "targetCue": "short text",
    "memoryCue": "short text",
    "riskCue": "short text",
    "rationale": "one sentence"
  },
  "skill": {
    "name": "APPROACH_TARGET|MOVE_TO_FRONTIER|SCAN_SECTOR|BACKOFF_AND_TURN|HOLD_POSITION",
    "intensity": 0.0,
    "rationale": "one sentence"
  },
  "reflection": {
    "lastOutcomeAssessment": "short text",
    "adjustment": "one sentence for next cycle"
  },
  "reason": {
    "code": "UPPER_SNAKE_CASE",
    "summary": "one sentence",
    "expectedThrottleSign": -1,
    "expectedSteeringSign": 1
  },
  "thought": "short driving intent",
  "analysis": "Obstacle summary + target summary",
  "control": {
    "throttle": 0.0,
    "steering": 0.0,
    "duration": 0.2
  },
  "actions": [
    {
      "throttle": 0.2,
      "steering": 0.1,
      "duration": 0.25,
      "reason": {
        "code": "FORWARD_TURN_APPROACH",
        "summary": "turn toward safer target side",
        "expectedThrottleSign": 1,
        "expectedSteeringSign": 1
      }
    },
    {
      "throttle": 0.2,
      "steering": -0.1,
      "duration": 0.25,
      "reason": {
        "code": "MICRO_CORRECTION",
        "summary": "small correction for obstacle clearance",
        "expectedThrottleSign": 1,
        "expectedSteeringSign": -1
      }
    }
  ]
}
`;

    const startTime = performance.now();
    let retryPromptUsed = "";

    const requestModelResponse = async (promptText, numPredict = MODEL_NUM_PREDICT_PRIMARY) => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), OLLAMA_TIMEOUT_MS);
        try {
            const response = await fetch(OLLAMA_GENERATE, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: modelName,
                    prompt: promptText,
                    stream: false,
                    format: "json",
                    options: {
                        temperature: 0.2,
                        num_predict: numPredict
                    }
                }),
                signal: controller.signal
            });
            if (!response.ok) {
                throw new Error(`Ollama API error: ${response.statusText}`);
            }
            const data = await response.json();
            return typeof data?.response === "string" ? data.response : "";
        } finally {
            clearTimeout(timeoutId);
        }
    };

    try {
        let rawResponse = await requestModelResponse(prompt, MODEL_NUM_PREDICT_PRIMARY);
        let parsedResult = parseModelResponseJson(rawResponse);
        const shouldRetryForStrict = (
            !parsedResult?.data
            || parsedResult.method === "loose_recovery"
            || isLikelyTruncatedJsonResponse(rawResponse)
        );

        if (shouldRetryForStrict) {
            const rawTail = stripCodeFences(rawResponse).replace(/\s+/g, " ").slice(-420);
            retryPromptUsed = `
Return ONLY one complete JSON object in the required schema.
Your previous output was incomplete or malformed.
Do not add comments.
Keep text fields short (<= 10 words).
If uncertain, still fill all required fields with safe defaults.
Previous partial tail:
${rawTail}
`;
            try {
                const retryRaw = await requestModelResponse(retryPromptUsed, MODEL_NUM_PREDICT_RETRY);
                const retryParsed = parseModelResponseJson(retryRaw);
                if (retryParsed?.data && !isLikelyTruncatedJsonResponse(retryRaw)) {
                    rawResponse = retryRaw;
                    parsedResult = {
                        ...retryParsed,
                        method: `${retryParsed.method}_retry`
                    };
                }
            } catch (retryErr) {
                console.warn("Retry request for strict JSON failed", retryErr);
            }
        }

        const latency = Math.round(performance.now() - startTime);
        const promptForLog = retryPromptUsed
            ? `${prompt}\n\n[TRUNCATION_RETRY]\n${retryPromptUsed}`
            : prompt;

        try {
            if (!parsedResult?.data) throw new Error("UNPARSEABLE_MODEL_OUTPUT");
            const result = parsedResult.data;
            const control = (result?.control && typeof result.control === "object") ? result.control : result;
            const rootActions = Array.isArray(result?.actions) ? result.actions : [];
            const controlActions = Array.isArray(control?.actions) ? control.actions : [];
            const actionPlanSource = rootActions.length > 0 ? rootActions : controlActions;
            const firstPlannedStep = actionPlanSource.length > 0
                ? normalizeActionStep(actionPlanSource[0], { throttle: 0.2, steering: 0, duration: 0.3 })
                : null;
            const strategy = (result?.strategy && typeof result.strategy === "object") ? result.strategy : {};
            const skill = (result?.skill && typeof result.skill === "object") ? result.skill : {};
            const reflection = (result?.reflection && typeof result.reflection === "object") ? result.reflection : {};
            const reason = (result?.reason && typeof result.reason === "object") ? result.reason : {};

            let strategyMode = normalizeStrategyMode(strategy.mode, fallbackMode);
            let strategyConfidence = clampNumber(strategy.confidence, 0, 1, 0.5);
            const strategySector = (typeof strategy.chosenSector === "string" && ["L", "F", "R", "B"].includes(strategy.chosenSector.toUpperCase()))
                ? strategy.chosenSector.toUpperCase()
                : (strategicSnapshot.preferredSector || "F");
            const inferredSkill = strategyMode === "TARGET_LOCK"
                ? "APPROACH_TARGET"
                : strategyMode === "ESCAPE_RECOVERY"
                    ? "BACKOFF_AND_TURN"
                    : "MOVE_TO_FRONTIER";
            let skillName = normalizeSkillName(skill.name, inferredSkill);
            const skillIntensity = clampNumber(skill.intensity, 0, 1, 0.5);
            const skillRationale = typeof skill.rationale === "string" ? skill.rationale.trim() : "";
            const reflectionOutcome = typeof reflection.lastOutcomeAssessment === "string"
                ? reflection.lastOutcomeAssessment.trim()
                : "";
            const reflectionAdjustment = typeof reflection.adjustment === "string"
                ? reflection.adjustment.trim()
                : "";

            const strategyTargetCue = typeof strategy.targetCue === "string" ? strategy.targetCue : "";
            const strategyMemoryCue = typeof strategy.memoryCue === "string" ? strategy.memoryCue : "";
            const strategyRiskCue = typeof strategy.riskCue === "string" ? strategy.riskCue : "";
            const modeResolved = resolveStrategyModeWithContext(strategyMode, contextModeSignal, previousStrategyMode);
            strategyMode = modeResolved.mode;
            if (modeResolved.corrected) {
                strategyConfidence = Math.min(strategyConfidence, 0.55);
            }
            const hysteresisResolved = stabilizeStrategyModeWithHysteresis(smoothingState, strategyMode, contextModeSignal);
            strategyMode = hysteresisResolved.mode;
            if (hysteresisResolved.held) {
                strategyConfidence = Math.min(strategyConfidence, 0.72);
            }
            const strategyTransition = deriveStrategyTransition(
                strategy.transition,
                previousStrategyMode,
                strategyMode,
                modeResolved.corrected || hysteresisResolved.forced
            );
            let finalThrottle = clampNumber(control?.throttle, -1.0, 1.0, firstPlannedStep ? firstPlannedStep.throttle : 0.4);
            let rawSteering = clampNumber(control?.steering, -1.0, 1.0, firstPlannedStep ? firstPlannedStep.steering : 0.0);
            let finalDuration = clampNumber(control?.duration, 0.1, 3.0, firstPlannedStep ? firstPlannedStep.duration : 1.0);
            let finalThought = result?.thought || strategy.rationale || "Driving";
            const analysis = (typeof result?.analysis === "string" && result.analysis.trim().length > 0)
                ? result.analysis.trim()
                : `${buildObstacleSummary(sensorData)} TargetAngle:${asNum(sensorData?.angleToTarget, 0).toFixed(0)}deg.`;
            const skillResolved = resolveSkillWithContext({
                requestedSkill: skillName,
                strategyMode,
                strategySector,
                contextModeSignal,
                sensorData
            });
            skillName = skillResolved.skill;
            const skillCadenceResolved = enforceSkillCadence({
                requestedSkill: skillName,
                strategyMode,
                contextModeSignal,
                sensorData,
                approachBias: skillResolved.approachBias,
                smoothingState
            });
            skillName = skillCadenceResolved.skill;
            const skillOverrideReasons = [
                skillResolved.overrideReason,
                skillCadenceResolved.overrideReason
            ].filter(Boolean).join("|");
            if (
                (skillOverrideReasons.includes("target_lock_forces_approach")
                    || skillOverrideReasons.includes("approach_hold_lock"))
                && finalDuration < 0.28
            ) {
                finalDuration = 0.28;
            }

            // Steering sign convention is unified with Car physics: +1=LEFT, -1=RIGHT.
            const ALPHA = 1.0;
            let targetSteering = rawSteering;
            let smoothedSteering = smoothingState.lastSteering;

            const minFrontDist = Math.min(
                asNum(sensorData?.front, 99),
                asNum(sensorData?.leftDiag, 99),
                asNum(sensorData?.rightDiag, 99)
            );
            const backClear = asNum(sensorData?.back, 99) > 3.0;

            // Safety-only hard override.
            if (minFrontDist < 2.5 && finalThrottle > 0) {
                const leftEscape = Math.min(asNum(sensorData?.left, 99), asNum(sensorData?.leftDiag, 99));
                const rightEscape = Math.min(asNum(sensorData?.right, 99), asNum(sensorData?.rightDiag, 99));

                if (backClear) {
                    finalThrottle = -0.5;
                    rawSteering = leftEscape >= rightEscape ? 0.6 : -0.6;
                    targetSteering = rawSteering;
                    finalThought = `[Safety] Front blocked ${minFrontDist.toFixed(1)}m, backing off.`;
                } else {
                    finalThrottle = 0;
                    rawSteering = leftEscape >= rightEscape ? 0.8 : -0.8;
                    targetSteering = rawSteering;
                    finalThought = "[Safety] Boxed in front/back, pivoting.";
                }
            }

            // Keep target lock assertive but still LLM-led.
            if (strategyMode === "TARGET_LOCK" && minFrontDist > 3.0 && Math.abs(finalThrottle) < 0.08) {
                finalThrottle = 0.2;
            }

            // Explicit target-reacquire behavior after prolonged 0-contact periods.
            if (contextModeSignal.reacquireActive && strategyMode === "MEMORY_EXPLORE" && !contextModeSignal.danger && !contextModeSignal.hasTargetContact) {
                const sweepDir = contextModeSignal.reacquireTurnDir > 0 ? 1 : -1;
                if (Math.abs(rawSteering) < 0.35) {
                    rawSteering = 0.62 * sweepDir;
                    targetSteering = rawSteering;
                }
                if (finalThrottle < 0.25) finalThrottle = 0.35;
                if (finalDuration < 0.45) finalDuration = 0.45;
                finalThought = `[Reacquire] No target contact for ${(contextModeSignal.noContactMs / 1000).toFixed(1)}s. Sweeping ${sweepDir > 0 ? "LEFT" : "RIGHT"} to reacquire.`;
            }

            const skillExecution = applySkillExecutor({
                skillName,
                strategySector,
                throttle: finalThrottle,
                steering: rawSteering,
                duration: finalDuration,
                contextModeSignal,
                sensorData,
                skillIntensity
            });
            finalThrottle = skillExecution.throttle;
            rawSteering = skillExecution.steering;
            targetSteering = rawSteering;
            finalDuration = skillExecution.duration;

            const guardResult = applyMemorySafetyGuard({
                sensorData,
                explorationContext,
                strategySector,
                runtimeDiagnostics,
                throttle: finalThrottle,
                steering: rawSteering,
                thought: finalThought
            });
            if (guardResult.guardApplied) {
                finalThrottle = guardResult.throttle;
                rawSteering = guardResult.steering;
                targetSteering = rawSteering;
                finalThought = guardResult.thought;
            }

            if (Math.abs(rawSteering) > 0.7 && minFrontDist > 4.0) {
                smoothingState.lastSteering = targetSteering;
            }

            if (finalThrottle < 0 && !backClear) {
                finalThrottle = 0;
                finalThought = "[Safety] Reverse denied, rear blocked.";
            }

            smoothedSteering = (targetSteering * ALPHA) + (smoothingState.lastSteering * (1.0 - ALPHA));
            smoothingState.lastSteering = smoothedSteering;
            smoothingState.lastStrategyMode = strategyMode;
            smoothingState.lastSkillName = skillName;
            if (reflectionAdjustment) smoothingState.lastReflectionHint = reflectionAdjustment;

            let baseRiskCue = modeResolved.corrected
                ? `${strategyRiskCue} | corrected:${modeResolved.reason}`
                : (parsedResult.recovered ? `${strategyRiskCue} | parsed:${parsedResult.method}` : strategyRiskCue);
            if (hysteresisResolved.held) {
                baseRiskCue = `${baseRiskCue} | modeHold:${hysteresisResolved.reason}`;
            } else if (hysteresisResolved.forced) {
                baseRiskCue = `${baseRiskCue} | modeForce:${hysteresisResolved.reason}`;
            }
            const guardCue = guardResult.guardApplied ? ` | guard:${guardResult.guardReason}` : "";
            const blockedSensorCue = guardResult.profile.sensorBlockedSectors.length > 0
                ? ` | sensorBlocked:${guardResult.profile.sensorBlockedSectors.join("/")}`
                : "";
            const blockedMemoryCue = guardResult.profile.memoryBlockedSectors.length > 0
                ? ` | memoryBlocked:${guardResult.profile.memoryBlockedSectors.join("/")}`
                : "";
            const skillOverrideCue = skillOverrideReasons ? ` | skillOverride:${skillOverrideReasons}` : "";
            const cadenceCue = ` | cadence:approachHold=${skillCadenceResolved.approachHoldRemaining},scanBurst=${skillCadenceResolved.scanBurstCount},backoffBurst=${skillCadenceResolved.backoffBurstCount}`;
            const riskCueWithState = `${baseRiskCue} | skill:${skillName} | pseudo:${contextModeSignal.pseudoContactScore.toFixed(2)} | noContactMs:${Math.round(contextModeSignal.noContactMs)} | reacquire:${contextModeSignal.reacquireActive}${skillOverrideCue}${cadenceCue}${guardCue}${blockedSensorCue}${blockedMemoryCue}`;
            const decisionReason = normalizeActionReason(
                reason,
                buildDefaultDecisionReason({
                    strategyMode,
                    skillName,
                    strategySector,
                    throttle: finalThrottle,
                    steering: smoothedSteering
                }),
                { throttle: finalThrottle, steering: smoothedSteering, duration: finalDuration }
            );
            const normalizedActionPlan = normalizeActionPlan(actionPlanSource, {
                throttle: finalThrottle,
                steering: smoothedSteering,
                duration: finalDuration,
                reason: decisionReason
            });
            const actionPlan = normalizedActionPlan.length > 0
                ? normalizedActionPlan.map((step, idx) => (idx === 0
                    ? {
                        throttle: finalThrottle,
                        steering: smoothedSteering,
                        duration: finalDuration,
                        reason: normalizeActionReason(
                            step.reason,
                            decisionReason,
                            { throttle: finalThrottle, steering: smoothedSteering, duration: finalDuration }
                        )
                    }
                    : {
                        ...step,
                        reason: normalizeActionReason(step.reason, decisionReason, step)
                    }))
                : [{
                    throttle: finalThrottle,
                    steering: smoothedSteering,
                    duration: finalDuration,
                    reason: normalizeActionReason(
                        null,
                        decisionReason,
                        { throttle: finalThrottle, steering: smoothedSteering, duration: finalDuration }
                    )
                }];

            return {
                throttle: finalThrottle,
                steering: smoothedSteering,
                duration: finalDuration,
                actionPlan,
                reason: decisionReason,
                action: buildActionDescription(finalThrottle, smoothedSteering),
                thought: `[${strategyMode}/${strategyTransition}] ${finalThought}`,
                analysis,
                strategy: {
                    mode: strategyMode,
                    transition: strategyTransition,
                    confidence: strategyConfidence,
                    chosenSector: strategySector,
                    targetCue: strategyTargetCue,
                    memoryCue: strategyMemoryCue,
                    riskCue: riskCueWithState
                },
                skill: {
                    name: skillName,
                    intensity: skillIntensity,
                    rationale: skillRationale,
                    executorNote: skillExecution.executorNote
                },
                reflection: {
                    lastOutcomeAssessment: reflectionOutcome || "no_assessment",
                    adjustment: reflectionAdjustment || "Keep safe progress while reducing repeated loop segments."
                },
                latency,
                raw: rawResponse,
                prompt: promptForLog,
                model: modelName,
                parseMethod: parsedResult.method,
                parseRecovered: !!parsedResult.recovered,
                safetyGuard: guardResult
            };
        } catch (e) {
            console.warn("JSON Parse Error", e);
            const failMode = contextModeSignal.expectedMode || fallbackMode;
            smoothingState.lastStrategyMode = failMode;
            const fallbackReason = normalizeActionReason(
                { code: "PARSER_ERROR_HOLD", summary: "Parse failed, hold and retry strict JSON." },
                buildDefaultDecisionReason({
                    strategyMode: failMode,
                    skillName: "HOLD_POSITION",
                    strategySector: "F",
                    throttle: 0,
                    steering: 0
                }),
                { throttle: 0, steering: 0, duration: 1.0 }
            );
            return {
                throttle: 0,
                steering: 0,
                duration: 1.0,
                actionPlan: [{ throttle: 0, steering: 0, duration: 1.0, reason: fallbackReason }],
                reason: fallbackReason,
                action: "ERROR",
                thought: "JSON Error",
                analysis: buildObstacleSummary(sensorData),
                strategy: { mode: failMode, transition: deriveStrategyTransition("HOLD", previousStrategyMode, failMode, false), confidence: 0, chosenSector: "F", targetCue: "", memoryCue: "", riskCue: "parser_error_fallback" },
                skill: { name: "HOLD_POSITION", intensity: 1, rationale: "parser_error_fallback", executorNote: "fallback" },
                reflection: { lastOutcomeAssessment: "parse_failed", adjustment: "Return strict JSON next cycle." },
                latency,
                raw: rawResponse,
                prompt: promptForLog,
                model: modelName,
                parseMethod: retryPromptUsed ? "unparseable_model_output_retry" : "unparseable_model_output",
                parseRecovered: false
            };
        }
    } catch (err) {
        console.error("Ollama Service Error:", err);
        const failMode = contextModeSignal.expectedMode || fallbackMode;
        smoothingState.lastStrategyMode = failMode;
        const isTimeout = err?.name === "AbortError";
        const fallbackReason = normalizeActionReason(
            {
                code: isTimeout ? "API_TIMEOUT_HOLD" : "API_ERROR_HOLD",
                summary: isTimeout
                    ? "API timeout, hold and retry next cycle."
                    : "API error, hold and retry next cycle."
            },
            buildDefaultDecisionReason({
                strategyMode: failMode,
                skillName: "HOLD_POSITION",
                strategySector: "F",
                throttle: 0,
                steering: 0
            }),
            { throttle: 0, steering: 0, duration: 1.0 }
        );
        return {
            throttle: 0,
            steering: 0,
            duration: 1.0,
            actionPlan: [{ throttle: 0, steering: 0, duration: 1.0, reason: fallbackReason }],
            reason: fallbackReason,
            action: "ERROR",
            thought: isTimeout ? `API Timeout (${Math.round(OLLAMA_TIMEOUT_MS / 1000)}s)` : "API Error",
            analysis: buildObstacleSummary(sensorData),
            strategy: { mode: failMode, transition: deriveStrategyTransition("HOLD", previousStrategyMode, failMode, false), confidence: 0, chosenSector: "F", targetCue: "", memoryCue: "", riskCue: isTimeout ? "api_timeout_fallback" : "api_error_fallback" },
            skill: { name: "HOLD_POSITION", intensity: 1, rationale: isTimeout ? "api_timeout_fallback" : "api_error_fallback", executorNote: "fallback" },
            reflection: { lastOutcomeAssessment: isTimeout ? "api_timeout" : "api_error", adjustment: "Hold and retry with strict JSON response." },
            latency: 0,
            raw: err.message,
            prompt,
            model: modelName,
            parseMethod: isTimeout ? "api_timeout" : "api_error",
            parseRecovered: false
        };
    }
}
