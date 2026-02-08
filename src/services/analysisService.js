/**
 * Service to analyze driving data and generate reports.
 */

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const asNumber = (value, fallback = null) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);
const percent = (part, whole) => (whole > 0 ? (part / whole) * 100 : 0);

const normalizeReviewHtml = (raw) => {
    if (!raw) return "<p>No review generated.</p>";

    let text = raw.trim();
    text = text.replace(/```html/gi, "").replace(/```/g, "").trim();
    text = text.replace(/<!DOCTYPE[\s\S]*?>/gi, "");
    text = text.replace(/<html[^>]*>/gi, "").replace(/<\/html>/gi, "");
    text = text.replace(/<head[\s\S]*?<\/head>/gi, "");
    text = text.replace(/<body[^>]*>/gi, "").replace(/<\/body>/gi, "");

    return text.trim() || "<p>No review generated.</p>";
};

const inferContactDirection = (pt) => {
    const hits = pt?.targetHits || {};
    const hitLeft = !!hits.left || !!hits.leftDiag;
    const hitRight = !!hits.right || !!hits.rightDiag;
    const hitFront = !!hits.front;

    if (hitLeft && !hitRight) return "LEFT";
    if (hitRight && !hitLeft) return "RIGHT";
    if (hitFront || (hitLeft && hitRight)) return "STRAIGHT";
    return null;
};

const normalizeStrategyMode = (mode) => {
    const raw = typeof mode === "string" ? mode.trim().toUpperCase() : "";
    if (raw === "TARGET_LOCK" || raw === "MEMORY_EXPLORE" || raw === "ESCAPE_RECOVERY") return raw;
    return "UNKNOWN";
};

const downsampleHistory = (history, maxPoints = 1200) => {
    if (!history || history.length <= maxPoints) return history || [];
    const step = Math.ceil(history.length / maxPoints);
    return history.filter((_, i) => i % step === 0 || i === history.length - 1);
};

// Calculate session statistics focused on target capture, safety, and decision quality.
const calculateStats = (history) => {
    if (!Array.isArray(history) || history.length === 0) return null;

    let sumIntent = 0;
    let intentSamples = 0;
    let sumSteerAbs = 0;
    let steerSamples = 0;
    let reverseCount = 0;
    let logicAlignedCount = 0;
    let logicSamples = 0;
    let stuckCount = 0;
    let safetyRiskCount = 0;
    let targetContactCount = 0;
    let contactFollowAlignedCount = 0;
    let contactFollowSamples = 0;
    let sumSpeed = 0;
    let speedSamples = 0;
    let progressTowardCount = 0;
    let progressAwayCount = 0;
    let progressSamples = 0;
    let sumLatency = 0;
    let latencySamples = 0;
    let targetCaptures = 0;
    let strategySamples = 0;
    let strategySwitchCount = 0;
    let strategyConfidenceSum = 0;
    let strategyConfidenceSamples = 0;
    let strategyModeAlignedCount = 0;
    let targetLockContactCount = 0;
    let targetContactSamples = 0;
    let escapeDangerCount = 0;
    let dangerSamples = 0;
    let memoryNoGoSum = 0;
    let memoryNoGoSamples = 0;
    let memorySelectedNoGoCount = 0;
    let memorySelectionSamples = 0;
    let memorySelectedWeightSum = 0;
    let memorySelectedWeightSamples = 0;
    let collisionCountMax = 0;
    let sameWallCollisionCountMax = 0;
    let sameWallConsecutiveCollisionCountMax = 0;
    let collisionOuterNorthMax = 0;
    let collisionOuterSouthMax = 0;
    let collisionOuterEastMax = 0;
    let collisionOuterWestMax = 0;
    let collisionInnerObstacleMax = 0;
    let collisionOutsideBoundsMax = 0;

    const strategyModeCounts = {
        TARGET_LOCK: 0,
        MEMORY_EXPLORE: 0,
        ESCAPE_RECOVERY: 0,
        UNKNOWN: 0
    };

    let firstTs = null;
    let lastTs = null;

    const completeness = {
        distanceToTarget: 0,
        speed: 0,
        minObstacleDist: 0,
        targetHits: 0,
        aiLatencyMs: 0,
        aiStrategyMode: 0,
        aiStrategyConfidence: 0
    };

    history.forEach((pt) => {
        const ts = asNumber(pt.time);
        if (ts !== null) {
            if (firstTs === null || ts < firstTs) firstTs = ts;
            if (lastTs === null || ts > lastTs) lastTs = ts;
        }

        const intent = asNumber(pt.intentionality);
        if (intent !== null) {
            sumIntent += intent;
            intentSamples += 1;
        }

        const steer = asNumber(pt.steering);
        if (steer !== null) {
            sumSteerAbs += Math.abs(steer);
            steerSamples += 1;
        }

        const throttle = asNumber(pt.throttle, 0);
        if (throttle < -0.1) reverseCount += 1;

        const angle = asNumber(pt.targetAngle);
        if (angle !== null && steer !== null) {
            // Current vehicle convention in this app: positive steering = left, negative steering = right.
            const isAligned = (angle > 5 && steer > 0) || (angle < -5 && steer < 0) || Math.abs(angle) <= 5;
            if (isAligned) logicAlignedCount += 1;
            logicSamples += 1;
        }

        if (pt.isStuck) stuckCount += 1;

        const minObstacle = asNumber(pt.minObstacleDist);
        if (minObstacle !== null && minObstacle < 2.5) safetyRiskCount += 1;

        const hasTargetHits = pt.targetHits && typeof pt.targetHits === "object";
        const targetContact = !!pt.targetContact || asNumber(pt.targetHitCount, 0) > 0;
        if (targetContact) targetContactCount += 1;
        if (targetContact) targetContactSamples += 1;

        const speed = asNumber(pt.speed);
        if (speed !== null) {
            sumSpeed += speed;
            speedSamples += 1;
        }

        const progress = asNumber(pt.progressDelta);
        if (progress !== null) {
            if (progress > 0.03) progressTowardCount += 1;
            else if (progress < -0.03) progressAwayCount += 1;
            progressSamples += 1;
        }

        const latency = asNumber(pt.aiLatencyMs);
        if (latency !== null) {
            sumLatency += latency;
            latencySamples += 1;
        }

        const strategyMode = normalizeStrategyMode(pt.aiStrategyMode);
        const strategyTransition = typeof pt.aiStrategyTransition === "string" ? pt.aiStrategyTransition.toUpperCase() : "HOLD";
        const strategyConfidence = asNumber(pt.aiStrategyConfidence);
        const hasStrategyMode = strategyMode !== "UNKNOWN";
        const memoryNoGoRatio = asNumber(pt.memoryNoGoRatio);
        const memorySelectedWeight = asNumber(pt.memorySelectedWeight);
        const memorySelectedNoGo = !!pt.memorySelectedNoGo;
        const collisionCount = asNumber(pt.collisionCount, 0);
        const sameWallCollisionCount = asNumber(pt.sameWallCollisionCount, 0);
        const sameWallConsecutiveCollisionCount = asNumber(pt.sameWallConsecutiveCollisionCount, 0);
        const collisionOuterNorthCount = asNumber(pt.collisionOuterNorthCount, 0);
        const collisionOuterSouthCount = asNumber(pt.collisionOuterSouthCount, 0);
        const collisionOuterEastCount = asNumber(pt.collisionOuterEastCount, 0);
        const collisionOuterWestCount = asNumber(pt.collisionOuterWestCount, 0);
        const collisionInnerObstacleCount = asNumber(pt.collisionInnerObstacleCount, 0);
        const collisionOutsideBoundsCount = asNumber(pt.collisionOutsideBoundsCount, 0);

        if (hasStrategyMode) {
            strategySamples += 1;
            strategyModeCounts[strategyMode] += 1;
            if (strategyTransition === "SWITCH") strategySwitchCount += 1;
            if (strategyConfidence !== null) {
                strategyConfidenceSum += strategyConfidence;
                strategyConfidenceSamples += 1;
            }

            if (memoryNoGoRatio !== null) {
                memoryNoGoSum += clamp(memoryNoGoRatio, 0, 1);
                memoryNoGoSamples += 1;
            }
            if (memorySelectedWeight !== null) {
                memorySelectedWeightSum += memorySelectedWeight;
                memorySelectedWeightSamples += 1;
            }
            memorySelectionSamples += 1;
            if (memorySelectedNoGo) memorySelectedNoGoCount += 1;
        } else {
            strategyModeCounts.UNKNOWN += 1;
        }

        const captureCount = asNumber(pt.targetsReached);
        if (captureCount !== null) targetCaptures = Math.max(targetCaptures, captureCount);
        collisionCountMax = Math.max(collisionCountMax, collisionCount);
        sameWallCollisionCountMax = Math.max(sameWallCollisionCountMax, sameWallCollisionCount);
        sameWallConsecutiveCollisionCountMax = Math.max(sameWallConsecutiveCollisionCountMax, sameWallConsecutiveCollisionCount);
        collisionOuterNorthMax = Math.max(collisionOuterNorthMax, collisionOuterNorthCount);
        collisionOuterSouthMax = Math.max(collisionOuterSouthMax, collisionOuterSouthCount);
        collisionOuterEastMax = Math.max(collisionOuterEastMax, collisionOuterEastCount);
        collisionOuterWestMax = Math.max(collisionOuterWestMax, collisionOuterWestCount);
        collisionInnerObstacleMax = Math.max(collisionInnerObstacleMax, collisionInnerObstacleCount);
        collisionOutsideBoundsMax = Math.max(collisionOutsideBoundsMax, collisionOutsideBoundsCount);

        // Contact-follow quality: when contact exists and front arc is not critical, did steering follow contact direction?
        if (targetContact && (minObstacle === null || minObstacle > 3.0) && steer !== null) {
            const contactDirection = inferContactDirection(pt);
            if (contactDirection) {
                const follows =
                    (contactDirection === "LEFT" && steer > 0.1) ||
                    (contactDirection === "RIGHT" && steer < -0.1) ||
                    (contactDirection === "STRAIGHT" && Math.abs(steer) <= 0.35);
                if (follows) contactFollowAlignedCount += 1;
                contactFollowSamples += 1;
            }
        }

        const angleAbs = Math.abs(asNumber(pt.targetAngle, 180));
        const distance = asNumber(pt.distanceToTarget, 99);
        const isDanger = !!pt.isStuck || (minObstacle !== null && minObstacle < 2.5);
        if (isDanger) {
            dangerSamples += 1;
            if (strategyMode === "ESCAPE_RECOVERY") escapeDangerCount += 1;
        }
        if (targetContact && strategyMode === "TARGET_LOCK") targetLockContactCount += 1;

        if (hasStrategyMode) {
            const expectTargetLock = targetContact || (distance < 14 && angleAbs <= 35 && !isDanger);
            const expectEscape = isDanger;
            const expectedMode = expectEscape ? "ESCAPE_RECOVERY" : expectTargetLock ? "TARGET_LOCK" : "MEMORY_EXPLORE";
            if (strategyMode === expectedMode) strategyModeAlignedCount += 1;
        }

        if (asNumber(pt.distanceToTarget) !== null) completeness.distanceToTarget += 1;
        if (speed !== null) completeness.speed += 1;
        if (minObstacle !== null) completeness.minObstacleDist += 1;
        if (hasTargetHits) completeness.targetHits += 1;
        if (latency !== null) completeness.aiLatencyMs += 1;
        if (hasStrategyMode) completeness.aiStrategyMode += 1;
        if (strategyConfidence !== null) completeness.aiStrategyConfidence += 1;
    });

    const durationSeconds = (firstTs !== null && lastTs !== null && lastTs > firstTs)
        ? (lastTs - firstTs) / 1000
        : history.length * 0.1;

    const completenessKeys = Object.keys(completeness);
    const completenessRatio = completenessKeys.reduce((acc, key) => acc + percent(completeness[key], history.length), 0) / completenessKeys.length;
    const strategyKnownSamples = Math.max(1, strategySamples);

    return {
        count: history.length,
        durationSeconds,
        avgIntentionality: intentSamples > 0 ? sumIntent / intentSamples : 0,
        avgSteeringActivity: steerSamples > 0 ? sumSteerAbs / steerSamples : 0,
        reverseTimeRatio: percent(reverseCount, history.length),
        logicAccuracy: percent(logicAlignedCount, logicSamples),
        avgSpeed: speedSamples > 0 ? sumSpeed / speedSamples : 0,
        stuckRatio: percent(stuckCount, history.length),
        safetyRiskRatio: percent(safetyRiskCount, history.length),
        targetContactRatio: percent(targetContactCount, history.length),
        contactFollowAccuracy: contactFollowSamples > 0 ? percent(contactFollowAlignedCount, contactFollowSamples) : null,
        progressEfficiency: progressSamples > 0 ? percent(progressTowardCount, progressTowardCount + progressAwayCount) : null,
        targetCaptures,
        capturesPerMinute: durationSeconds > 0 ? targetCaptures / (durationSeconds / 60) : 0,
        avgAiLatencyMs: latencySamples > 0 ? sumLatency / latencySamples : null,
        dataCompleteness: completenessRatio,
        strategyCoverage: percent(strategySamples, history.length),
        strategySwitchRate: percent(strategySwitchCount, strategyKnownSamples),
        strategyAvgConfidence: strategyConfidenceSamples > 0 ? (strategyConfidenceSum / strategyConfidenceSamples) : null,
        strategyModeAlignment: percent(strategyModeAlignedCount, strategyKnownSamples),
        targetLockWhenContactRate: percent(targetLockContactCount, Math.max(1, targetContactSamples)),
        escapeWhenDangerRate: percent(escapeDangerCount, Math.max(1, dangerSamples)),
        memoryNoGoRatioAvg: memoryNoGoSamples > 0 ? percent(memoryNoGoSum, memoryNoGoSamples) : null,
        memorySelectedNoGoRate: memorySelectionSamples > 0 ? percent(memorySelectedNoGoCount, memorySelectionSamples) : null,
        memorySelectedWeightAvg: memorySelectedWeightSamples > 0 ? (memorySelectedWeightSum / memorySelectedWeightSamples) : null,
        collisionCount: collisionCountMax,
        sameWallCollisionCount: sameWallCollisionCountMax,
        sameWallConsecutiveCollisionCount: sameWallConsecutiveCollisionCountMax,
        collisionsPerMinute: durationSeconds > 0 ? collisionCountMax / (durationSeconds / 60) : 0,
        collisionByRegion: {
            OUTER_NORTH: collisionOuterNorthMax,
            OUTER_SOUTH: collisionOuterSouthMax,
            OUTER_EAST: collisionOuterEastMax,
            OUTER_WEST: collisionOuterWestMax,
            INNER_OBSTACLE: collisionInnerObstacleMax,
            OUTSIDE_BOUNDS: collisionOutsideBoundsMax
        },
        strategyModeShare: {
            TARGET_LOCK: percent(strategyModeCounts.TARGET_LOCK, strategyKnownSamples),
            MEMORY_EXPLORE: percent(strategyModeCounts.MEMORY_EXPLORE, strategyKnownSamples),
            ESCAPE_RECOVERY: percent(strategyModeCounts.ESCAPE_RECOVERY, strategyKnownSamples)
        }
    };
};

/**
 * Ask Gemma to review the driver based on expanded metrics.
 */
export const generateAIReview = async (history, modelName = "gemma3:12b") => {
    const stats = calculateStats(history);
    if (!stats) return "No data recorded.";

    const contactFollow = stats.contactFollowAccuracy === null ? "N/A" : `${stats.contactFollowAccuracy.toFixed(1)}%`;
    const progressEfficiency = stats.progressEfficiency === null ? "N/A" : `${stats.progressEfficiency.toFixed(1)}%`;
    const avgLatency = stats.avgAiLatencyMs === null ? "N/A" : `${stats.avgAiLatencyMs.toFixed(0)}ms`;
    const strategyConfidence = stats.strategyAvgConfidence === null ? "N/A" : stats.strategyAvgConfidence.toFixed(3);
    const memoryNoGo = stats.memoryNoGoRatioAvg === null ? "N/A" : `${stats.memoryNoGoRatioAvg.toFixed(1)}%`;
    const memorySelectedNoGo = stats.memorySelectedNoGoRate === null ? "N/A" : `${stats.memorySelectedNoGoRate.toFixed(1)}%`;
    const memorySelectedWeight = stats.memorySelectedWeightAvg === null ? "N/A" : stats.memorySelectedWeightAvg.toFixed(3);

    const prompt = `
You are a motorsport telemetry engineer reviewing an autonomous driving agent.

Primary objective of this analysis:
1) Reach blue targets quickly and repeatedly.
2) Stay safe around obstacles (avoid high-risk close calls and deadlock).
3) Keep decisions consistent with target direction and target-contact signals.
4) Evaluate whether LLM strategy-mode selection is coherent with context.

Data Summary:
- Session Duration: ${stats.durationSeconds.toFixed(1)} seconds
- Total Samples: ${stats.count}
- Target Captures: ${stats.targetCaptures}
- Captures per Minute: ${stats.capturesPerMinute.toFixed(2)}
- Intentionality Score: ${stats.avgIntentionality.toFixed(3)}
- Logic Consistency: ${stats.logicAccuracy.toFixed(1)}%
- Target Contact Ratio: ${stats.targetContactRatio.toFixed(1)}%
- Contact-Follow Accuracy: ${contactFollow}
- Progress Efficiency (toward target vs away): ${progressEfficiency}
- Reverse Time: ${stats.reverseTimeRatio.toFixed(1)}%
- Stuck Ratio: ${stats.stuckRatio.toFixed(1)}%
- Safety Risk Ratio (min obstacle < 2.5m): ${stats.safetyRiskRatio.toFixed(1)}%
- Average Speed: ${stats.avgSpeed.toFixed(2)}
- Average AI Latency: ${avgLatency}
- Strategy Coverage: ${stats.strategyCoverage.toFixed(1)}%
- Strategy Mode Alignment: ${stats.strategyModeAlignment.toFixed(1)}%
- Strategy Avg Confidence: ${strategyConfidence}
- TargetLock@Contact: ${stats.targetLockWhenContactRate.toFixed(1)}%
- Escape@Danger: ${stats.escapeWhenDangerRate.toFixed(1)}%
- Memory No-Go Ratio (avg): ${memoryNoGo}
- Memory Selected No-Go Rate: ${memorySelectedNoGo}
- Memory Selected Weight (avg): ${memorySelectedWeight}
- Collision Count: ${stats.collisionCount}
- Same-Wall Collision Repeat: ${stats.sameWallCollisionCount}
- Consecutive Same-Wall Repeat: ${stats.sameWallConsecutiveCollisionCount}
- Collisions per Minute: ${stats.collisionsPerMinute.toFixed(2)}
- Collision By Region: N=${stats.collisionByRegion.OUTER_NORTH}, S=${stats.collisionByRegion.OUTER_SOUTH}, E=${stats.collisionByRegion.OUTER_EAST}, W=${stats.collisionByRegion.OUTER_WEST}, InnerObs=${stats.collisionByRegion.INNER_OBSTACLE}, OOB=${stats.collisionByRegion.OUTSIDE_BOUNDS}
- Mode Share: TARGET_LOCK=${stats.strategyModeShare.TARGET_LOCK.toFixed(1)}%, MEMORY_EXPLORE=${stats.strategyModeShare.MEMORY_EXPLORE.toFixed(1)}%, ESCAPE_RECOVERY=${stats.strategyModeShare.ESCAPE_RECOVERY.toFixed(1)}%
- Data Completeness: ${stats.dataCompleteness.toFixed(1)}%

Task:
1) Give a grade (S, A, B, C, F).
2) Explain performance on target efficiency, safety, and consistency.
3) Call out the single biggest bottleneck.
4) Provide 2 concrete code-level tuning actions.

Output rules:
- Return HTML fragment only (no markdown, no code fences, no <html>/<body> wrapper).
- Keep it concise and actionable.
`;

    try {
        const response = await fetch("http://localhost:11434/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: modelName,
                prompt,
                stream: false
            }),
        });

        if (!response.ok) {
            throw new Error(`Ollama API error: ${response.statusText}`);
        }

        const data = await response.json();
        return normalizeReviewHtml(data.response);
    } catch (e) {
        console.error("Analysis Error:", e);
        return "<b>Error:</b> Could not contact AI Analyst.";
    }
};

/**
 * Build HTML report content.
 */
export const buildHTMLReportContent = (history, aiReview) => {
    const stats = calculateStats(history);
    if (!stats) return null;

    const sampled = downsampleHistory(history, 1400);
    const firstSampleTs = asNumber(sampled[0]?.time, 0);
    const labels = sampled.map((pt, i) => {
        const t = asNumber(pt.time);
        if (t !== null && firstSampleTs !== null) return ((t - firstSampleTs) / 1000).toFixed(1);
        return (i * 0.1).toFixed(1);
    });

    const dataIntent = sampled.map(pt => asNumber(pt.intentionality, 0));
    const dataAngle = sampled.map(pt => clamp(asNumber(pt.targetAngle, 0) / 180, -1, 1));
    const dataSteer = sampled.map(pt => asNumber(pt.steering, 0));
    const dataThrottle = sampled.map(pt => asNumber(pt.throttle, 0));

    const distanceValues = sampled.map(pt => asNumber(pt.distanceToTarget, 0));
    const maxDistance = Math.max(1, ...distanceValues);
    const dataDistanceNorm = distanceValues.map(v => clamp(v / maxDistance, 0, 1));
    const dataMinObstacleNorm = sampled.map(pt => clamp(asNumber(pt.minObstacleDist, 10) / 10, 0, 1));
    const dataTargetContact = sampled.map(pt => (pt.targetContact ? 1 : 0));
    const dataStrategyTargetLock = sampled.map(pt => normalizeStrategyMode(pt.aiStrategyMode) === "TARGET_LOCK" ? 1 : 0);
    const dataStrategyExplore = sampled.map(pt => normalizeStrategyMode(pt.aiStrategyMode) === "MEMORY_EXPLORE" ? 1 : 0);
    const dataStrategyEscape = sampled.map(pt => normalizeStrategyMode(pt.aiStrategyMode) === "ESCAPE_RECOVERY" ? 1 : 0);
    const dataStrategyConfidence = sampled.map(pt => clamp(asNumber(pt.aiStrategyConfidence, 0), 0, 1));
    const dataStrategySwitch = sampled.map(pt => (typeof pt.aiStrategyTransition === "string" && pt.aiStrategyTransition.toUpperCase() === "SWITCH") ? 1 : 0);

    const contactFollowText = stats.contactFollowAccuracy === null ? "N/A" : `${stats.contactFollowAccuracy.toFixed(1)}%`;
    const progressText = stats.progressEfficiency === null ? "N/A" : `${stats.progressEfficiency.toFixed(1)}%`;
    const latencyText = stats.avgAiLatencyMs === null ? "N/A" : `${stats.avgAiLatencyMs.toFixed(0)}ms`;
    const strategyConfidenceText = stats.strategyAvgConfidence === null ? "N/A" : stats.strategyAvgConfidence.toFixed(3);
    const memoryNoGoText = stats.memoryNoGoRatioAvg === null ? "N/A" : `${stats.memoryNoGoRatioAvg.toFixed(1)}%`;
    const memorySelectedNoGoText = stats.memorySelectedNoGoRate === null ? "N/A" : `${stats.memorySelectedNoGoRate.toFixed(1)}%`;
    const memorySelectedWeightText = stats.memorySelectedWeightAvg === null ? "N/A" : stats.memorySelectedWeightAvg.toFixed(3);
    const collisionsPerMinuteText = Number.isFinite(stats.collisionsPerMinute) ? stats.collisionsPerMinute.toFixed(2) : "0.00";
    const wallByRegionText = `N:${stats.collisionByRegion.OUTER_NORTH} S:${stats.collisionByRegion.OUTER_SOUTH} E:${stats.collisionByRegion.OUTER_EAST} W:${stats.collisionByRegion.OUTER_WEST}`;
    const reviewHtml = normalizeReviewHtml(aiReview);

    const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>Gemma 3 Telemetry Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; background: #111; color: #eee; padding: 20px; }
        .container { max-width: 1100px; margin: 0 auto; }
        .card { background: #222; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #444; }
        h1 { color: #22c55e; }
        h2 { color: #a855f7; border-bottom: 1px solid #444; padding-bottom: 5px; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        .stat-box { background: #333; padding: 10px; text-align: center; border-radius: 5px; }
        .stat-val { font-size: 22px; font-weight: bold; display: block; }
        .stat-label { font-size: 12px; color: #aaa; }
        .objective-list li { margin-bottom: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Gemma 3 Autonomous Report</h1>
            <p>Generated at ${new Date().toLocaleString()}</p>

            <h2>Analysis Objective</h2>
            <ul class="objective-list">
                <li>Reach blue targets quickly and repeatedly.</li>
                <li>Avoid dangerous close-range wall situations and deadlock.</li>
                <li>Maintain consistent steering decisions toward target cues.</li>
            </ul>

            <div class="stats-grid">
                <div class="stat-box"><span class="stat-val">${stats.targetCaptures}</span><span class="stat-label">Target Captures</span></div>
                <div class="stat-box"><span class="stat-val">${stats.capturesPerMinute.toFixed(2)}</span><span class="stat-label">Captures / Min</span></div>
                <div class="stat-box"><span class="stat-val" style="color:${stats.avgIntentionality > 0.5 ? "#4ade80" : "#facc15"}">${stats.avgIntentionality.toFixed(3)}</span><span class="stat-label">Intentionality</span></div>
                <div class="stat-box"><span class="stat-val">${stats.logicAccuracy.toFixed(1)}%</span><span class="stat-label">Logic Accuracy</span></div>
                <div class="stat-box"><span class="stat-val">${contactFollowText}</span><span class="stat-label">Contact Follow</span></div>
                <div class="stat-box"><span class="stat-val">${progressText}</span><span class="stat-label">Progress Efficiency</span></div>
                <div class="stat-box"><span class="stat-val">${stats.safetyRiskRatio.toFixed(1)}%</span><span class="stat-label">Safety Risk</span></div>
                <div class="stat-box"><span class="stat-val">${stats.stuckRatio.toFixed(1)}%</span><span class="stat-label">Stuck Ratio</span></div>
                <div class="stat-box"><span class="stat-val">${stats.reverseTimeRatio.toFixed(1)}%</span><span class="stat-label">Reverse Time</span></div>
                <div class="stat-box"><span class="stat-val">${stats.avgSpeed.toFixed(2)}</span><span class="stat-label">Average Speed</span></div>
                <div class="stat-box"><span class="stat-val">${latencyText}</span><span class="stat-label">Avg AI Latency</span></div>
                <div class="stat-box"><span class="stat-val">${stats.strategyCoverage.toFixed(1)}%</span><span class="stat-label">Strategy Coverage</span></div>
                <div class="stat-box"><span class="stat-val">${stats.strategyModeAlignment.toFixed(1)}%</span><span class="stat-label">Strategy Alignment</span></div>
                <div class="stat-box"><span class="stat-val">${strategyConfidenceText}</span><span class="stat-label">Strategy Confidence</span></div>
                <div class="stat-box"><span class="stat-val">${stats.strategySwitchRate.toFixed(1)}%</span><span class="stat-label">Strategy Switch Rate</span></div>
                <div class="stat-box"><span class="stat-val">${stats.targetLockWhenContactRate.toFixed(1)}%</span><span class="stat-label">TargetLock@Contact</span></div>
                <div class="stat-box"><span class="stat-val">${stats.escapeWhenDangerRate.toFixed(1)}%</span><span class="stat-label">Escape@Danger</span></div>
                <div class="stat-box"><span class="stat-val">${memoryNoGoText}</span><span class="stat-label">Memory No-Go (avg)</span></div>
                <div class="stat-box"><span class="stat-val">${memorySelectedNoGoText}</span><span class="stat-label">Selected No-Go</span></div>
                <div class="stat-box"><span class="stat-val">${memorySelectedWeightText}</span><span class="stat-label">Selected Weight</span></div>
                <div class="stat-box"><span class="stat-val">${stats.collisionCount}</span><span class="stat-label">Collision Count</span></div>
                <div class="stat-box"><span class="stat-val">${stats.sameWallCollisionCount}</span><span class="stat-label">Same-Wall Repeat</span></div>
                <div class="stat-box"><span class="stat-val">${stats.sameWallConsecutiveCollisionCount}</span><span class="stat-label">Consecutive Same-Wall</span></div>
                <div class="stat-box"><span class="stat-val">${collisionsPerMinuteText}</span><span class="stat-label">Collisions / Min</span></div>
                <div class="stat-box"><span class="stat-val">${wallByRegionText}</span><span class="stat-label">Wall Collision (N/S/E/W)</span></div>
                <div class="stat-box"><span class="stat-val">${stats.collisionByRegion.INNER_OBSTACLE}</span><span class="stat-label">Inner Obstacle Hits</span></div>
                <div class="stat-box"><span class="stat-val">${stats.collisionByRegion.OUTSIDE_BOUNDS}</span><span class="stat-label">Outside Bounds Hits</span></div>
                <div class="stat-box"><span class="stat-val">${stats.strategyModeShare.TARGET_LOCK.toFixed(1)}%</span><span class="stat-label">Mode Share: LOCK</span></div>
                <div class="stat-box"><span class="stat-val">${stats.strategyModeShare.MEMORY_EXPLORE.toFixed(1)}%</span><span class="stat-label">Mode Share: EXPLORE</span></div>
                <div class="stat-box"><span class="stat-val">${stats.strategyModeShare.ESCAPE_RECOVERY.toFixed(1)}%</span><span class="stat-label">Mode Share: ESCAPE</span></div>
                <div class="stat-box"><span class="stat-val">${stats.dataCompleteness.toFixed(1)}%</span><span class="stat-label">Data Completeness</span></div>
                <div class="stat-box"><span class="stat-val">${stats.durationSeconds.toFixed(1)}s</span><span class="stat-label">Session Duration</span></div>
                <div class="stat-box"><span class="stat-val">${stats.count}</span><span class="stat-label">Data Points</span></div>
            </div>
        </div>

        <div class="card">
            <h2>AI Analyst Review</h2>
            <div style="font-size: 16px; line-height: 1.6;">${reviewHtml}</div>
        </div>

        <div class="card">
            <h2>Control vs Target</h2>
            <canvas id="controlChart"></canvas>
        </div>

        <div class="card">
            <h2>Objective Tracking</h2>
            <canvas id="objectiveChart"></canvas>
        </div>

        <div class="card">
            <h2>Strategy Diagnostics</h2>
            <canvas id="strategyChart"></canvas>
        </div>
    </div>

    <script>
        const labels = [${labels.join(",")}];

        const controlCtx = document.getElementById('controlChart').getContext('2d');
        new Chart(controlCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Intentionality', data: [${dataIntent.join(",")}], borderColor: '#4ade80', backgroundColor: 'rgba(74, 222, 128, 0.1)', fill: true, tension: 0.35 },
                    { label: 'Target Angle (Norm)', data: [${dataAngle.join(",")}], borderColor: '#22d3ee', borderDash: [5, 5], tension: 0.1 },
                    { label: 'Steering', data: [${dataSteer.join(",")}], borderColor: '#e879f9', tension: 0.1 },
                    { label: 'Throttle', data: [${dataThrottle.join(",")}], borderColor: '#fbbf24', borderDash: [2, 2], tension: 0.1 }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { ticks: { color: '#888' } },
                    y: { min: -1.2, max: 1.2, ticks: { color: '#888' }, grid: { color: '#333' } }
                }
            }
        });

        const objectiveCtx = document.getElementById('objectiveChart').getContext('2d');
        new Chart(objectiveCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Distance to Target (Norm, lower is better)', data: [${dataDistanceNorm.join(",")}], borderColor: '#60a5fa', tension: 0.2 },
                    { label: 'Min Obstacle Distance (Norm)', data: [${dataMinObstacleNorm.join(",")}], borderColor: '#f97316', tension: 0.2 },
                    { label: 'Target Contact (0/1)', data: [${dataTargetContact.join(",")}], borderColor: '#14b8a6', borderDash: [4, 4], tension: 0 }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { ticks: { color: '#888' } },
                    y: { min: 0, max: 1.05, ticks: { color: '#888' }, grid: { color: '#333' } }
                }
            }
        });

        const strategyCtx = document.getElementById('strategyChart').getContext('2d');
        new Chart(strategyCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Mode: TARGET_LOCK', data: [${dataStrategyTargetLock.join(",")}], borderColor: '#22c55e', tension: 0.05 },
                    { label: 'Mode: MEMORY_EXPLORE', data: [${dataStrategyExplore.join(",")}], borderColor: '#38bdf8', tension: 0.05 },
                    { label: 'Mode: ESCAPE_RECOVERY', data: [${dataStrategyEscape.join(",")}], borderColor: '#f97316', tension: 0.05 },
                    { label: 'Strategy Confidence', data: [${dataStrategyConfidence.join(",")}], borderColor: '#fbbf24', borderDash: [4, 3], tension: 0.1 },
                    { label: 'Transition SWITCH (0/1)', data: [${dataStrategySwitch.join(",")}], borderColor: '#e879f9', borderDash: [2, 2], tension: 0 }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { ticks: { color: '#888' } },
                    y: { min: 0, max: 1.05, ticks: { color: '#888' }, grid: { color: '#333' } }
                }
            }
        });
    </script>
</body>
</html>
    `;

    return htmlContent;
};

/**
 * Generate and download HTML report.
 */
export const downloadHTMLReport = (history, aiReview) => {
    const htmlContent = buildHTMLReportContent(history, aiReview);
    if (!htmlContent) return;

    const blob = new Blob([htmlContent], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `driver_limit_report_${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
};
