const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const asNumber = (value, fallback = null) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);
const round = (value, digits = 3) => {
    if (!Number.isFinite(value)) return 0;
    const base = 10 ** digits;
    return Math.round(value * base) / base;
};

const cellKey = (ix, iz) => `${ix},${iz}`;

const parseCellKey = (key) => {
    const [x, z] = key.split(",");
    return { ix: Number(x), iz: Number(z) };
};

const toGrid = (x, z, cellSize) => ({
    ix: Math.floor(x / cellSize),
    iz: Math.floor(z / cellSize),
});

const toCellCenter = (ix, iz, cellSize) => ({
    x: (ix + 0.5) * cellSize,
    z: (iz + 0.5) * cellSize,
});

const SENSOR_RAYS = [
    { name: "front", offsetDeg: 0 },
    { name: "leftDiag", offsetDeg: 45 },
    { name: "left", offsetDeg: 90 },
    { name: "backLeft", offsetDeg: 135 },
    { name: "back", offsetDeg: 180 },
    { name: "backRight", offsetDeg: -135 },
    { name: "right", offsetDeg: -90 },
    { name: "rightDiag", offsetDeg: -45 },
];
const OUTER_WALL_REPEAT_HARD_NO_GO_HITS = 2;
const OBSTACLE_REPEAT_HARD_NO_GO_HITS = 3;

const getBoundsPenalty = (x, z, bounds) => {
    if (!bounds) return { outside: false, penalty: 0 };

    const softMargin = bounds.softMargin ?? 2.5;
    const outside = x < bounds.minX || x > bounds.maxX || z < bounds.minZ || z > bounds.maxZ;
    if (outside) return { outside: true, penalty: 2.0 };

    const edgeDist = Math.min(
        x - bounds.minX,
        bounds.maxX - x,
        z - bounds.minZ,
        bounds.maxZ - z
    );
    const edgePenalty = edgeDist < softMargin ? ((softMargin - edgeDist) / softMargin) * 0.55 : 0;
    return { outside: false, penalty: edgePenalty };
};

const isInsideBounds = (x, z, bounds) => {
    if (!bounds) return true;
    return x >= bounds.minX && x <= bounds.maxX && z >= bounds.minZ && z <= bounds.maxZ;
};

const clampPointToBounds = (x, z, bounds, epsilon = 0.001) => {
    if (!bounds) return { x, z };
    return {
        x: clamp(x, bounds.minX + epsilon, bounds.maxX - epsilon),
        z: clamp(z, bounds.minZ + epsilon, bounds.maxZ - epsilon),
    };
};

const isNearOuterBoundary = (x, z, bounds, margin = 0.75) => {
    if (!bounds) return false;
    const nearMinX = Math.abs(x - bounds.minX) <= margin;
    const nearMaxX = Math.abs(x - bounds.maxX) <= margin;
    const nearMinZ = Math.abs(z - bounds.minZ) <= margin;
    const nearMaxZ = Math.abs(z - bounds.maxZ) <= margin;
    return nearMinX || nearMaxX || nearMinZ || nearMaxZ;
};

const relativeSector = (fromX, fromZ, headingDeg, toX, toZ) => {
    const dx = toX - fromX;
    const dz = toZ - fromZ;
    const mag = Math.hypot(dx, dz);
    if (mag < 1e-6) return "F";

    // Heading 0 deg is +Z.
    const headingRad = (headingDeg * Math.PI) / 180;
    const fx = Math.sin(headingRad);
    const fz = Math.cos(headingRad);

    // Signed angle from forward to target.
    const dot = clamp((fx * dx + fz * dz) / mag, -1, 1);
    const crossY = fx * dz - fz * dx;
    const angle = Math.atan2(crossY, dot) * (180 / Math.PI);
    const abs = Math.abs(angle);

    if (abs <= 30) return "F";
    if (abs >= 150) return "B";
    return angle > 0 ? "R" : "L";
};

export function createExplorationMemory(options = {}) {
    const state = {
        cellSize: options.cellSize ?? 2.0,
        sensorRange: options.sensorRange ?? 10.0,
        maxCells: options.maxCells ?? 5000,
        worldBounds: options.worldBounds ?? null,
        cells: new Map(),
        path: [],
    };

    const getOrCreateCell = (ix, iz) => {
        const key = cellKey(ix, iz);
        let cell = state.cells.get(key);
        if (!cell) {
            cell = {
                visits: 0,
                lastSeen: 0,
                riskEMA: 0,
                stuckCount: 0,
                targetHitCount: 0,
                targetMissCount: 0,
                targetAbsenceEMA: 0,
                lastTargetSeenAt: 0,
                lastTargetMissAt: 0,
                obstacleHits: 0,
                outerWallHits: 0,
                openHits: 0,
            };
            state.cells.set(key, cell);
        }
        return cell;
    };

    const pruneIfNeeded = () => {
        if (state.cells.size <= state.maxCells) return;

        const items = Array.from(state.cells.entries())
            .sort((a, b) => (a[1].lastSeen || 0) - (b[1].lastSeen || 0));

        const toDelete = Math.max(1, Math.floor(items.length * 0.1));
        for (let i = 0; i < toDelete; i += 1) {
            state.cells.delete(items[i][0]);
        }
    };

    const isCellObstacleDominant = (cell) => {
        if (!cell) return false;
        const obstacleHits = cell.obstacleHits ?? 0;
        const outerWallHits = cell.outerWallHits ?? 0;
        const openHits = cell.openHits ?? 0;
        const risk = clamp(cell.riskEMA ?? 0, 0, 1);
        const repeatedOuterWall = outerWallHits >= OUTER_WALL_REPEAT_HARD_NO_GO_HITS;
        const repeatedObstacle = obstacleHits >= 2 && obstacleHits >= (openHits * 0.9);
        const singleHardObstacle = obstacleHits >= 1 && (openHits === 0 || risk >= 0.62);
        return repeatedOuterWall || repeatedObstacle || singleHardObstacle;
    };

    const hasBarrierBetween = (fromIx, fromIz, toIx, toIz) => {
        const dx = toIx - fromIx;
        const dz = toIz - fromIz;
        const steps = Math.max(Math.abs(dx), Math.abs(dz));
        if (steps <= 1) {
            const targetCell = state.cells.get(cellKey(toIx, toIz));
            if (!targetCell) return false;
            const targetRisk = clamp(targetCell.riskEMA ?? 0, 0, 1);
            return isCellObstacleDominant(targetCell) || targetRisk > 0.72;
        }

        for (let i = 1; i < steps; i += 1) {
            const sx = Math.round(fromIx + ((dx * i) / steps));
            const sz = Math.round(fromIz + ((dz * i) / steps));
            const cell = state.cells.get(cellKey(sx, sz));
            if (!cell) continue;

            const risk = clamp(cell.riskEMA ?? 0, 0, 1);
            if (isCellObstacleDominant(cell) || risk > 0.78) {
                return true;
            }
        }

        return false;
    };

    const getNoGoReasons = (candidate) => {
        const reasons = [];
        if (candidate.outsideBounds) reasons.push("outsideBounds");
        if (candidate.barrierBlocked) reasons.push("barrierBlocked");
        if (candidate.obstacleDominant) reasons.push("obstacleDominant");
        if ((candidate.obstacleHits ?? 0) >= 1 && candidate.risk >= 0.62) reasons.push("recentObstacleHit");
        if ((candidate.obstacleHits ?? 0) >= OBSTACLE_REPEAT_HARD_NO_GO_HITS) reasons.push("obstacleRepeat");
        if ((candidate.outerWallHits ?? 0) >= OUTER_WALL_REPEAT_HARD_NO_GO_HITS) reasons.push("outerWallRepeat");
        if ((candidate.repeatPenalty ?? 0) >= 0.85) reasons.push("repeatPenaltyHigh");
        if (candidate.risk >= 0.78) reasons.push("highRisk");
        return reasons;
    };

    const toCandidateSummary = (candidate, includeFlags = true) => {
        if (!candidate) return null;
        const summary = {
            ix: candidate.ix,
            iz: candidate.iz,
            sector: candidate.sector,
            score: round(candidate.score),
            risk: round(candidate.risk),
            visits: candidate.visits,
            targetHitCount: candidate.targetHitCount ?? 0,
            targetMissCount: candidate.targetMissCount ?? 0,
            targetAbsence: round(candidate.targetAbsenceEMA ?? 0),
            targetPenalty: round(candidate.targetPenalty ?? 0),
            obstacleHits: candidate.obstacleHits ?? 0,
            outerWallHits: candidate.outerWallHits ?? 0,
            repeatPenalty: round(candidate.repeatPenalty ?? 0),
        };
        if (includeFlags) {
            const noGoReasons = candidate.noGoReasons || [];
            summary.isNoGo = noGoReasons.length > 0;
            summary.noGoReasons = noGoReasons;
        }
        return summary;
    };

    const evaluateCandidate = (originX, originZ, headingDeg, originIx, originIz, nix, niz) => {
        const key = cellKey(nix, niz);
        const cell = state.cells.get(key);
        const visits = cell?.visits ?? 0;
        const risk = clamp(cell?.riskEMA ?? 0, 0, 1);
        const openHits = cell?.openHits ?? 0;
        const obstacleHits = cell?.obstacleHits ?? 0;
        const outerWallHits = cell?.outerWallHits ?? 0;
        const evidence = openHits + obstacleHits;
        const obstacleDominant = isCellObstacleDominant(cell);
        const openBias = openHits / Math.max(1, evidence);
        const novelty = 1 / (1 + visits);
        const targetHitCount = cell?.targetHitCount ?? 0;
        const targetMissCount = cell?.targetMissCount ?? 0;
        const targetAbsenceEMA = clamp(cell?.targetAbsenceEMA ?? (targetMissCount > 0 ? targetMissCount / 18 : 0), 0, 1);
        const targetPenalty = targetHitCount > 0
            ? 0
            : clamp((targetAbsenceEMA * 0.6) + (Math.max(0, targetMissCount - 2) / 45), 0, 0.9);
        const dx = nix - originIx;
        const dz = niz - originIz;
        const distance = Math.hypot(dx, dz);
        const center = toCellCenter(nix, niz, state.cellSize);
        const sector = relativeSector(originX, originZ, headingDeg, center.x, center.z);
        const barrierBlocked = hasBarrierBetween(originIx, originIz, nix, niz);
        const boundsPenaltyInfo = getBoundsPenalty(center.x, center.z, state.worldBounds);
        const unknownPenalty = evidence === 0 ? 0.15 : 0;
        const obstacleRepeatPenalty = clamp(Math.max(0, obstacleHits - 1) * 0.12, 0, 0.85);
        const outerWallRepeatPenalty = clamp(outerWallHits * 0.22, 0, 1.2);
        const repeatPenalty = clamp(obstacleRepeatPenalty + outerWallRepeatPenalty, 0, 1.5);

        let score = (novelty * 0.95) + (openBias * 0.4) - (risk * 0.95) - (targetPenalty * 0.95) - (distance * 0.09);
        if (visits === 0 && !barrierBlocked && !boundsPenaltyInfo.outside) {
            score += 0.22;
        }
        score -= boundsPenaltyInfo.penalty;
        score -= unknownPenalty;
        if (barrierBlocked) score -= 0.9;
        if (boundsPenaltyInfo.outside) score -= 1.2;
        if (obstacleDominant) score -= 0.95;
        if (obstacleHits >= 1 && risk >= 0.62) score -= 0.55;
        if (obstacleHits >= OBSTACLE_REPEAT_HARD_NO_GO_HITS) score -= 0.45;
        if (outerWallHits >= OUTER_WALL_REPEAT_HARD_NO_GO_HITS) score -= 0.85;
        score -= repeatPenalty;

        return {
            ix: nix,
            iz: niz,
            dx,
            dz,
            visits,
            risk,
            novelty,
            openBias,
            distance,
            sector,
            score,
            targetPenalty,
            targetHitCount,
            targetMissCount,
            targetAbsenceEMA,
            barrierBlocked,
            outsideBounds: boundsPenaltyInfo.outside,
            obstacleDominant,
            obstacleHits,
            outerWallHits,
            openHits,
            repeatPenalty,
            hasData: !!cell,
        };
    };

    const update = (sensorData, now = Date.now()) => {
        const x = asNumber(sensorData?.worldX);
        const z = asNumber(sensorData?.worldZ);
        if (x === null || z === null) return;
        const sensorRange = clamp(asNumber(sensorData?.sensorRange, state.sensorRange), 4.0, 30.0);
        state.sensorRange = sensorRange;

        const headingDeg = asNumber(sensorData?.headingDeg, 0);
        const { ix, iz } = toGrid(x, z, state.cellSize);
        const currentCell = getOrCreateCell(ix, iz);

        const rayDistances = [
            sensorData?.front,
            sensorData?.leftDiag,
            sensorData?.rightDiag,
            sensorData?.left,
            sensorData?.right,
            sensorData?.back,
            sensorData?.backLeft,
            sensorData?.backRight,
        ].map((v) => asNumber(v, sensorRange));

        const minObstacleDist = Math.min(...rayDistances);
        const riskInstant = clamp((3.2 - minObstacleDist) / 3.2, 0, 1);

        currentCell.visits += 1;
        currentCell.lastSeen = now;
        currentCell.riskEMA = currentCell.visits === 1
            ? riskInstant
            : (currentCell.riskEMA * 0.85) + (riskInstant * 0.15);
        if (sensorData?.isStuck) currentCell.stuckCount += 1;

        const hits = sensorData?.targetHits || {};
        const hitCount = Object.values(hits).filter(Boolean).length;
        if (hitCount > 0) {
            currentCell.targetHitCount += hitCount;
            currentCell.lastTargetSeenAt = now;
            currentCell.targetAbsenceEMA = clamp((currentCell.targetAbsenceEMA ?? 0) * 0.75, 0, 1);
        } else {
            currentCell.targetMissCount += 1;
            currentCell.lastTargetMissAt = now;
            const prevAbsence = clamp(currentCell.targetAbsenceEMA ?? 0, 0, 1);
            currentCell.targetAbsenceEMA = clamp((prevAbsence * 0.92) + 0.08, 0, 1);
        }

        state.path.push({ ix, iz, t: now });
        if (state.path.length > 500) state.path.shift();

        // Integrate all rays so wall direction is encoded in memory map, not only front.
        SENSOR_RAYS.forEach((ray) => {
            const dist = asNumber(sensorData?.[ray.name], sensorRange);
            const worldDeg = headingDeg + ray.offsetDeg;
            const worldRad = (worldDeg * Math.PI) / 180;
            const dirX = Math.sin(worldRad);
            const dirZ = Math.cos(worldRad);
            const rayDist = clamp(dist, 0, sensorRange);
            const hasObstacleHit = rayDist < sensorRange - 0.05;
            const travelDist = hasObstacleHit ? rayDist : sensorRange;

            // Record traversed cells as open evidence (free-space map), not only hit endpoint.
            const step = Math.max(0.35, state.cellSize * 0.6);
            const visitedOpen = new Set();
            for (let t = step; t < Math.max(step, travelDist - 0.15); t += step) {
                const sampleX = x + (dirX * t);
                const sampleZ = z + (dirZ * t);
                if (!isInsideBounds(sampleX, sampleZ, state.worldBounds)) break;

                const sampleGrid = toGrid(sampleX, sampleZ, state.cellSize);
                const sampleKey = cellKey(sampleGrid.ix, sampleGrid.iz);
                if (visitedOpen.has(sampleKey)) continue;
                visitedOpen.add(sampleKey);

                const openCell = getOrCreateCell(sampleGrid.ix, sampleGrid.iz);
                openCell.openHits += 1;
                openCell.lastSeen = now;
            }

            if (hasObstacleHit) {
                const hitX = x + (dirX * rayDist);
                const hitZ = z + (dirZ * rayDist);
                const boundedHit = clampPointToBounds(hitX, hitZ, state.worldBounds);
                if (isInsideBounds(boundedHit.x, boundedHit.z, state.worldBounds)) {
                    const hitGrid = toGrid(boundedHit.x, boundedHit.z, state.cellSize);
                    const hitCell = getOrCreateCell(hitGrid.ix, hitGrid.iz);
                    const localRisk = clamp((sensorRange - rayDist) / sensorRange, 0, 1);
                    const impactRisk = clamp((localRisk * 0.85) + 0.15, 0, 1);
                    const nearOuterWall = isNearOuterBoundary(
                        boundedHit.x,
                        boundedHit.z,
                        state.worldBounds,
                        Math.max(0.4, state.cellSize * 0.65)
                    );
                    const immediateHazardFloor = nearOuterWall
                        ? (rayDist < 3.2 ? 0.94 : 0.84)
                        : (rayDist < 2.8 ? 0.88 : rayDist < 4.2 ? 0.74 : 0.58);
                    hitCell.obstacleHits += 1;
                    if (nearOuterWall) hitCell.outerWallHits += 1;
                    hitCell.lastSeen = now;
                    hitCell.riskEMA = Math.max((hitCell.riskEMA * 0.78) + (impactRisk * 0.22), immediateHazardFloor);
                }
            } else {
                const openX = x + (dirX * sensorRange * 0.8);
                const openZ = z + (dirZ * sensorRange * 0.8);
                if (isInsideBounds(openX, openZ, state.worldBounds)) {
                    const openGrid = toGrid(openX, openZ, state.cellSize);
                    const openCell = getOrCreateCell(openGrid.ix, openGrid.iz);
                    openCell.openHits += 1;
                    openCell.lastSeen = now;
                }
            }
        });

        pruneIfNeeded();
    };

    const getContext = (sensorData, optionsArg = {}) => {
        const x = asNumber(sensorData?.worldX);
        const z = asNumber(sensorData?.worldZ);
        if (x === null || z === null) return null;

        const headingDeg = asNumber(sensorData?.headingDeg, 0);
        const radiusCells = optionsArg.radiusCells ?? 2;
        const maxFrontier = optionsArg.maxFrontier ?? 6;
        const maxRisky = optionsArg.maxRisky ?? 5;

        const { ix, iz } = toGrid(x, z, state.cellSize);
        const currentCell = state.cells.get(cellKey(ix, iz)) || {
            visits: 0,
            riskEMA: 0,
            stuckCount: 0,
            targetHitCount: 0,
            targetMissCount: 0,
            targetAbsenceEMA: 0,
            obstacleHits: 0,
            outerWallHits: 0,
            lastSeen: 0,
        };

        const candidates = [];
        const sectorTotals = {
            L: { novelty: 0, risk: 0, open: 0, targetPenalty: 0, count: 0 },
            F: { novelty: 0, risk: 0, open: 0, targetPenalty: 0, count: 0 },
            R: { novelty: 0, risk: 0, open: 0, targetPenalty: 0, count: 0 },
            B: { novelty: 0, risk: 0, open: 0, targetPenalty: 0, count: 0 },
        };

        for (let dz = -radiusCells; dz <= radiusCells; dz += 1) {
            for (let dx = -radiusCells; dx <= radiusCells; dx += 1) {
                if (dx === 0 && dz === 0) continue;

                const nix = ix + dx;
                const niz = iz + dz;
                const candidate = evaluateCandidate(x, z, headingDeg, ix, iz, nix, niz);
                candidates.push(candidate);

                if (candidate.outsideBounds) continue;
                const bucket = sectorTotals[candidate.sector];
                bucket.novelty += candidate.novelty;
                bucket.risk += candidate.risk;
                bucket.open += candidate.openBias;
                bucket.targetPenalty += candidate.targetPenalty;
                bucket.count += 1;
            }
        }

        const sectorScores = Object.entries(sectorTotals).map(([sector, bucket]) => {
            const count = Math.max(1, bucket.count);
            const novelty = bucket.novelty / count;
            const risk = bucket.risk / count;
            const open = bucket.open / count;
            const targetPenalty = bucket.targetPenalty / count;
            const score = (novelty * 1.2) + (open * 0.4) - (risk * 0.95) - (targetPenalty * 0.9);
            return { sector, novelty, risk, open, targetPenalty, score };
        });

        sectorScores.sort((a, b) => b.score - a.score);
        const preferredSector = sectorScores[0]?.sector || "F";

        const diagnosticsCandidates = candidates.map((candidate) => {
            const noGoReasons = getNoGoReasons(candidate);
            return {
                ...candidate,
                noGoReasons,
            };
        });

        const safeCandidates = diagnosticsCandidates.filter((c) => c.noGoReasons.length === 0);

        const frontier = safeCandidates
            .filter((c) => c.risk < 0.9)
            .sort((a, b) => b.score - a.score)
            .slice(0, maxFrontier)
            .map((c) => toCandidateSummary(c, false));

        const risky = diagnosticsCandidates
            .filter((c) => c.risk > 0.5)
            .sort((a, b) => b.risk - a.risk)
            .slice(0, maxRisky)
            .map((c) => ({
                ix: c.ix,
                iz: c.iz,
                sector: c.sector,
                risk: round(c.risk),
                visits: c.visits,
                barrierBlocked: c.barrierBlocked,
                outsideBounds: c.outsideBounds,
                obstacleDominant: c.obstacleDominant,
            }));

        const recentPath = state.path.slice(-80);
        const sameCellCount = recentPath.reduce((acc, p) => (
            acc + ((p.ix === ix && p.iz === iz) ? 1 : 0)
        ), 0);
        const loopRate = recentPath.length > 0 ? sameCellCount / recentPath.length : 0;

        const loopWarning = loopRate > 0.35 ? "HIGH_REVISIT_LOOP_RISK" : "LOW";
        const noGoCount = diagnosticsCandidates.filter((c) => c.noGoReasons.length > 0).length;
        const noGoRatio = diagnosticsCandidates.length > 0 ? noGoCount / diagnosticsCandidates.length : 0;
        const sortedCandidates = diagnosticsCandidates
            .slice()
            .sort((a, b) => b.score - a.score);
        const sortedSafeCandidates = safeCandidates
            .slice()
            .sort((a, b) => b.score - a.score);
        const preferredCandidate = sortedSafeCandidates.find((c) => c.sector === preferredSector)
            || sortedSafeCandidates[0]
            || sortedCandidates[0]
            || null;

        const sectorSafety = ["L", "F", "R", "B"].map((sector) => {
            const sectorCandidates = diagnosticsCandidates.filter((c) => c.sector === sector);
            const total = sectorCandidates.length;
            const noGo = sectorCandidates.filter((c) => c.noGoReasons.length > 0).length;
            const safe = total - noGo;
            const bestSafe = sectorCandidates
                .filter((c) => c.noGoReasons.length === 0)
                .sort((a, b) => b.score - a.score)[0] || null;
            return {
                sector,
                totalCount: total,
                safeCount: safe,
                noGoCount: noGo,
                noGoRatio: round(total > 0 ? noGo / total : 0),
                bestSafeScore: bestSafe ? round(bestSafe.score) : null
            };
        });

        const currentCellOpenHits = currentCell.openHits ?? 0;
        const currentCellObstacleHits = currentCell.obstacleHits ?? 0;
        const currentCellEvidence = currentCellOpenHits + currentCellObstacleHits;
        const currentCellOpenBias = currentCellOpenHits / Math.max(1, currentCellEvidence);
        const currentCellNovelty = 1 / (1 + (currentCell.visits ?? 0));
        const currentCellTargetAbsence = clamp(currentCell.targetAbsenceEMA ?? ((currentCell.targetMissCount ?? 0) > 0 ? (currentCell.targetMissCount ?? 0) / 18 : 0), 0, 1);
        const currentCellTargetPenalty = (currentCell.targetHitCount ?? 0) > 0 ? 0 : (currentCellTargetAbsence * 0.65);
        const currentCellWeightScore = (currentCellNovelty * 0.95) + (currentCellOpenBias * 0.4) - (clamp(currentCell.riskEMA ?? 0, 0, 1) * 0.95) - currentCellTargetPenalty;
        const targetColdCount = diagnosticsCandidates.filter((c) => c.targetPenalty > 0.35).length;
        const targetColdRatio = diagnosticsCandidates.length > 0 ? targetColdCount / diagnosticsCandidates.length : 0;

        return {
            schemaVersion: 4,
            gridCellSize: state.cellSize,
            currentCell: {
                ix,
                iz,
                visits: currentCell.visits,
                risk: round(currentCell.riskEMA),
                stuckCount: currentCell.stuckCount,
                targetHitCount: currentCell.targetHitCount,
                targetMissCount: currentCell.targetMissCount ?? 0,
                targetAbsence: round(currentCellTargetAbsence),
                obstacleHits: currentCellObstacleHits,
                outerWallHits: currentCell.outerWallHits ?? 0,
                weightScore: round(currentCellWeightScore),
            },
            loopRate: round(loopRate),
            loopWarning,
            preferredSector,
            sectorScores: sectorScores.map((s) => ({
                sector: s.sector,
                score: round(s.score),
                novelty: round(s.novelty),
                risk: round(s.risk),
                open: round(s.open),
                targetPenalty: round(s.targetPenalty),
            })),
            frontier,
            risky,
            memoryStats: {
                mappedCells: state.cells.size,
                recentPathLength: recentPath.length,
                sensorRange: round(state.sensorRange, 2),
            },
            diagnostics: {
                candidateCount: diagnosticsCandidates.length,
                safeCandidateCount: safeCandidates.length,
                noGoCandidateCount: noGoCount,
                noGoRatio: round(noGoRatio),
                revisitRate: round(loopRate),
                targetColdCount,
                targetColdRatio: round(targetColdRatio),
                topCandidates: sortedCandidates.slice(0, 3).map((c) => toCandidateSummary(c, true)),
                topSafeCandidates: sortedSafeCandidates.slice(0, 3).map((c) => toCandidateSummary(c, false)),
                preferredCandidate: toCandidateSummary(preferredCandidate, true),
                sectorSafety
            }
        };
    };

    const getVisualization = (sensorData, optionsArg = {}) => {
        const x = asNumber(sensorData?.worldX);
        const z = asNumber(sensorData?.worldZ);
        if (x === null || z === null) return null;

        const headingDeg = asNumber(sensorData?.headingDeg, 0);
        const radiusCells = optionsArg.radiusCells ?? 8;
        const maxFrontier = optionsArg.maxFrontier ?? 8;
        const maxRisky = optionsArg.maxRisky ?? 6;

        const { ix, iz } = toGrid(x, z, state.cellSize);
        const context = getContext(sensorData, {
            radiusCells: Math.max(2, Math.floor(radiusCells / 3)),
            maxFrontier,
            maxRisky
        }) || {
            frontier: [],
            risky: [],
            loopRate: 0,
            loopWarning: "LOW",
            preferredSector: "F",
            memoryStats: { mappedCells: state.cells.size, recentPathLength: 0 }
        };

        const frontierKeys = new Set((context.frontier || []).map((c) => cellKey(c.ix, c.iz)));
        const riskyKeys = new Set((context.risky || []).map((c) => cellKey(c.ix, c.iz)));

        const cells = [];
        for (let dz = -radiusCells; dz <= radiusCells; dz += 1) {
            for (let dx = -radiusCells; dx <= radiusCells; dx += 1) {
                const nix = ix + dx;
                const niz = iz + dz;
                const key = cellKey(nix, niz);
                const candidate = evaluateCandidate(x, z, headingDeg, ix, iz, nix, niz);

                cells.push({
                    ix: candidate.ix,
                    iz: candidate.iz,
                    dx: candidate.dx,
                    dz: candidate.dz,
                    visits: candidate.visits,
                    risk: round(candidate.risk),
                    score: round(candidate.score),
                    novelty: round(candidate.novelty),
                    openBias: round(candidate.openBias),
                    targetPenalty: round(candidate.targetPenalty),
                    targetHitCount: candidate.targetHitCount,
                    targetMissCount: candidate.targetMissCount,
                    targetAbsence: round(candidate.targetAbsenceEMA),
                    obstacleHits: candidate.obstacleHits,
                    outerWallHits: candidate.outerWallHits,
                    repeatPenalty: round(candidate.repeatPenalty),
                    isCurrent: nix === ix && niz === iz,
                    isFrontier: frontierKeys.has(key),
                    isRisky: riskyKeys.has(key),
                    hasData: candidate.hasData,
                    barrierBlocked: candidate.barrierBlocked,
                    outsideBounds: candidate.outsideBounds,
                    obstacleDominant: candidate.obstacleDominant
                });
            }
        }

        const recentPath = state.path.slice(-120).map((p) => ({
            dx: p.ix - ix,
            dz: p.iz - iz
        }));

        return {
            center: { ix, iz },
            headingDeg: round(headingDeg, 1),
            cellSize: state.cellSize,
            sensorRange: round(state.sensorRange, 2),
            radiusCells,
            cells,
            frontier: context.frontier || [],
            risky: context.risky || [],
            preferredSector: context.preferredSector || "F",
            loopRate: context.loopRate ?? 0,
            loopWarning: context.loopWarning || "LOW",
            memoryStats: context.memoryStats || { mappedCells: state.cells.size, recentPathLength: recentPath.length },
            diagnostics: context.diagnostics || null,
            recentPath
        };
    };

    const reset = () => {
        state.cells.clear();
        state.path = [];
    };

    const exportCells = () => Array.from(state.cells.entries()).map(([key, cell]) => ({
        ...parseCellKey(key),
        ...cell,
    }));

    return {
        update,
        getContext,
        getVisualization,
        reset,
        exportCells,
    };
}
