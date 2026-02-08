import { useBox } from "@react-three/cannon";
import { useFrame } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Text } from "@react-three/drei";

const clampValue = (value, min, max) => Math.max(min, Math.min(max, value));

export default function Car({
    onSensorUpdate,
    onCollisionEvent,
    lastAction,
    targetPosition,
    spawnPosition = [0, 0.65, -10],
    spawnRotation = [0, 0, 0],
    controls = { throttle: "IDLE", steering: "IDLE" },
    physicsSettings,
    worldBounds = null
}) {
    // console.log("Car Component Rendered"); // Trace 3: Is it alive?

    // Debug: Check if props are arriving
    useEffect(() => {
        if (controls.throttle !== "IDLE") console.log("Car Received Controls:", controls);
    }, [controls]);

    // Debug: Check if props are arriving
    useEffect(() => {
        if (lastAction !== "IDLE") console.log("Car Action:", lastAction);
    }, [lastAction]);

    // Config
    const SPEED_FORCE = physicsSettings?.speedForce ?? 8000;
    const TURN_TORQUE = physicsSettings?.turnTorque ?? 200;
    const MAX_SPEED = 25;

    // Crash State
    const crashCooldown = useRef(0);
    const collisionEventCooldown = useRef(0);
    const apiRef = useRef(); // Ref to hold API to avoid temporal dead zone in onCollide
    const classifyCollisionRegion = (x, z) => {
        if (!worldBounds || !Number.isFinite(x) || !Number.isFinite(z)) return "UNKNOWN";
        const edgeBand = 2.6;
        const distWest = Math.abs(x - worldBounds.minX);
        const distEast = Math.abs(worldBounds.maxX - x);
        const distNorth = Math.abs(z - worldBounds.minZ);
        const distSouth = Math.abs(worldBounds.maxZ - z);
        const minEdge = Math.min(distWest, distEast, distNorth, distSouth);
        if (x < worldBounds.minX || x > worldBounds.maxX || z < worldBounds.minZ || z > worldBounds.maxZ) {
            return "OUTSIDE_BOUNDS";
        }
        if (minEdge > edgeBand) return "INNER_OBSTACLE";
        if (minEdge === distWest) return "OUTER_WEST";
        if (minEdge === distEast) return "OUTER_EAST";
        if (minEdge === distNorth) return "OUTER_NORTH";
        return "OUTER_SOUTH";
    };

    // Physics Body: "Ice Box" (Sliding, non-flipping)
    const [ref, api] = useBox(() => ({
        mass: 200,
        position: spawnPosition,
        rotation: spawnRotation,
        args: [2, 1, 4],
        linearDamping: 0.95, // ARCADE: High damping = "Tires" (Stops when force stops)
        angularDamping: 0.95, // ARCADE: Stops spinning quickly
        angularFactor: [0, 1, 0], // CRITICAL: Only allow Y rotation (Turn), lock X/Z (Flips)
        material: { friction: 0.5, restitution: 0.0 }, // Grip, no bounce
        onCollide: (e) => {
            // DETECT IMPACT (Realistic Collision)
            const speed = Math.abs(e.contact.impactVelocity);
            const now = Date.now();
            if (speed > 0.9 && now > collisionEventCooldown.current) {
                const posX = Number.isFinite(position.current?.[0]) ? position.current[0] : null;
                const posZ = Number.isFinite(position.current?.[2]) ? position.current[2] : null;
                const region = classifyCollisionRegion(posX, posZ);
                if (typeof onCollisionEvent === "function") {
                    onCollisionEvent({
                        time: new Date(now).toISOString(),
                        impactVelocity: Number(speed.toFixed(3)),
                        region,
                        worldX: posX === null ? null : Number(posX.toFixed(3)),
                        worldZ: posZ === null ? null : Number(posZ.toFixed(3)),
                        otherBodyId: Number.isFinite(e?.body?.id) ? e.body.id : null
                    });
                }
                collisionEventCooldown.current = now + 250;
            }
            if (speed > 2.0 && Date.now() > crashCooldown.current) {
                console.log("💥 CRASH DETECTED! Speed:", speed.toFixed(1));

                // Kill Velocity Instantly (Stop pushing)
                if (apiRef.current) {
                    apiRef.current.velocity.set(0, 0, 0);
                }

                // Set Cooldown (Stall Engine for 1s)
                crashCooldown.current = Date.now() + 1000;
            }
        },
        allowSleep: false
    }));

    // Store api in ref to avoid temporal dead zone
    useEffect(() => {
        apiRef.current = api;
    }, [api]);

    // Sensors & State
    const [sensorLines, setSensorLines] = useState([]);

    // Physics Synced State
    const velocity = useRef([0, 0, 0]);
    useEffect(() => api.velocity.subscribe((v) => (velocity.current = v)), [api.velocity]);

    const position = useRef([0, 0, 0]);
    useEffect(() => api.position.subscribe((p) => (position.current = p)), [api.position]);

    const quaternion = useRef([0, 0, 0, 1]);
    useEffect(() => api.quaternion.subscribe((q) => (quaternion.current = q)), [api.quaternion]);

    const raycaster = useRef(new THREE.Raycaster());
    const scene = useRef(null);
    const frameCounter = useRef(0); // Frame throttling counter
    const readingsRef = useRef({ front: 10, leftDiag: 10, rightDiag: 10 }); // LATEST READINGS for Safety
    const sensorRangeRef = useRef(10);


    useFrame((state) => {
        if (!scene.current) scene.current = state.scene;
        if (!ref.current) return;

        // --- Crash Check ---
        const isStalled = Date.now() < crashCooldown.current;

        // --- Apply Controls ---
        // 1. Get Current Rotation from Physics Engine (Most Accurate)
        const currentQuat = new THREE.Quaternion(...quaternion.current);

        // Calculate Forward Vector (Local -Z)
        // Calculate Forward Vector (Local +Z)
        const localForward = new THREE.Vector3(0, 0, 1).applyQuaternion(currentQuat);

        // --- ARCADE PHYSICS (Direct Velocity - Single Channel) ---

        // 1. Calculate Target Velocities
        // GRAVITY ASSIST: Prevent flying. If going up (>1), slam it down.
        let currentVy = velocity.current[1];
        if (currentVy > 0.5) currentVy = 0.5; // Cap upward velocity
        if (position.current[1] > 2.5) currentVy -= 0.5; // Apply extra gravity if off ground

        let targetVx = 0;
        let targetVz = 0;
        let targetRotY = 0;

        const MOVESPEED = (physicsSettings?.speedForce ?? 8000) * 0.002;
        const TURNSPEED = (physicsSettings?.turnTorque ?? 200) * 0.01;

        // --- ANALOG PHYSICS ---
        // If Stalled (Crashed), Input is effectively 0
        const throttleInput = isStalled ? 0 : (controls?.throttle || 0);
        const steeringInput = controls?.steering || 0; // Steering might still work? Or lock it too? Let's allow steering.

        // Forward/Reverse
        if (Math.abs(throttleInput) > 0.05) {
            targetVx = localForward.x * MOVESPEED * throttleInput;
            targetVz = localForward.z * MOVESPEED * throttleInput;
        } else {
            // Friction/Drag when idle (Strong braking)
            targetVx = velocity.current[0] * 0.80;
            targetVz = velocity.current[2] * 0.80;
        }

        // Turning (Allow turning while moving, or even pivot turning)
        // Physics tweak: Turning is easier when moving? 
        // For Arcade fun, we allow pivot turning but maybe slightly slower.
        targetRotY = steeringInput * TURNSPEED;

        // Fix ReferenceError: Call velocity check early
        const currentSpeed = new THREE.Vector3(...velocity.current).length();

        // DEBUG: Log every 60 frames (~1 sec)
        if (state.clock.elapsedTime % 1 < 0.05) {
            console.log("PHYSICS:", {
                throttle: throttleInput.toFixed(2),
                crashed: isStalled,
                speed: currentSpeed.toFixed(2)
            });
        }

        // Apply Physics
        api.velocity.set(targetVx, currentVy, targetVz); // Directly setting velocity (Arcade Style)
        api.angularVelocity.set(0, targetRotY, 0);       // Directly setting angular velocity

        // Debug
        // if (lastAction !== "IDLE") console.log("Action:", lastAction);

        // Throttling: Only run Raycast & Update every 0.1s (10Hz)
        // CRITICAL FIX: Raycast MUST run every frame for safety, or at least very high freq.
        // We removed the return logic to notify app, but we need local raycast for safety.
        // For performance, we can skip App update but NOT the safety check.
        // redesign: Run raycast every frame? Or every 2 frames?
        // Let's run every 3 frames (~20fps) for safety.
        frameCounter.current += 1;

        // --- REFLEXIVE SAFETY STOP MOVED TO BOTTOM (Needs Drift Logic) ---
        // We calculate Drift/Movement Direction at end of loop, then apply safety.

        if (frameCounter.current % 3 !== 0) return;

        // --- Sensors Calc (Dynamic 8-Ray Lidar) ---
        const sensorRangeMin = Math.max(4.0, physicsSettings?.sensorRangeMin ?? 7.0);
        const sensorRangeMax = Math.max(sensorRangeMin + 0.5, physicsSettings?.sensorRangeMax ?? 14.0);
        const sensorDynamic = physicsSettings?.sensorDynamic !== false;
        const speedNorm = clampValue(currentSpeed / MAX_SPEED, 0, 1);
        const steeringMag = Math.abs(controls?.steering || 0);
        const movingForward = throttleInput > 0.12;
        const movingBackward = throttleInput < -0.12;
        const stuckRisk = Math.abs(throttleInput) > 0.2 && currentSpeed < 0.8;
        const rangeSpan = sensorRangeMax - sensorRangeMin;
        let desiredSensorRange = sensorRangeMin + (rangeSpan * (0.25 + (speedNorm * 0.55)));

        if (movingForward) desiredSensorRange += rangeSpan * (0.22 * (1 - steeringMag));
        if (stuckRisk) desiredSensorRange += rangeSpan * 0.2;
        if (steeringMag > 0.45) desiredSensorRange -= rangeSpan * 0.18;
        if (movingBackward) desiredSensorRange -= rangeSpan * 0.12;

        desiredSensorRange = clampValue(desiredSensorRange, sensorRangeMin, sensorRangeMax);
        if (!sensorDynamic) {
            desiredSensorRange = (sensorRangeMin + sensorRangeMax) * 0.5;
        }

        const smoothedSensorRange = sensorRangeRef.current + ((desiredSensorRange - sensorRangeRef.current) * 0.2);
        const SENSOR_LENGTH = clampValue(smoothedSensorRange, sensorRangeMin, sensorRangeMax);
        sensorRangeRef.current = SENSOR_LENGTH;
        const currentQ = new THREE.Quaternion(...quaternion.current);

        // Helper to get local vector rotated by car
        const getDir = (x, z) => new THREE.Vector3(x, 0, z).applyQuaternion(currentQ).normalize();

        const directions = [
            { name: "left", vec: getDir(1.0, 0.0) }, // Left is now +X (facing +Z)
            { name: "leftDiag", vec: getDir(0.5, 0.5) },
            { name: "front", vec: getDir(0.0, 1.0) }, // Front is +Z
            { name: "rightDiag", vec: getDir(-0.5, 0.5) },
            { name: "right", vec: getDir(-1.0, 0.0) },  // Right is -X
            { name: "backRight", vec: getDir(-0.5, -0.5) },
            { name: "back", vec: getDir(0.0, -1.0) },  // Back is -Z
            { name: "backLeft", vec: getDir(0.5, -0.5) }
        ];

        let readings = {
            left: SENSOR_LENGTH, leftDiag: SENSOR_LENGTH, front: SENSOR_LENGTH, rightDiag: SENSOR_LENGTH, right: SENSOR_LENGTH,
            backRight: SENSOR_LENGTH, back: SENSOR_LENGTH, backLeft: SENSOR_LENGTH
        };
        // TRACK TARGET HITS
        let targetHits = {
            left: false, leftDiag: false, front: false, rightDiag: false, right: false,
            backRight: false, back: false, backLeft: false
        };

        const visualLines = [];
        const obstacles = [];
        if (scene.current) {
            scene.current.traverse(obj => {
                if (obj.isMesh) {
                    // ALLOW Target in Raycaster for Visuals (Cyan Line)
                    // We will filter it out of the *Logic* later so AI doesn't stop.
                    // if (obj.userData?.isTarget) return;

                    // Robust Check: Is this object part of the Car?
                    let isCarPart = false;
                    let parent = obj;
                    while (parent) {
                        if (parent.uuid === ref.current.uuid) { isCarPart = true; break; }
                        parent = parent.parent;
                    }
                    if (!isCarPart) obstacles.push(obj);
                }
            });
        }

        const originWorld = new THREE.Vector3(...position.current);
        originWorld.y += 0.5; // ROOF MOUNT (avoid body)

        // Visual Lines are children of the Car Mesh, so they need LOCAL coordinates.
        // Origin Local: (0, 0.5, 0)
        // We do NOT rotate visual lines because the parent mesh rotates.
        const originLocal = new THREE.Vector3(0, 0.5, 0);

        directions.forEach(dir => {
            // Raycast needs WORLD coordinates
            raycaster.current.set(originWorld, dir.vec);
            const intersects = raycaster.current.intersectObjects(obstacles);

            let visualDist = SENSOR_LENGTH;
            let logicalDist = SENSOR_LENGTH; // What the AI/Reflex sees
            let color = "green";

            if (intersects.length > 0 && intersects[0].distance < SENSOR_LENGTH) {
                const hit = intersects[0];
                const isTarget = hit.object.userData?.isTarget || hit.object.parent?.userData?.isTarget;

                visualDist = hit.distance;

                if (isTarget) {
                    // TARGET HIT: Visual = Show Hit, Logical = Ignore (Don't Stop)
                    color = "cyan"; // Special Target Color
                    logicalDist = SENSOR_LENGTH; // AI sees "Clear" for SAFETY system
                    targetHits[dir.name] = true; // REPORT HIT for NAVIGATION system
                } else {
                    // WALL HIT: Visual = Warning, Logical = Obstacle
                    logicalDist = hit.distance; // AI sees Wall

                    if (visualDist < 4) color = "red";
                    else if (visualDist < 7) color = "orange";
                }
            }

            readings[dir.name] = logicalDist;

            // Visuals need LOCAL coordinates
            // We use the 'dir.localVec' which is unrotated (relative to car)
            let localDirVec = new THREE.Vector3();
            if (dir.name === "left") localDirVec.set(1, 0, 0);
            else if (dir.name === "leftDiag") localDirVec.set(0.5, 0, 0.5).normalize();
            else if (dir.name === "front") localDirVec.set(0, 0, 1);
            else if (dir.name === "rightDiag") localDirVec.set(-0.5, 0, 0.5).normalize();
            else if (dir.name === "right") localDirVec.set(-1, 0, 0);
            else if (dir.name === "backRight") localDirVec.set(-0.5, 0, -0.5).normalize();
            else if (dir.name === "back") localDirVec.set(0, 0, -1);
            else if (dir.name === "backLeft") localDirVec.set(0.5, 0, -0.5).normalize();

            const endLocal = originLocal.clone().add(localDirVec.multiplyScalar(visualDist));

            visualLines.push({ start: originLocal, end: endLocal, color, name: dir.name });
        });

        setSensorLines(visualLines);

        // UPDATE REF for SAFETY CHECK (So it persists between frames)
        readingsRef.current = {
            front: readings.front,
            leftDiag: readings.leftDiag,
            rightDiag: readings.rightDiag
        };

        // Target Logic
        let angleToTarget = 0;
        let distanceToTarget = 0;
        if (targetPosition) {
            const currentPos = new THREE.Vector3(...position.current);
            const targetPos = new THREE.Vector3(...targetPosition);
            const toTargetFlat = new THREE.Vector3(targetPos.x - currentPos.x, 0, targetPos.z - currentPos.z);
            distanceToTarget = toTargetFlat.length();

            const forward2D = new THREE.Vector3(localForward.x, 0, localForward.z).normalize();
            const target2D = distanceToTarget > 1e-6
                ? toTargetFlat.clone().normalize()
                : forward2D.clone();

            const dot = forward2D.dot(target2D);
            const cross = new THREE.Vector3().crossVectors(forward2D, target2D);
            angleToTarget = THREE.MathUtils.radToDeg(Math.atan2(cross.y, dot));
        }

        // Movement Direction Analysis
        const velVec = new THREE.Vector3(...velocity.current);
        const velMag = velVec.length();
        let moveDir = "IDLE";
        let blockedDist = 99;

        if (velMag > 0.5) {
            const velNorm = velVec.normalize();
            const dotFwd = localForward.dot(velNorm);

            if (dotFwd > 0.7) {
                moveDir = "FORWARD";
                blockedDist = readingsRef.current.front;
            } else if (dotFwd < -0.7) {
                moveDir = "BACKWARD";
                blockedDist = readings.back;
            } else {
                moveDir = "DRIFT";
                blockedDist = Math.min(readings.left, readings.right);
            }
        }

        // Stuck Detection
        const isMovingInput = Math.abs(controls?.throttle || 0) > 0.1;
        const isStuck = isMovingInput && currentSpeed < 0.5;
        const headingDeg = THREE.MathUtils.radToDeg(Math.atan2(localForward.x, localForward.z));

        // Update App with Sensor Data
        onSensorUpdate({
            ...readings,
            sensorRange: SENSOR_LENGTH,
            worldX: position.current[0],
            worldY: position.current[1],
            worldZ: position.current[2],
            headingDeg,
            targetHits,
            angleToTarget,
            distanceToTarget,
            speed: currentSpeed,
            verticalSpeed: velocity.current[1],
            grounded: position.current[1] <= 1.2,
            isStuck,
            moveDir,
            blockedDist
        });
    });

    return (
        <mesh ref={ref} name="car-collider" castShadow>
            {/* Invisible Box Collider Wireframe for Debug */}
            {/* <boxGeometry args={[2, 1, 4]} /> */}
            {/* <meshBasicMaterial transparent opacity={0.3} wireframe color="cyan" /> */}

            {/* Car Visual Mesh - Centered in the physics box */}
            <group position={[0, -0.2, 0]}>
                {/* Main Body */}
                <mesh castShadow receiveShadow position={[0, 0.25, 0]}>
                    <boxGeometry args={[1.8, 0.8, 3.8]} />
                    <meshStandardMaterial color="orange" metalness={0.6} roughness={0.2} />
                </mesh>
                {/* Cabin */}
                <mesh position={[0, 0.8, -0.5]} castShadow>
                    <boxGeometry args={[1.4, 0.6, 1.8]} />
                    <meshStandardMaterial color="cyan" metalness={0.9} roughness={0.1} />
                </mesh>

                {/* DEBUG: Heading Arrow (Red = Forward) */}
                <arrowHelper args={[new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0), 4, 0xff0000]} />

                {/* Orientation Labels */}
                <Text position={[0, 1.5, 2.2]} fontSize={0.5} color="yellow" billboard>FRONT</Text>
                <Text position={[0, 1.5, -2.2]} fontSize={0.5} color="red" billboard>REAR</Text>
            </group>

            {/* Visual Sensor Lines */}
            {sensorLines.map((line, i) => (
                <group key={i}>
                    <line>
                        <bufferGeometry>
                            <float32BufferAttribute
                                attach="attributes-position"
                                args={[new Float32Array([line.start.x, line.start.y, line.start.z, line.end.x, line.end.y, line.end.z]), 3]}
                            />
                        </bufferGeometry>
                        <lineBasicMaterial color={line.color} />
                    </line>
                    <Text
                        position={[line.end.x, line.end.y + 0.5, line.end.z]}
                        fontSize={0.3}
                        color="white"
                        anchorX="center"
                        anchorY="middle"
                        billboard
                    >
                        {line.name}
                    </Text>
                </group>
            ))}
        </mesh>
    );
}
