import { Canvas } from "@react-three/fiber";
import { Physics, useBox, usePlane } from "@react-three/cannon";
import { OrbitControls, Environment, Sky, Text } from "@react-three/drei";
import Car from "./Car";
import { Suspense } from "react";

function Ground() {
    const [ref] = usePlane(() => ({
        rotation: [-Math.PI / 2, 0, 0],
        type: "Static",
        material: { friction: 0.0 } // Ice Ground
    }));
    return (
        <mesh ref={ref} receiveShadow>
            <planeGeometry args={[100, 100]} />
            <meshStandardMaterial color="#404040" />
        </mesh>
    );
}

function Wall({ position, args, rotation = [0, 0, 0] }) {
    const [ref] = useBox(() => ({ position, args, rotation, type: "Static" }));
    return (
        <mesh ref={ref} castShadow receiveShadow>
            <boxGeometry args={args} />
            <meshStandardMaterial color="#884444" />
        </mesh>
    );
}

function Track() {
    // Simple oval/rectangular track
    return (
        <>
            <Ground />
            {/* Outer Walls */}
            <Wall position={[0, 1, -20]} args={[40, 2, 1]} /> {/* North */}
            <Wall position={[0, 1, 20]} args={[40, 2, 1]} />  {/* South */}
            <Wall position={[-20, 1, 0]} args={[1, 2, 40]} /> {/* West */}
            <Wall position={[20, 1, 0]} args={[1, 2, 40]} />  {/* East */}

            {/* Inner Obstacles - Testing Collision Avoidance */}

            {/* 1. The Monolith (Center Block) */}
            <Wall position={[0, 1, 0]} args={[10, 2, 2]} />

            {/* 2. Scattered Pillars */}
            <Wall position={[-10, 1, -10]} args={[2, 2, 2]} />
            <Wall position={[10, 1, 10]} args={[2, 2, 2]} />
            <Wall position={[-10, 1, 10]} args={[2, 2, 2]} />
            <Wall position={[10, 1, -10]} args={[2, 2, 2]} />
        </>
    );
}

function CompassMarkers() {
    return (
        <group>
            <Text position={[0, 3, -23]} fontSize={1.1} color="#22d3ee" anchorX="center" anchorY="middle" billboard>
                N (-Z)
            </Text>
            <Text position={[0, 3, 23]} fontSize={1.1} color="#22d3ee" anchorX="center" anchorY="middle" billboard>
                S (+Z)
            </Text>
            <Text position={[23, 3, 0]} fontSize={1.1} color="#22d3ee" anchorX="center" anchorY="middle" billboard>
                E (+X)
            </Text>
            <Text position={[-23, 3, 0]} fontSize={1.1} color="#22d3ee" anchorX="center" anchorY="middle" billboard>
                W (-X)
            </Text>
        </group>
    );
}

export default function GameScene({
    onSensorUpdate,
    onCollisionEvent,
    lastAction,
    targetPosition,
    controls,
    carResetNonce = 0,
    carSpawnPosition = [0, 0.65, -10],
    carSpawnRotation = [0, 0, 0],
    physicsSettings,
    worldBounds
}) {
    console.log("GameScene Render:", controls);
    return (
        <div className="w-full h-full bg-slate-900">
            <Canvas shadows camera={{ position: [0, 30, 30], fov: 60 }}>
                <fog attach="fog" args={["#202020", 30, 250]} />
                <color attach="background" args={["#202020"]} />
                <ambientLight intensity={1.5} />
                <directionalLight
                    position={[10, 50, 20]}
                    intensity={2.5}
                    castShadow
                />
                <Sky sunPosition={[100, 10, 100]} />

                <Physics gravity={[0, -9.8, 0]}>
                    <Track />
                    <Car
                        key={`car-reset-${carResetNonce}`}
                        onSensorUpdate={onSensorUpdate}
                        onCollisionEvent={onCollisionEvent}
                        lastAction={lastAction} // Keeping logic for resetting stuck timer? Or maybe remove later.
                        targetPosition={targetPosition}
                        controls={controls}
                        spawnPosition={carSpawnPosition}
                        spawnRotation={carSpawnRotation}
                        physicsSettings={physicsSettings}
                        worldBounds={worldBounds}
                    />
                    {/* Visual Target */}
                    <mesh position={targetPosition} userData={{ isTarget: true }}>
                        <cylinderGeometry args={[0.5, 0.5, 4, 32]} />
                        <meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={2} />
                    </mesh>
                    <pointLight position={[targetPosition[0], 2, targetPosition[2]]} intensity={2} color="#00ffff" distance={10} />
                </Physics>
                <CompassMarkers />

                <OrbitControls />
            </Canvas>
        </div>
    );
}
