
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Sphere, Stars } from '@react-three/drei';
import * as THREE from 'three';

// Represents the "Common Source" (Patient/Latent Space z)
const CentralSource = () => {
  const meshRef = useRef<THREE.Mesh>(null);
  const coreRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current && coreRef.current) {
      const t = state.clock.getElapsedTime();
      // Pulsating effect representing the "beating heart" of the data source
      const scale = 1 + Math.sin(t * 1.5) * 0.05;
      meshRef.current.scale.set(scale, scale, scale);
      meshRef.current.rotation.y = t * 0.2;
      meshRef.current.rotation.z = t * 0.1;
      
      // Core pulse
      const coreScale = 0.6 + Math.sin(t * 3) * 0.1;
      coreRef.current.scale.set(coreScale, coreScale, coreScale);
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.2} floatIntensity={0.2}>
      {/* Outer geometric shell */}
      <Sphere ref={meshRef} args={[1.2, 16, 16]} position={[0, 0, 0]}>
        <meshStandardMaterial
          color="#3B82F6"
          emissive="#1D4ED8"
          emissiveIntensity={0.5}
          wireframe
          transparent
          opacity={0.3}
        />
      </Sphere>
       {/* Inner glowing core */}
       <Sphere ref={coreRef} args={[0.8, 32, 32]}>
        <meshStandardMaterial 
            color="#10B981"
            emissive="#10B981"
            emissiveIntensity={2}
            roughness={0.2}
        />
      </Sphere>
    </Float>
  );
};

// Represents a single Time Series Channel generated from the source
const ChannelStream = ({ 
    index, 
    total, 
    color 
}: { 
    index: number, 
    total: number, 
    color: string 
}) => {
    const lineRef = useRef<any>(null);
    const particleRef = useRef<THREE.Mesh>(null);
    
    // Create initial points for the line
    const points = useMemo(() => {
        return new Array(100).fill(0).map((_, i) => new THREE.Vector3(i * 0.1, 0, 0));
    }, []);

    useFrame((state) => {
        if (lineRef.current && particleRef.current) {
            const t = state.clock.getElapsedTime();
            const positions = lineRef.current.geometry.attributes.position.array;
            
            // Angle for this channel around the source
            const angleStep = (Math.PI * 2) / total;
            const angle = index * angleStep + t * 0.1; // Slowly rotate entire system
            
            // The wave originates from center (0,0,0) and flows outward
            // We map the linear points to a curve extending from center
            for (let i = 0; i < 100; i++) {
                const dist = 1.5 + (i * 0.15); // Distance from center
                
                // Wave function: Unique frequency/phase per channel, but synchronized by 't'
                // This represents the "Common Source" correlation
                const wave = Math.sin(dist * 0.5 - t * 3 + index); 
                const verticalWave = Math.cos(dist * 0.3 - t * 2 + index * 2);

                // Convert polar to cartesian
                const x = Math.cos(angle) * dist;
                const z = Math.sin(angle) * dist;
                const y = wave * 0.5 + verticalWave * 0.2;

                positions[i * 3] = x;
                positions[i * 3 + 1] = y;
                positions[i * 3 + 2] = z;

                // Place particle at the leading edge (end of line)
                if (i === 99) {
                    particleRef.current.position.set(x, y, z);
                }
            }
            lineRef.current.geometry.attributes.position.needsUpdate = true;
        }
    });

    return (
        <group>
            <line ref={lineRef}>
                <bufferGeometry>
                    <bufferAttribute 
                        attach="attributes-position" 
                        count={points.length} 
                        array={new Float32Array(points.length * 3)} 
                        itemSize={3} 
                    />
                </bufferGeometry>
                <lineBasicMaterial color={color} transparent opacity={0.6} linewidth={2} />
            </line>
            <Sphere ref={particleRef} args={[0.08, 8, 8]}>
                <meshBasicMaterial color="white" />
            </Sphere>
        </group>
    )
}

export const SourceScene: React.FC = () => {
  const channels = 6;
  const colors = ['#3B82F6', '#10B981', '#8B5CF6', '#06B6D4', '#3B82F6', '#10B981'];

  return (
    <div className="absolute inset-0 z-0 opacity-100 pointer-events-none">
      <Canvas camera={{ position: [0, 4, 12], fov: 45 }}>
        <color attach="background" args={['#0F172A']} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#3B82F6" />
        
        <CentralSource />

        {/* Generating Multiple Correlated Channels */}
        {Array.from({ length: channels }).map((_, i) => (
            <ChannelStream key={i} index={i} total={channels} color={colors[i]} />
        ))}

        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
      </Canvas>
    </div>
  );
};
