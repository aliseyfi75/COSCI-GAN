
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

// --- ARCHITECTURE DIAGRAM ---
export const ArchitectureDiagram: React.FC = () => {
  return (
    <div className="w-full max-w-5xl mx-auto p-4 select-none">
       <div className="relative bg-slate-900/90 backdrop-blur-md rounded-xl border border-slate-700 h-[850px] overflow-hidden shadow-2xl">
          <div className="absolute top-4 left-4 text-slate-400 text-xs font-mono z-20">
            FIG 1. COSCI-GAN ARCHITECTURE
          </div>
          <DiagramAnimation />
       </div>
    </div>
  );
};

const DiagramAnimation = () => {
    // SVG ViewBox: 0 0 800 850
    const width = 800;
    const height = 850;

    // Coordinates
    const zPos = { x: 400, y: 60 };
    
    // Generators (Row 1)
    const gPos = [
        { x: 150, y: 220, label: 'Generator 1', type: 'sine', color: '#3B82F6' },
        { x: 400, y: 220, label: 'Generator 2', type: 'square', color: '#F59E0B' },
        { x: 650, y: 220, label: 'Generator 3', type: 'sawtooth', color: '#EC4899' },
    ];

    // Channel Discriminators (Row 2 - Below Generators)
    const dPos = [
        { x: 150, y: 450, label: 'Discriminator 1' },
        { x: 400, y: 450, label: 'Discriminator 2' },
        { x: 650, y: 450, label: 'Discriminator 3' },
    ];

    // Central Discriminator (Row 3 - Bottom Center)
    // Positioned to receive converging flows directly from generators
    const cdPos = { x: 400, y: 680 };

    // Animation State
    // 0: Noise -> Generators
    // 1: Generators Active -> Packets travel to Channel D & Central D
    // 2: Signals arrive in D & Central D (Show Signals - Distorted immediately if Fake)
    // 3: Decisions (Color change labels)
    const [stage, setStage] = useState(0);
    
    // Random decisions for the current cycle (true = real/green, false = fake/red)
    const [decisions, setDecisions] = useState({
        d1: true,
        d2: true,
        d3: true,
        cd: true
    });

    useEffect(() => {
        const interval = setInterval(() => {
            setStage(prev => {
                if (prev === 3) {
                    // Reset decisions for next loop at the start of the cycle
                    setDecisions({
                        d1: Math.random() > 0.4,
                        d2: Math.random() > 0.4,
                        d3: Math.random() > 0.4,
                        cd: Math.random() > 0.3
                    });
                    return 0;
                }
                return prev + 1;
            });
        }, 2000); 
        return () => clearInterval(interval);
    }, []);

    const getDecisionColor = (isReal: boolean) => isReal ? "#10B981" : "#EF4444"; // Green vs Red

    // Wave shapes for moving packets (width ~20)
    const packetPathD = {
        sine: "M 0 0 Q 5 -10 10 0 T 20 0",
        square: "M 0 0 L 5 0 L 5 -10 L 15 -10 L 15 0 L 20 0",
        sawtooth: "M 0 0 L 10 -10 L 10 0 L 20 0"
    };

    return (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#475569" />
                </marker>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>

            {/* --- LAYER 1: CONNECTIONS (Back) --- */}
            
            {/* 1. Z to Generators (Dotted) */}
            {gPos.map((g, i) => (
                <path 
                    key={`line-z-g${i}`}
                    d={`M ${zPos.x} ${zPos.y + 35} C ${zPos.x} ${zPos.y + 100}, ${g.x} ${g.y - 100}, ${g.x} ${g.y - 40}`}
                    fill="none"
                    stroke="#334155"
                    strokeWidth="2"
                    strokeDasharray="6,6"
                />
            ))}

            {/* 2a. Generators to Channel Discriminators (Straight Down) */}
            {gPos.map((g, i) => (
                <line 
                    key={`line-g-d${i}`}
                    x1={g.x} y1={g.y + 40} 
                    x2={dPos[i].x} y2={dPos[i].y - 35} 
                    stroke="#475569" 
                    strokeWidth="2"
                    markerEnd="url(#arrowhead)"
                />
            ))}

            {/* 2b. Generators to Central Discriminator (Curved Converging) */}
            {gPos.map((g, i) => (
                <path 
                    key={`line-g-cd${i}`}
                    d={`M ${g.x} ${g.y + 40} C ${g.x} ${g.y + 250}, ${cdPos.x + (i-1)*60} ${cdPos.y - 150}, ${cdPos.x + (i-1)*20} ${cdPos.y - 70}`}
                    fill="none"
                    stroke="#334155"
                    strokeWidth="1"
                    opacity="0.6"
                    markerEnd="url(#arrowhead)"
                />
            ))}


            {/* --- LAYER 2: MOVING PACKETS (Middle) --- */}

            {/* Stage 0: Noise Particles */}
            {gPos.map((g, i) => (
                <motion.circle 
                    key={`p-z-g${i}`}
                    r="6" 
                    fill="#fff"
                    initial={{ offsetDistance: "0%", opacity: 0 }}
                    animate={{ 
                        offsetDistance: "100%",
                        opacity: stage === 0 ? [0, 1, 1, 0] : 0
                    }}
                    transition={{ 
                        duration: 2, 
                        ease: "linear",
                        repeat: stage === 0 ? Infinity : 0
                    }}
                    style={{ 
                        offsetPath: `path("M ${zPos.x} ${zPos.y + 35} C ${zPos.x} ${zPos.y + 100}, ${g.x} ${g.y - 100}, ${g.x} ${g.y - 40}")` 
                    }}
                />
            ))}

            {/* Stage 1: Waveforms to Channel Ds */}
            {gPos.map((g, i) => (
                <motion.path
                    key={`p-g-d${i}`}
                    d={packetPathD[g.type as keyof typeof packetPathD]}
                    fill="none"
                    stroke={g.color}
                    strokeWidth="3"
                    initial={{ offsetDistance: "0%", opacity: 0 }}
                    animate={{ 
                        offsetDistance: "100%",
                        opacity: stage === 1 ? [0, 1, 1, 0] : 0
                    }}
                    transition={{ duration: 2, ease: "linear", repeat: stage === 1 ? Infinity : 0 }}
                    style={{
                        offsetPath: `path("M ${g.x} ${g.y + 40} L ${dPos[i].x} ${dPos[i].y - 35}")`,
                        offsetRotate: "auto"
                    }}
                />
            ))}

            {/* Stage 1: Waveforms to Central D (Converging) */}
            {gPos.map((g, i) => (
                <motion.path
                    key={`p-g-cd${i}`}
                    d={packetPathD[g.type as keyof typeof packetPathD]}
                    fill="none"
                    stroke={g.color}
                    strokeWidth="3"
                    initial={{ offsetDistance: "0%", opacity: 0 }}
                    animate={{ 
                        offsetDistance: "100%",
                        opacity: stage === 1 ? [0, 1, 1, 0] : 0
                    }}
                    transition={{ duration: 2, ease: "linear", repeat: stage === 1 ? Infinity : 0 }}
                    style={{
                        offsetPath: `path("M ${g.x} ${g.y + 40} C ${g.x} ${g.y + 250}, ${cdPos.x + (i-1)*60} ${cdPos.y - 150}, ${cdPos.x + (i-1)*20} ${cdPos.y - 70}")`,
                        offsetRotate: "auto"
                    }}
                />
            ))}


            {/* --- LAYER 3: NODES & TEXT (Front) --- */}

            {/* Common Source Z */}
            <g transform={`translate(${zPos.x}, ${zPos.y})`}>
                <motion.circle 
                    r="35" 
                    fill="#1E293B" 
                    stroke="#E2E8F0" 
                    strokeWidth="3"
                    animate={{ 
                        scale: stage === 0 ? [1, 1.1, 1] : 1,
                        stroke: stage === 0 ? "#3B82F6" : "#E2E8F0"
                    }}
                />
                <text x="0" y="8" textAnchor="middle" fill="white" className="text-2xl font-bold font-mono">z</text>
                <text x="0" y="-45" textAnchor="middle" fill="#94A3B8" className="text-xs uppercase tracking-wider font-bold bg-slate-900/80 px-2 rounded">Common Noise</text>
            </g>

            {/* Generators */}
            {gPos.map((g, i) => (
                <g key={`node-g${i}`} transform={`translate(${g.x}, ${g.y})`}>
                    <motion.rect 
                        x="-40" y="-40" width="80" height="80" rx="8"
                        fill="#1E293B"
                        stroke={g.color}
                        strokeWidth="2"
                        animate={{
                            scale: stage === 1 ? 1.1 : 1,
                            fill: stage === 1 ? "#0F172A" : "#1E293B",
                            boxShadow: stage === 1 ? `0 0 15px ${g.color}` : "none"
                        }}
                    />
                    <text x="0" y="55" textAnchor="middle" fill={g.color} className="text-xs font-bold uppercase">{g.label}</text>
                    {/* Wave icon inside box */}
                    <path d={getWaveIcon(g.type)} fill="none" stroke={g.color} strokeWidth="3" transform="translate(-20, 5) scale(1)" />
                </g>
            ))}

            {/* Channel Discriminators */}
            {dPos.map((d, i) => {
                const isReal = i === 0 ? decisions.d1 : i === 1 ? decisions.d2 : decisions.d3;
                const gen = gPos[i];
                const showSignal = stage >= 2; 
                // NOTE: If it's Fake, we show distorted signal from the start of Stage 2
                // The fake signal is determined per channel discriminator decision
                const isDistorted = !isReal;

                return (
                    <g key={`node-d${i}`} transform={`translate(${d.x}, ${d.y})`}>
                        <motion.rect 
                            x="-35" y="-35" width="70" height="70" rx="8"
                            fill="#1E293B"
                            stroke={stage >= 2 ? "#64748B" : "#475569"}
                            strokeWidth="2"
                            animate={{
                                stroke: stage === 3 ? getDecisionColor(isReal) : "#64748B",
                                scale: stage === 2 ? 1.05 : 1
                            }}
                        />
                        <text x="0" y="55" textAnchor="middle" fill="#94A3B8" className="text-[10px] font-bold uppercase tracking-tighter">{d.label}</text>
                        
                        {/* Signal Visualization inside Discriminator */}
                        {/* Using exact same generator function as packet but scaled */}
                        <motion.path 
                            d={isDistorted ? getDistortedWaveIcon(gen.type, 40) : getWaveIcon(gen.type, 40)} 
                            fill="none" 
                            stroke={gen.color} 
                            strokeWidth="2" 
                            transform="translate(-20, -10) scale(1)"
                            initial={{ opacity: 0, pathLength: 0 }}
                            animate={{ 
                                opacity: showSignal ? 1 : 0,
                                pathLength: showSignal ? 1 : 0,
                            }}
                            transition={{ duration: 0.5 }}
                        />

                        {/* Decision Label */}
                        <motion.g opacity={stage === 3 ? 1 : 0}>
                             <rect x="-25" y="-25" width="50" height="20" rx="4" fill={getDecisionColor(isReal)} />
                             <text x="0" y="-12" textAnchor="middle" fill="white" className="text-[10px] font-bold">
                                {isReal ? "REAL" : "FAKE"}
                             </text>
                        </motion.g>
                    </g>
                )
            })}

            {/* Central Discriminator */}
            <g transform={`translate(${cdPos.x}, ${cdPos.y})`}>
                <motion.rect 
                    x="-100" y="-60" width="200" height="120" rx="12"
                    fill="#1E293B"
                    stroke={stage === 3 ? getDecisionColor(decisions.cd) : "#8B5CF6"}
                    strokeWidth="3"
                    animate={{
                        scale: stage === 2 ? 1.02 : 1,
                        stroke: stage === 3 ? getDecisionColor(decisions.cd) : "#8B5CF6",
                        filter: stage === 3 ? "url(#glow)" : "none"
                    }}
                />
                <text x="0" y="80" textAnchor="middle" fill="#A78BFA" className="text-base font-bold">Central Discriminator</text>
                
                {/* Stacked Waves inside Central Discriminator */}
                {gPos.map((g, i) => {
                    const showSignal = stage >= 2;
                    // CRITICAL: The central discriminator sees the EXACT same signals that were passed to individual channel discriminators
                    // So we must use the decision of the individual channel to determine if that specific line is distorted.
                    // This represents the fact that the signal itself was generated poorly (distorted) or well (clean).
                    const channelIsReal = i === 0 ? decisions.d1 : i === 1 ? decisions.d2 : decisions.d3;
                    const isDistorted = !channelIsReal; 
                    
                    return (
                        <motion.path 
                            key={`cd-wave-${i}`}
                            // Use a wider width (120) for the Central D view
                            d={isDistorted ? getDistortedWaveIcon(g.type, 120) : getWaveIcon(g.type, 120)}
                            fill="none" 
                            stroke={g.color} 
                            strokeWidth="2" 
                            transform={`translate(-60, ${-30 + i*20}) scale(1, 0.5)`}
                            initial={{ opacity: 0, pathLength: 0 }}
                            animate={{ 
                                opacity: showSignal ? 1 : 0,
                                pathLength: showSignal ? 1 : 0,
                            }}
                            transition={{ duration: 0.5, delay: i * 0.1 }} // Staggered appearance
                        />
                    )
                })}

                {/* Final Decision Output */}
                <motion.g opacity={stage === 3 ? 1 : 0}>
                    <rect x="120" y="-20" width="80" height="40" rx="4" fill={getDecisionColor(decisions.cd)} />
                    <text x="160" y="5" textAnchor="middle" fill="white" className="font-bold text-sm">
                        {decisions.cd ? "REAL" : "FAKE"}
                    </text>
                    <line x1="100" y1="0" x2="120" y2="0" stroke="#64748B" strokeWidth="2" />
                </motion.g>
            </g>

        </svg>
    );
};

// Helper to generate clean wave shapes
// width defaults to 40 (standard icon size), can be scaled up for long waves
function getWaveIcon(type: string, width = 40) {
    const scaleX = width / 40;
    switch(type) {
        case 'sine': return `M 0 10 Q ${10*scaleX} -10 ${20*scaleX} 10 T ${40*scaleX} 10`;
        case 'square': return `M 0 15 L ${10*scaleX} 15 L ${10*scaleX} 5 L ${30*scaleX} 5 L ${30*scaleX} 15 L ${40*scaleX} 15`;
        case 'sawtooth': return `M 0 15 L ${20*scaleX} 5 L ${20*scaleX} 15 L ${40*scaleX} 5`;
        default: return `M 0 10 L ${40*scaleX} 10`;
    }
}

// Helper to generate distorted/noisy wave shapes for "Fake" visualization
function getDistortedWaveIcon(type: string, width = 40) {
    const scaleX = width / 40;
    switch(type) {
        case 'sine': 
            return `M 0 10 Q ${5*scaleX} 2 ${10*scaleX} 15 T ${25*scaleX} 5 T ${32*scaleX} 15 T ${40*scaleX} 10`;
        case 'square': 
            return `M 0 15 L ${5*scaleX} 15 L ${5*scaleX} 8 L ${15*scaleX} 2 L ${15*scaleX} 12 L ${25*scaleX} 15 L ${25*scaleX} 5 L ${35*scaleX} 8 L ${35*scaleX} 15 L ${40*scaleX} 15`;
        case 'sawtooth': 
            return `M 0 15 L ${8*scaleX} 8 L ${8*scaleX} 12 L ${20*scaleX} 5 L ${22*scaleX} 15 L ${32*scaleX} 2 L ${32*scaleX} 10 L ${40*scaleX} 5`;
        default: 
            return `M 0 10 L ${40*scaleX} 10`;
    }
}

// --- HEATMAP COMPARISON (Based on Fig 3 & Fig 14) ---
export const HeatmapComparison: React.FC = () => {
    const [mode, setMode] = useState<'real' | 'cosci' | 'nocd'>('real');

    const getGridData = (type: string) => {
        const size = 10;
        const grid = [];
        for(let y=0; y<size; y++) {
            for(let x=0; x<size; x++) {
                let val = 0;
                if (x === y) val = 1; // Identity diagonal
                else {
                    if (type === 'real') {
                        if (Math.floor(x/3) === Math.floor(y/3)) val = 0.7 + Math.random() * 0.2;
                        else val = Math.random() * 0.3;
                    } else if (type === 'cosci') {
                        if (Math.floor(x/3) === Math.floor(y/3)) val = 0.65 + Math.random() * 0.25;
                        else val = Math.random() * 0.35;
                    } else { // nocd
                        val = Math.random() * 0.5;
                    }
                }
                grid.push(val);
            }
        }
        return grid;
    };

    const data = getGridData(mode);

    return (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col h-full">
            <div className="flex justify-between items-center mb-6">
                <div className="flex gap-2 bg-slate-100 p-1 rounded-lg">
                    <button 
                        onClick={() => setMode('real')}
                        className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${mode === 'real' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                    >
                        Real Data
                    </button>
                    <button 
                        onClick={() => setMode('cosci')}
                        className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${mode === 'cosci' ? 'bg-white text-tech-blue shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                    >
                        COSCI-GAN
                    </button>
                    <button 
                        onClick={() => setMode('nocd')}
                        className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${mode === 'nocd' ? 'bg-white text-red-500 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                    >
                        Without CD
                    </button>
                </div>
            </div>

            <div className="flex-grow flex items-center justify-center">
                <div className="grid grid-cols-10 gap-0.5 bg-slate-200 p-1 w-64 h-64">
                    {data.map((val, idx) => (
                        <motion.div 
                            key={`${mode}-${idx}`}
                            initial={{ opacity: 0.5, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.2 }}
                            className="w-full h-full"
                            style={{ 
                                backgroundColor: val > 0.8 ? '#0F172A' : 
                                               val > 0.6 ? '#3B82F6' : 
                                               val > 0.4 ? '#10B981' : 
                                               val > 0.2 ? '#94A3B8' : 
                                               '#E2E8F0'
                            }}
                        />
                    ))}
                </div>
            </div>
            
            <div className="mt-6 text-center">
                <p className="text-xs text-slate-500 italic min-h-[2.5rem]">
                    {mode === 'real' && "Real data shows distinct 'blocks' of correlation between features."}
                    {mode === 'cosci' && "COSCI-GAN successfully recovers these correlation blocks."}
                    {mode === 'nocd' && "Without Central Discriminator, correlations are lost (random noise)."}
                </p>
            </div>
        </div>
    )
}

// --- ACCURACY CHART (Based on Slide 10 Augmentation Experiment) ---
export const AccuracyChart: React.FC = () => {
    // Data approximation from Slide 10 (Augmentation Experiment)
    const data = [
        { name: 'Real Data', value: 70, color: 'bg-slate-300', label: 'Baseline' },
        { name: 'TimeGAN', value: 80, color: 'bg-purple-400', label: 'SOTA 1' },
        { name: 'Fourier Flows', value: 83, color: 'bg-indigo-400', label: 'SOTA 2' },
        { name: 'COSCI-GAN', value: 88, color: 'bg-tech-teal', label: 'Ours' },
    ];

    return (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm h-full flex flex-col justify-center">
            <div className="mb-6">
                <h5 className="font-bold text-slate-800">Eye Blink Classification Accuracy</h5>
                <p className="text-xs text-slate-500">Data Augmentation Experiment (Higher is Better)</p>
            </div>
            
            <div className="space-y-6">
                {data.map((item) => (
                    <div key={item.name} className="relative">
                        <div className="flex justify-between text-sm mb-2">
                            <div className="flex items-baseline gap-2">
                                <span className="font-bold text-slate-700">{item.name}</span>
                                {item.name === 'COSCI-GAN' && <span className="text-[10px] bg-tech-teal/20 text-tech-teal px-1.5 py-0.5 rounded font-bold">BEST</span>}
                            </div>
                            <span className="font-mono font-bold text-slate-900">{item.value}%</span>
                        </div>
                        <div className="w-full h-8 bg-slate-100 rounded-md overflow-hidden relative">
                            <div className="absolute inset-0 flex justify-between px-2">
                                {[0, 25, 50, 75, 100].map(p => (
                                    <div key={p} className="h-full w-px bg-white/50"></div>
                                ))}
                            </div>
                            <motion.div 
                                className={`h-full ${item.color}`}
                                initial={{ width: 0 }}
                                whileInView={{ width: `${item.value}%` }}
                                transition={{ duration: 1.2, ease: "easeOut" }}
                            />
                        </div>
                    </div>
                ))}
            </div>
            <div className="mt-6 p-3 bg-slate-50 rounded border border-slate-100 text-xs text-slate-500">
                Augmenting the dataset with COSCI-GAN generated samples improves classification accuracy significantly more than other State-of-the-Art methods.
            </div>
        </div>
    )
}
