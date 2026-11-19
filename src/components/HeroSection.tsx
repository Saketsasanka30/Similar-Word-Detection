import { ArrowRight, Upload } from 'lucide-react';
import { motion } from 'motion/react';
import { useState } from 'react';

interface HeroSectionProps {
  onUploadDataset: () => void;
}

interface NetworkNode {
  id: string;
  label: string;
  x: number;
  y: number;
  primary?: boolean;
}

const HERO_NODES: NetworkNode[] = [
  { id: 'vehicle', label: 'Vehicle', x: 250, y: 105, primary: true },
  { id: 'car', label: 'Car', x: 112, y: 185, primary: true },
  { id: 'automobile', label: 'Automobile', x: 260, y: 235 },
  { id: 'transport', label: 'Transport', x: 392, y: 158 },
  { id: 'engine', label: 'Engine', x: 108, y: 315 },
  { id: 'road', label: 'Road', x: 376, y: 308 },
];

const HERO_EDGES: Array<{ from: string; to: string }> = [
  { from: 'vehicle', to: 'car' },
  { from: 'vehicle', to: 'automobile' },
  { from: 'vehicle', to: 'transport' },
  { from: 'car', to: 'automobile' },
  { from: 'car', to: 'engine' },
  { from: 'car', to: 'road' },
  { from: 'automobile', to: 'road' },
  { from: 'transport', to: 'road' },
];

export function HeroSection({
  onUploadDataset,
}: HeroSectionProps) {
  const [hoveredId, setHoveredId] = useState<string | null>('car');
  const nodeMap = new Map(HERO_NODES.map((n) => [n.id, n]));

  return (
    <section className="relative overflow-hidden border-b border-[rgba(207,221,216,0.08)]">
      {/* Luminex-style subtle atmospheric radial spotlight & fine grid */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_55%_at_70%_45%,rgba(20,63,98,0.28),transparent_72%)]"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(207,221,216,0.025)_1px,transparent_1px),linear-gradient(90deg,rgba(207,221,216,0.025)_1px,transparent_1px)] bg-[size:64px_64px] [mask-image:radial-gradient(ellipse_70%_70%_at_50%_40%,#000_40%,transparent_100%)]"
      />

      <div className="relative max-w-[1440px] mx-auto px-4 sm:px-8 py-20 sm:py-28 lg:py-32">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left Column: Clean Luminex Hero Typography */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
            className="lg:col-span-7 flex flex-col items-start"
          >
            <p className="font-mono text-[11px] tracking-[0.2em] text-[#91A4B0] mb-5">
              SEMANTIQ · LANGUAGE SIMILARITY
            </p>

            <h1 className="font-display font-bold tracking-[-0.03em] leading-[1.04] text-[clamp(2.6rem,5.4vw,4.5rem)] text-[#CFDDD8] text-balance mb-6">
              Words are connected.
              <br />
              <span className="bg-gradient-to-r from-[#5BA4D6] to-[#91C8EC] bg-clip-text text-transparent">
                Discover how.
              </span>
            </h1>

            <p className="text-base sm:text-lg leading-relaxed text-[#91A4B0] max-w-lg mb-9">
              Upload a text dataset to train word-context vectors and discover the most similar words with cosine similarity.
            </p>

            <div className="flex flex-wrap items-center gap-4 mb-12">
              <button
                type="button"
                onClick={onUploadDataset}
                className="inline-flex items-center gap-2.5 px-6 py-3 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/40 text-sm font-medium transition-colors cursor-pointer whitespace-nowrap"
              >
                <Upload className="w-4 h-4 text-[#5BA4D6]" />
                Upload Dataset
                <ArrowRight className="w-4 h-4 text-[#5BA4D6]" />
              </button>
            </div>

            <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-[#647987]">
              <span>Upload Text Corpus</span>
              <span aria-hidden="true">·</span>
              <span>Word-Context Vectors</span>
              <span aria-hidden="true">·</span>
              <span>Cosine Similarity</span>
            </div>
          </motion.div>

          {/* Right Column: Borderless Atmospheric Constellation */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.65, delay: 0.1 }}
            className="lg:col-span-5"
          >
            <svg
              viewBox="0 0 490 400"
              className="w-full h-auto select-none"
              role="img"
              aria-label="Interactive semantic network linking Vehicle, Car, Automobile, Transport, Engine, and Road"
            >
              <defs>
                <radialGradient id="node-glow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#5BA4D6" stopOpacity="0.28" />
                  <stop offset="100%" stopColor="#5BA4D6" stopOpacity="0" />
                </radialGradient>
              </defs>

              <circle cx="245" cy="200" r="170" fill="url(#node-glow)" opacity="0.45" />

              {HERO_EDGES.map((edge) => {
                const s = nodeMap.get(edge.from);
                const t = nodeMap.get(edge.to);
                if (!s || !t) return null;
                const active = hoveredId === s.id || hoveredId === t.id;
                return (
                  <line
                    key={`${edge.from}-${edge.to}`}
                    x1={s.x}
                    y1={s.y}
                    x2={t.x}
                    y2={t.y}
                    stroke={active ? '#5BA4D6' : 'rgba(207,221,216,0.16)'}
                    strokeWidth={active ? 1.5 : 1}
                    strokeOpacity={active ? 0.8 : 0.65}
                  />
                );
              })}

              {HERO_NODES.map((node) => {
                const isHovered = hoveredId === node.id;
                const radius = node.primary ? 24 : 20;

                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x}, ${node.y})`}
                    onMouseEnter={() => setHoveredId(node.id)}
                    onClick={onUploadDataset}
                    className="cursor-pointer"
                    role="button"
                    tabIndex={0}
                    aria-label={`Upload dataset (${node.label})`}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        onUploadDataset();
                      }
                    }}
                  >
                    {isHovered && (
                      <circle
                        r={radius + 10}
                        fill="none"
                        stroke="#5BA4D6"
                        strokeOpacity="0.25"
                        strokeWidth="1"
                      />
                    )}
                    <circle
                      r={radius}
                      fill={isHovered ? '#143F62' : '#071522'}
                      stroke={isHovered ? '#5BA4D6' : 'rgba(207,221,216,0.22)'}
                      strokeWidth={isHovered ? 1.6 : 1.1}
                    />
                    <circle r={3} fill={isHovered ? '#5BA4D6' : '#91A4B0'} />
                    <text
                      y={radius + 18}
                      textAnchor="middle"
                      className="font-display text-[12px] font-medium"
                      fill={isHovered ? '#CFDDD8' : '#91A4B0'}
                    >
                      {node.label}
                    </text>
                  </g>
                );
              })}
            </svg>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
