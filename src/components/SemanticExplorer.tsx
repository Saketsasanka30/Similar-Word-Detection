import { AlertCircle, Loader2, Search } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { SemanticExploreResponse, SemanticGraphNode } from '../types/nlp';
import { TermTooltip } from './TermTooltip';

interface SemanticExplorerProps {
  initialWord?: string;
}

const QUICK_WORDS = ['Vehicle', 'Happy', 'Bank', 'Computer', 'Ocean', 'Music'];

const RELATION_STYLE: Record<
  'synonym' | 'hypernym' | 'hyponym' | 'related' | 'embedding',
  { stroke: string; dasharray?: string; label: string }
> = {
  synonym: { stroke: '#5BA4D6', label: 'Synonym' },
  hypernym: { stroke: '#65C99A', dasharray: '6 4', label: 'Hypernym (broader)' },
  hyponym: { stroke: '#91A4B0', dasharray: '2 4', label: 'Hyponym (narrower)' },
  related: { stroke: '#E9B96E', dasharray: '8 3 2 3', label: 'Lexical Related' },
  embedding: { stroke: '#B497CF', dasharray: '4 2', label: 'Embedding Neighbor' },
};

export function SemanticExplorer({ initialWord = 'Vehicle' }: SemanticExplorerProps) {
  const [query, setQuery] = useState(initialWord);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SemanticExploreResponse | null>(null);
  const [hoveredNode, setHoveredNode] = useState<SemanticGraphNode | null>(null);

  const fetchNetwork = async (targetWord: string) => {
    const clean = targetWord.trim();
    if (!clean) {
      setError('Enter a word to explore its semantic neighborhood.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/semantic/related/${encodeURIComponent(clean)}`);
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload?.error || 'Failed to load semantic network.');
      }
      setData(payload as SemanticExploreResponse);
      setHoveredNode((payload as SemanticExploreResponse).nodes[0] ?? null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Unable to query Semantic Explorer.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setQuery(initialWord);
    void fetchNetwork(initialWord);
  }, [initialWord]);

  const layoutNodes = useMemo(() => {
    if (!data || !data.nodes.length) return [];
    const cx = 380;
    const cy = 240;

    const centerNode = data.nodes.find((n) => n.relation === 'center');
    const satellites = data.nodes.filter((n) => n.relation !== 'center');

    const positioned: Array<SemanticGraphNode & { x: number; y: number }> = [];
    if (centerNode) {
      positioned.push({ ...centerNode, x: cx, y: cy });
    }

    const count = satellites.length;
    satellites.forEach((node, idx) => {
      const angle = (idx / Math.max(1, count)) * Math.PI * 2 - Math.PI / 2;
      const radiusX =
        node.relation === 'synonym'
          ? 148
          : node.relation === 'hypernym'
            ? 178
            : node.relation === 'embedding'
              ? 200
              : 215;
      const radiusY =
        node.relation === 'synonym'
          ? 114
          : node.relation === 'hypernym'
            ? 140
            : node.relation === 'embedding'
              ? 158
              : 168;
      positioned.push({
        ...node,
        x: cx + Math.cos(angle) * radiusX,
        y: cy + Math.sin(angle) * radiusY,
      });
    });

    return positioned;
  }, [data]);

  const posMap = useMemo(
    () => new Map(layoutNodes.map((n) => [n.id, n])),
    [layoutNodes]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void fetchNetwork(query);
  };

  return (
    <section className="max-w-[1440px] mx-auto px-4 sm:px-8 py-14 sm:py-20">
      <div className="mb-8">
        <p className="font-mono text-[11px] tracking-[0.16em] text-[#5BA4D6] mb-2">
          SEMANTIC EXPLORER
        </p>
        <h2 className="font-display font-bold text-2xl sm:text-4xl tracking-tight text-[#CFDDD8]">
          Explore the language network.
        </h2>
        <p className="text-sm sm:text-base text-[#91A4B0] mt-2 max-w-2xl">
          Inspect <TermTooltip termKey="WordNet">WordNet</TermTooltip> synonyms, hypernyms, and hyponyms alongside neural embedding neighbors. Click any connected word to recenter the network.
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-5 mb-6"
      >
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-[#647987] absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter an English word (e.g. Vehicle, Happy, Computer)..."
              aria-label="Word to explore"
              className="w-full h-11 pl-10 pr-4 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-sm text-[#CFDDD8] placeholder:text-[#647987]"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-6 h-11 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/40 text-sm font-medium transition-colors cursor-pointer whitespace-nowrap"
          >
            {loading ? 'Loading...' : 'Explore Word'}
          </button>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="font-mono text-[11px] text-[#647987] mr-1">Try:</span>
          {QUICK_WORDS.map((w) => (
            <button
              key={w}
              type="button"
              onClick={() => {
                setQuery(w);
                void fetchNetwork(w);
              }}
              className={`px-3 py-1 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
                data?.word.toLowerCase() === w.toLowerCase()
                  ? 'bg-[#143F62] border-[#5BA4D6]/60 text-[#CFDDD8]'
                  : 'bg-[#071522] border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8]'
              }`}
            >
              {w}
            </button>
          ))}
        </div>

        {error && (
          <div
            role="alert"
            className="mt-4 flex items-center gap-2 p-3 rounded-md bg-[#E17E7E]/10 border border-[#E17E7E]/30 text-sm text-[#E17E7E]"
          >
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </form>

      {loading && (
        <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-16 flex flex-col items-center justify-center gap-3 text-sm text-[#91A4B0]">
          <Loader2 className="w-6 h-6 animate-spin text-[#5BA4D6]" />
          <span>Building lexical &amp; embedding network...</span>
        </div>
      )}

      {!loading && data && !data.found && (
        <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-10 text-center">
          <p className="font-display text-lg font-medium text-[#CFDDD8]">
            No lexical synsets found for &ldquo;{data.word}&rdquo;
          </p>
          <p className="text-sm text-[#91A4B0] mt-1.5 max-w-md mx-auto">
            {data.message}
          </p>
        </div>
      )}

      {!loading && data && data.found && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          <div className="lg:col-span-8 rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-4 sm:p-5 flex flex-col">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-3 border-b border-[rgba(207,221,216,0.10)] text-xs">
              <div className="flex flex-wrap items-center gap-4">
                {Object.entries(RELATION_STYLE).map(([key, style]) => (
                  <div key={key} className="inline-flex items-center gap-2 text-[#91A4B0]">
                    <svg width="24" height="6" aria-hidden="true">
                      <line
                        x1="0"
                        y1="3"
                        x2="24"
                        y2="3"
                        stroke={style.stroke}
                        strokeWidth="2"
                        strokeDasharray={style.dasharray}
                      />
                    </svg>
                    <span>{style.label}</span>
                  </div>
                ))}
              </div>
              <span className="font-mono text-[11px] text-[#647987]">
                {data.nodes.length} nodes · {data.edges.length} edges
              </span>
            </div>

            <div className="flex-1 min-h-[420px] flex items-center justify-center bg-[#071522]/60 rounded-md overflow-hidden">
              <svg
                viewBox="0 0 760 480"
                className="w-full h-full max-h-[480px] select-none"
                role="img"
                aria-label={`Semantic network centered on ${data.word}`}
              >
                {data.edges.map((edge, idx) => {
                  const s = posMap.get(edge.source);
                  const t = posMap.get(edge.target);
                  if (!s || !t) return null;
                  const style = RELATION_STYLE[edge.relation] ?? RELATION_STYLE.related;
                  const isHighlighted =
                    hoveredNode?.id === t.id || hoveredNode?.id === s.id;
                  return (
                    <line
                      key={`${edge.source}-${edge.target}-${idx}`}
                      x1={s.x}
                      y1={s.y}
                      x2={t.x}
                      y2={t.y}
                      stroke={style.stroke}
                      strokeDasharray={style.dasharray}
                      strokeWidth={isHighlighted ? 2 : 1.25}
                      strokeOpacity={isHighlighted ? 0.95 : 0.45}
                    />
                  );
                })}

                {layoutNodes.map((node) => {
                  const isCenter = node.relation === 'center';
                  const isHovered = hoveredNode?.id === node.id;
                  const relColor =
                    node.relation === 'center'
                      ? '#5BA4D6'
                      : (RELATION_STYLE[node.relation]?.stroke ?? '#5BA4D6');

                  return (
                    <g
                      key={node.id}
                      transform={`translate(${node.x}, ${node.y})`}
                      onMouseEnter={() => setHoveredNode(node)}
                      onClick={() => {
                        if (!isCenter) {
                          setQuery(node.label);
                          void fetchNetwork(node.label);
                        }
                      }}
                      className={isCenter ? 'cursor-default' : 'cursor-pointer'}
                      role="button"
                      tabIndex={0}
                      aria-label={`${node.label} (${node.relation})`}
                      onKeyDown={(e) => {
                        if ((e.key === 'Enter' || e.key === ' ') && !isCenter) {
                          e.preventDefault();
                          setQuery(node.label);
                          void fetchNetwork(node.label);
                        }
                      }}
                    >
                      <circle
                        r={isCenter ? 28 : isHovered ? 18 : 14}
                        fill={isCenter ? '#143F62' : isHovered ? '#1B5278' : '#0B1925'}
                        stroke={relColor}
                        strokeWidth={isCenter ? 2 : 1.5}
                      />
                      <text
                        y={isCenter ? 44 : 30}
                        textAnchor="middle"
                        className="font-display text-[12px] font-medium"
                        fill={isCenter || isHovered ? '#CFDDD8' : '#91A4B0'}
                      >
                        {node.label}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>

          <div className="lg:col-span-4 rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-5 flex flex-col justify-between gap-6">
            <div>
              <p className="font-mono text-[11px] tracking-wider text-[#5BA4D6] mb-2">
                NODE INSPECTOR
              </p>
              {hoveredNode ? (
                <div className="p-4 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.10)]">
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="font-display text-lg font-semibold text-[#CFDDD8]">
                      {hoveredNode.label}
                    </h3>
                    <span className="font-mono text-xs text-[#91A4B0]">
                      {hoveredNode.pos}
                    </span>
                  </div>
                  <p className="font-mono text-[11px] text-[#5BA4D6] mt-1 capitalize">
                    Relation: {hoveredNode.relation}
                    {hoveredNode.similarityScore !== undefined
                      ? ` (${hoveredNode.similarityScore.toFixed(3)})`
                      : ''}
                  </p>
                  <p className="text-xs text-[#91A4B0] leading-relaxed mt-2.5">
                    {hoveredNode.gloss}
                  </p>
                </div>
              ) : null}

              <div className="mt-5">
                <p className="font-mono text-[11px] tracking-wider text-[#91A4B0] mb-2.5">
                  IDENTIFIED WORDNET SYNSETS ({data.synsets.length})
                </p>
                <ul className="space-y-2 max-h-[240px] overflow-y-auto pr-1">
                  {data.synsets.map((syn) => (
                    <li
                      key={syn.id}
                      className="p-3 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.08)] text-xs"
                    >
                      <div className="font-mono text-[11px] text-[#5BA4D6]">
                        {syn.id} · {syn.posLabel}
                      </div>
                      <div className="text-[#CFDDD8] mt-1">
                        {syn.lemmas.join(', ')}
                      </div>
                      <div className="text-[#91A4B0] mt-0.5 leading-relaxed">
                        {syn.gloss}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <p className="text-[11px] text-[#647987] border-t border-[rgba(207,221,216,0.08)] pt-3">
              Combines Princeton WordNet 3.1 lexical pointers with neural embedding neighbors.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
