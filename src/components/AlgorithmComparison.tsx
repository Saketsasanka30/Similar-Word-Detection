import { AlertCircle, Loader2, Upload } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type {
  AlgorithmCatalogItem,
  BenchmarkDatasetId,
  BenchmarkPairInput,
  EvaluationResponse,
} from '../types/nlp';
import { TermTooltip } from './TermTooltip';

const FALLBACK_CATALOG: AlgorithmCatalogItem[] = [
  {
    id: 'wordnet',
    name: 'WordNet (Wu-Palmer)',
    category: 'Lexical',
    method: 'Wu-Palmer and Path Similarity',
    model: 'Princeton WordNet 3.1',
    strength: 'Identifies dictionary-based semantic relationships (synonyms, hypernyms, hyponyms).',
    limitation: 'Limited by indexed dictionary senses and affected by word-sense ambiguity.',
    metrics: {
      semanticUnderstanding: 68,
      computationalSpeed: 95,
      contextAwareness: 25,
      synonymDetection: 92,
      sentenceSupport: 15,
    },
  },
  {
    id: 'tfidf',
    name: 'TF-IDF + Cosine',
    category: 'Statistical',
    method: 'Weighted sparse term vectors',
    model: 'Smooth IDF + L2 Cosine Vectorizer',
    strength: 'Fast lexical comparison and exact keyword relevance scoring.',
    limitation: 'Does not understand synonyms or context; unrelated surface words score zero.',
    metrics: {
      semanticUnderstanding: 20,
      computationalSpeed: 98,
      contextAwareness: 15,
      synonymDetection: 10,
      sentenceSupport: 70,
    },
  },
  {
    id: 'word2vec',
    name: 'Word2Vec',
    category: 'Word embeddings',
    method: 'Static Dense Word Vectors (256-dim)',
    model: 'Distilled Word Embeddings (minishlab/M2V_base_output)',
    strength: 'Captures semantic associations learned from distributional word contexts.',
    limitation: 'Assigns one static vector per word regardless of context and flags OOV terms.',
    metrics: {
      semanticUnderstanding: 74,
      computationalSpeed: 90,
      contextAwareness: 40,
      synonymDetection: 82,
      sentenceSupport: 60,
    },
  },
  {
    id: 'fasttext',
    name: 'FastText',
    category: 'Subword embeddings',
    method: 'Subword & Char N-gram Vectors (256-dim)',
    model: 'Subword FastText Embeddings (minishlab/potion-base-8M)',
    strength: 'Captures word relationships and handles rare or unfamiliar words via subwords.',
    limitation: 'Static subword composition cannot resolve polysemy across sentences.',
    metrics: {
      semanticUnderstanding: 78,
      computationalSpeed: 88,
      contextAwareness: 45,
      synonymDetection: 85,
      sentenceSupport: 65,
    },
  },
  {
    id: 'transformer',
    name: 'Sentence-BERT',
    category: 'Contextual embeddings',
    method: 'Contextual embeddings & cosine similarity (384-dim)',
    model: 'sentence-transformers/all-MiniLM-L6-v2',
    strength: 'Primary semantic engine; aligns paraphrased words and sentences via attention.',
    limitation: 'Requires pretrained neural model weights and higher compute per call.',
    metrics: {
      semanticUnderstanding: 95,
      computationalSpeed: 64,
      contextAwareness: 96,
      synonymDetection: 93,
      sentenceSupport: 98,
    },
  },
];

function parseBenchmarkCsv(content: string): BenchmarkPairInput[] {
  const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return [];
  const firstCols = lines[0].split(',').map((c) => c.trim());
  const hasHeader = firstCols.some((c) => /^(word_a|word1|a|score|human)/i.test(c));
  const start = hasHeader ? 1 : 0;

  return lines
    .slice(start, start + 60)
    .map((line): BenchmarkPairInput | null => {
      const cols = line.split(',').map((c) => c.trim().replace(/^"|"$/g, ''));
      if (cols.length < 3) return null;
      const score = Number(cols[2]);
      if (!cols[0] || !cols[1] || !Number.isFinite(score)) return null;
      return { word_a: cols[0], word_b: cols[1], human_score: score };
    })
    .filter((x): x is BenchmarkPairInput => x !== null);
}

export function AlgorithmComparison({
  onTryInAnalyzer,
  embedded = false,
}: {
  onTryInAnalyzer?: () => void;
  embedded?: boolean;
}) {
  const csvRef = useRef<HTMLInputElement | null>(null);
  const [catalog, setCatalog] = useState<AlgorithmCatalogItem[]>(FALLBACK_CATALOG);
  const [chartMode, setChartMode] = useState<'radar' | 'bar'>('radar');

  // Evaluation Lab state
  const [datasetId, setDatasetId] = useState<BenchmarkDatasetId>('wordsim353');
  const [customPairs, setCustomPairs] = useState<BenchmarkPairInput[]>([]);
  const [customFileName, setCustomFileName] = useState<string | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evalResult, setEvalResult] = useState<EvaluationResponse | null>(null);

  useEffect(() => {
    fetch('/api/algorithms')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.algorithms && Array.isArray(data.algorithms) && data.algorithms.length >= 5) {
          setCatalog(data.algorithms);
        }
      })
      .catch(() => {});
  }, []);

  const runEvaluation = async (
    targetId: BenchmarkDatasetId,
    pairsOverride?: BenchmarkPairInput[]
  ) => {
    setEvalError(null);
    setEvalLoading(true);
    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasetId: targetId,
          pairs: targetId === 'custom' ? (pairsOverride ?? customPairs) : undefined,
        }),
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload?.error || 'Benchmark evaluation failed.');
      setEvalResult(payload as EvaluationResponse);
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : 'Unable to run benchmark evaluation.');
    } finally {
      setEvalLoading(false);
    }
  };

  const handleCustomCsvUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setEvalError(null);
    try {
      const file = files[0];
      const text = await file.text();
      const parsed = parseBenchmarkCsv(text);
      if (parsed.length < 3) {
        setEvalError(
          `Found ${parsed.length} valid rows in "${file.name}". CSV must have 3+ rows of: word_a,word_b,human_score.`
        );
        return;
      }
      setCustomPairs(parsed);
      setCustomFileName(`${file.name} (${parsed.length} pairs)`);
      setDatasetId('custom');
      void runEvaluation('custom', parsed);
    } catch {
      setEvalError('Failed to read benchmark CSV file.');
    }
  };

  const wn = catalog.find((c) => c.id === 'wordnet') ?? FALLBACK_CATALOG[0];
  const tf = catalog.find((c) => c.id === 'tfidf') ?? FALLBACK_CATALOG[1];
  const w2v = catalog.find((c) => c.id === 'word2vec') ?? FALLBACK_CATALOG[2];
  const ft = catalog.find((c) => c.id === 'fasttext') ?? FALLBACK_CATALOG[3];
  const tr = catalog.find((c) => c.id === 'transformer') ?? FALLBACK_CATALOG[4];

  const chartData = [
    {
      dimension: 'Semantic Understanding',
      WordNet: wn.metrics.semanticUnderstanding,
      TFIDF: tf.metrics.semanticUnderstanding,
      Word2Vec: w2v.metrics.semanticUnderstanding,
      FastText: ft.metrics.semanticUnderstanding,
      SentenceBERT: tr.metrics.semanticUnderstanding,
    },
    {
      dimension: 'Computational Speed',
      WordNet: wn.metrics.computationalSpeed,
      TFIDF: tf.metrics.computationalSpeed,
      Word2Vec: w2v.metrics.computationalSpeed,
      FastText: ft.metrics.computationalSpeed,
      SentenceBERT: tr.metrics.computationalSpeed,
    },
    {
      dimension: 'Context Awareness',
      WordNet: wn.metrics.contextAwareness,
      TFIDF: tf.metrics.contextAwareness,
      Word2Vec: w2v.metrics.contextAwareness,
      FastText: ft.metrics.contextAwareness,
      SentenceBERT: tr.metrics.contextAwareness,
    },
    {
      dimension: 'Synonym Detection',
      WordNet: wn.metrics.synonymDetection,
      TFIDF: tf.metrics.synonymDetection,
      Word2Vec: w2v.metrics.synonymDetection,
      FastText: ft.metrics.synonymDetection,
      SentenceBERT: tr.metrics.synonymDetection,
    },
    {
      dimension: 'Sentence Support',
      WordNet: wn.metrics.sentenceSupport,
      TFIDF: tf.metrics.sentenceSupport,
      Word2Vec: w2v.metrics.sentenceSupport,
      FastText: ft.metrics.sentenceSupport,
      SentenceBERT: tr.metrics.sentenceSupport,
    },
  ];

  const evalChartData =
    evalResult?.algorithmMetrics.map((m) => ({
      name: m.name.replace(' (Wu-Palmer)', '').replace(' (Subword)', '').replace(' (Static)', ''),
      Spearman: m.spearmanRho !== null ? Number(m.spearmanRho.toFixed(3)) : 0,
      Pearson: m.pearsonR !== null ? Number(m.pearsonR.toFixed(3)) : 0,
    })) ?? [];

  return (
    <section
      id="algorithms"
      className={
        embedded
          ? 'space-y-10'
          : 'max-w-[1440px] mx-auto px-4 sm:px-8 py-14 sm:py-20 border-t border-[rgba(207,221,216,0.10)] space-y-10'
      }
    >
      <div>
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 mb-10">
          <div>
            <p className="font-mono text-[11px] tracking-[0.16em] text-[#5BA4D6] mb-2">
              5-ALGORITHM ARCHITECTURE
            </p>
            <h2 className="font-display font-bold text-2xl sm:text-4xl tracking-tight text-[#CFDDD8]">
              One language. Five perspectives.
            </h2>
            <p className="text-sm sm:text-base text-[#91A4B0] mt-2 max-w-2xl">
              Compare lexical graphs, statistical <TermTooltip termKey="TF-IDF">TF-IDF</TermTooltip>, static Word2Vec, subword FastText, and contextual Sentence-BERT embeddings side by side.
            </p>
          </div>

          {onTryInAnalyzer && (
            <button
              type="button"
              onClick={onTryInAnalyzer}
              className="self-start lg:self-auto px-4 py-2.5 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/35 text-xs font-medium transition-colors cursor-pointer whitespace-nowrap"
            >
              Compare Live on Words
            </button>
          )}
        </div>

        {/* 5 Algorithm Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4 mb-10">
          {catalog.map((algo, index) => (
            <article
              key={algo.id}
              className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-5 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between text-[11px] font-mono text-[#647987] mb-2.5">
                  <span>0{index + 1} · {algo.category}</span>
                </div>
                <h3 className="font-display text-lg font-semibold text-[#CFDDD8]">
                  {algo.name}
                </h3>
                <p className="font-mono text-[11px] text-[#5BA4D6] mt-1">{algo.method}</p>

                <dl className="mt-4 space-y-2.5 text-xs leading-relaxed">
                  <div>
                    <dt className="font-mono text-[10px] tracking-wider text-[#65C99A] mb-0.5">
                      STRENGTH
                    </dt>
                    <dd className="text-[#CFDDD8]">{algo.strength}</dd>
                  </div>
                  <div>
                    <dt className="font-mono text-[10px] tracking-wider text-[#E9B96E] mb-0.5">
                      LIMITATION
                    </dt>
                    <dd className="text-[#91A4B0]">{algo.limitation}</dd>
                  </div>
                </dl>
              </div>

              <div className="mt-4 pt-3 border-t border-[rgba(207,221,216,0.08)] font-mono text-[10px] text-[#647987] truncate">
                {algo.model}
              </div>
            </article>
          ))}
        </div>

        {/* Qualitative Profile Chart */}
        <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-6 sm:p-8">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
            <div>
              <h3 className="font-display text-lg font-semibold text-[#CFDDD8]">
                Algorithm Capability Trade-offs
              </h3>
              <p className="text-xs text-[#91A4B0] mt-0.5">
                Each algorithm captures distinct linguistic properties and operates on non-equivalent similarity scales.
              </p>
            </div>

            <div
              role="group"
              aria-label="Chart visualization mode"
              className="inline-flex p-1 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.12)] self-start"
            >
              <button
                type="button"
                onClick={() => setChartMode('radar')}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors cursor-pointer ${
                  chartMode === 'radar'
                    ? 'bg-[#143F62] text-[#CFDDD8]'
                    : 'text-[#91A4B0] hover:text-[#CFDDD8]'
                }`}
              >
                Radar Profile
              </button>
              <button
                type="button"
                onClick={() => setChartMode('bar')}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors cursor-pointer ${
                  chartMode === 'bar'
                    ? 'bg-[#143F62] text-[#CFDDD8]'
                    : 'text-[#91A4B0] hover:text-[#CFDDD8]'
                }`}
              >
                Grouped Bars
              </button>
            </div>
          </div>

          <div className="w-full h-[340px]">
            <ResponsiveContainer width="100%" height="100%">
              {chartMode === 'radar' ? (
                <RadarChart cx="50%" cy="50%" outerRadius="74%" data={chartData}>
                  <PolarGrid stroke="rgba(207,221,216,0.12)" />
                  <PolarAngleAxis dataKey="dimension" tick={{ fill: '#91A4B0', fontSize: 11 }} />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 100]}
                    tick={{ fill: '#647987', fontSize: 10 }}
                    axisLine={false}
                  />
                  <Radar name="WordNet" dataKey="WordNet" stroke="#E9B96E" fill="#E9B96E" fillOpacity={0.1} strokeWidth={1.6} />
                  <Radar name="TF-IDF" dataKey="TFIDF" stroke="#91A4B0" fill="#91A4B0" fillOpacity={0.08} strokeWidth={1.4} />
                  <Radar name="Word2Vec" dataKey="Word2Vec" stroke="#B497CF" fill="#B497CF" fillOpacity={0.12} strokeWidth={1.6} />
                  <Radar name="FastText" dataKey="FastText" stroke="#5BA4D6" fill="#5BA4D6" fillOpacity={0.14} strokeWidth={1.75} />
                  <Radar name="Sentence-BERT" dataKey="SentenceBERT" stroke="#65C99A" fill="#65C99A" fillOpacity={0.18} strokeWidth={2} />
                  <Legend wrapperStyle={{ fontSize: '12px', color: '#CFDDD8' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#071522',
                      borderColor: 'rgba(207,221,216,0.18)',
                      borderRadius: '6px',
                      fontSize: '12px',
                      color: '#CFDDD8',
                    }}
                  />
                </RadarChart>
              ) : (
                <BarChart data={chartData} margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(207,221,216,0.08)" />
                  <XAxis dataKey="dimension" tick={{ fill: '#91A4B0', fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: '#647987', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#071522',
                      borderColor: 'rgba(207,221,216,0.18)',
                      borderRadius: '6px',
                      fontSize: '12px',
                      color: '#CFDDD8',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="WordNet" fill="#E9B96E" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="TFIDF" fill="#91A4B0" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Word2Vec" fill="#B497CF" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="FastText" fill="#5BA4D6" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="SentenceBERT" fill="#65C99A" radius={[3, 3, 0, 0]} />
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Algorithm Evaluation Lab (WordSim-353, SimLex-999, Custom CSV) */}
      <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-6 sm:p-8 space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 pb-5 border-b border-[rgba(207,221,216,0.10)]">
          <div>
            <p className="font-mono text-[11px] tracking-[0.16em] text-[#5BA4D6] mb-1.5">
              ALGORITHM EVALUATION LAB
            </p>
            <h3 className="font-display text-xl sm:text-2xl font-bold text-[#CFDDD8]">
              Evaluate against human similarity benchmarks
            </h3>
            <p className="text-xs sm:text-sm text-[#91A4B0] mt-1 max-w-2xl">
              Compute real Spearman rank correlation (ρ) and Pearson correlation (r) against WordSim-353, SimLex-999, or a custom CSV (<span className="font-mono">word_a,word_b,human_score</span>).
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <input
              ref={csvRef}
              type="file"
              accept=".csv,.txt"
              onChange={(e) => void handleCustomCsvUpload(e.target.files)}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => csvRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-md bg-[#071522] hover:bg-[#010A12] border border-[rgba(207,221,216,0.14)] text-xs font-medium text-[#CFDDD8] cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5 text-[#5BA4D6]" />
              Upload Custom CSV
            </button>
            <button
              type="button"
              onClick={() => void runEvaluation(datasetId)}
              disabled={evalLoading}
              className="inline-flex items-center gap-2 px-5 py-2 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/40 text-xs font-medium cursor-pointer disabled:opacity-60"
            >
              {evalLoading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Evaluating 5 Algorithms...
                </>
              ) : (
                'Run Evaluation'
              )}
            </button>
          </div>
        </div>

        {/* Benchmark Selector */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setDatasetId('wordsim353')}
            className={`px-4 py-2 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
              datasetId === 'wordsim353'
                ? 'bg-[#143F62] border-[#5BA4D6]/60 text-[#CFDDD8]'
                : 'bg-[#071522] border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8]'
            }`}
          >
            WordSim-353 (20 pairs)
          </button>
          <button
            type="button"
            onClick={() => setDatasetId('simlex999')}
            className={`px-4 py-2 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
              datasetId === 'simlex999'
                ? 'bg-[#143F62] border-[#5BA4D6]/60 text-[#CFDDD8]'
                : 'bg-[#071522] border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8]'
            }`}
          >
            SimLex-999 (20 pairs)
          </button>
          {customFileName && (
            <button
              type="button"
              onClick={() => setDatasetId('custom')}
              className={`px-4 py-2 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
                datasetId === 'custom'
                  ? 'bg-[#143F62] border-[#5BA4D6]/60 text-[#CFDDD8]'
                  : 'bg-[#071522] border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8]'
              }`}
            >
              {customFileName}
            </button>
          )}
        </div>

        {evalError && (
          <div
            role="alert"
            className="flex items-center gap-2 p-3 rounded-md bg-[#E17E7E]/10 border border-[#E17E7E]/30 text-sm text-[#E17E7E]"
          >
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{evalError}</span>
          </div>
        )}

        {evalResult && !evalLoading && (
          <div className="space-y-6">
            <p className="text-sm text-[#CFDDD8]">{evalResult.summary}</p>

            {/* Correlation Comparison Chart */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
              <div className="lg:col-span-6 h-[270px] bg-[#071522]/70 rounded-md border border-[rgba(207,221,216,0.08)] p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={evalChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(207,221,216,0.08)" />
                    <XAxis dataKey="name" tick={{ fill: '#91A4B0', fontSize: 10 }} />
                    <YAxis domain={[-0.2, 1]} tick={{ fill: '#647987', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#071522',
                        borderColor: 'rgba(207,221,216,0.18)',
                        borderRadius: '6px',
                        fontSize: '12px',
                        color: '#CFDDD8',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Bar dataKey="Spearman" name="Spearman ρ" fill="#65C99A" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="Pearson" name="Pearson r" fill="#5BA4D6" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="lg:col-span-6 overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[rgba(207,221,216,0.12)] font-mono text-[10px] text-[#91A4B0]">
                      <th className="py-2 pr-3">ALGORITHM</th>
                      <th className="py-2 pr-3">SPEARMAN ρ</th>
                      <th className="py-2 pr-3">PEARSON r</th>
                      <th className="py-2 pr-3">MAE</th>
                      <th className="py-2">TIME</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[rgba(207,221,216,0.08)]">
                    {evalResult.algorithmMetrics.map((m) => (
                      <tr key={m.algorithm}>
                        <td className="py-2.5 pr-3 font-medium text-[#CFDDD8]">{m.name}</td>
                        <td className="py-2.5 pr-3 font-mono text-[#65C99A] tabular-nums">
                          {m.spearmanRho !== null ? m.spearmanRho.toFixed(3) : 'N/A'}
                        </td>
                        <td className="py-2.5 pr-3 font-mono text-[#5BA4D6] tabular-nums">
                          {m.pearsonR !== null ? m.pearsonR.toFixed(3) : 'N/A'}
                        </td>
                        <td className="py-2.5 pr-3 font-mono text-[#91A4B0] tabular-nums">
                          {m.mae !== null ? `${m.mae.toFixed(1)}%` : 'N/A'}
                        </td>
                        <td className="py-2.5 font-mono text-[#647987] tabular-nums">
                          {m.processingTimeMs}ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Sample Pair Predictions */}
            <div className="overflow-x-auto max-h-[300px] overflow-y-auto border border-[rgba(207,221,216,0.08)] rounded-md">
              <table className="w-full text-left border-collapse text-xs">
                <thead className="sticky top-0 bg-[#071522] border-b border-[rgba(207,221,216,0.12)] font-mono text-[10px] text-[#91A4B0]">
                  <tr>
                    <th className="py-2.5 px-3">PAIR</th>
                    <th className="py-2.5 px-3">HUMAN</th>
                    <th className="py-2.5 px-3">S-BERT</th>
                    <th className="py-2.5 px-3">WORDNET</th>
                    <th className="py-2.5 px-3">FASTTEXT</th>
                    <th className="py-2.5 px-3">WORD2VEC</th>
                    <th className="py-2.5 px-3">TF-IDF</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[rgba(207,221,216,0.06)]">
                  {evalResult.pairs.map((row, idx) => (
                    <tr key={`${row.word_a}-${row.word_b}-${idx}`} className="hover:bg-[#071522]/40">
                      <td className="py-2 px-3 font-display text-[#CFDDD8] whitespace-nowrap">
                        {row.word_a} ↔ {row.word_b}
                      </td>
                      <td className="py-2 px-3 font-mono text-[#E9B96E] tabular-nums">
                        {row.humanScore.toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 font-mono text-[#65C99A] tabular-nums">
                        {row.predictions.transformer !== null ? `${row.predictions.transformer.toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-2 px-3 font-mono text-[#91A4B0] tabular-nums">
                        {row.predictions.wordnet !== null ? `${row.predictions.wordnet.toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-2 px-3 font-mono text-[#91A4B0] tabular-nums">
                        {row.predictions.fasttext !== null ? `${row.predictions.fasttext.toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-2 px-3 font-mono text-[#91A4B0] tabular-nums">
                        {row.predictions.word2vec !== null ? `${row.predictions.word2vec.toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-2 px-3 font-mono text-[#647987] tabular-nums">
                        {row.predictions.tfidf !== null ? `${row.predictions.tfidf.toFixed(1)}%` : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
