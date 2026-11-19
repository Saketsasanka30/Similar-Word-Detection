import {
  AlertCircle,
  ArrowLeftRight,
  ChevronDown,
  ChevronUp,
  Loader2,
  RotateCcw,
  Search,
} from 'lucide-react';
import { motion } from 'motion/react';
import { useEffect, useState } from 'react';
import type {
  AlgorithmChoice,
  AlgorithmResult,
  SimilarWordsResponse,
  WordAnalysisResponse,
} from '../types/nlp';
import { ContextDetector } from './ContextDetector';
import { NlpPipelineSteps } from './NlpPipelineSteps';
import { SimilarityGauge } from './SimilarityGauge';
import { VectorAngleDiagram } from './VectorAngleDiagram';

interface WordAnalyzerProps {
  onAnalysisComplete: (result: WordAnalysisResponse) => void;
  restoredResult?: WordAnalysisResponse | null;
  onExploreWord?: (word: string) => void;
  embedded?: boolean;
}

const SINGLE_WORD_EXAMPLES = ['Computer', 'Happy', 'Car', 'Smart', 'Ocean', 'Music'];

const EXAMPLE_PAIRS: Array<{ a: string; b: string }> = [
  { a: 'Happy', b: 'Joyful' },
  { a: 'Car', b: 'Automobile' },
  { a: 'Computer', b: 'Code' },
  { a: 'King', b: 'Queen' },
  { a: 'Apple', b: 'Banana' },
  { a: 'Bank', b: 'River Bank' },
];

const ALGORITHM_OPTIONS: Array<{ value: AlgorithmChoice; label: string }> = [
  { value: 'all', label: 'Compare All (5 Algorithms)' },
  { value: 'transformer', label: 'Sentence-BERT (Contextual)' },
  { value: 'wordnet', label: 'WordNet (Wu-Palmer)' },
  { value: 'word2vec', label: 'Word2Vec (Word Embeddings)' },
  { value: 'fasttext', label: 'FastText (Subword Embeddings)' },
  { value: 'tfidf', label: 'TF-IDF + Cosine (Statistical)' },
];

function validateField(raw: string, label: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return `${label} is required.`;
  if (trimmed.length > 80) return `${label} must be 80 characters or fewer.`;
  if (!/^[a-zA-Z][a-zA-Z\s'-]*$/.test(trimmed)) {
    return `${label} must contain letters (spaces, hyphens, or apostrophes allowed).`;
  }
  return null;
}

export function WordAnalyzer({
  onAnalysisComplete,
  restoredResult,
  onExploreWord,
  embedded = false,
}: WordAnalyzerProps) {
  const [wordMode, setWordMode] = useState<'single' | 'pair' | 'context'>('single');

  // Single-word Smart Synonym Finder state
  const [singleWord, setSingleWord] = useState('Computer');
  const [topK, setTopK] = useState<5 | 10>(5);
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleError, setSingleError] = useState<string | null>(null);
  const [singleResult, setSingleResult] = useState<SimilarWordsResponse | null>(null);
  const [showSingleDetails, setShowSingleDetails] = useState(false);

  // Two-word Pair Comparison state
  const [wordA, setWordA] = useState('Happy');
  const [wordB, setWordB] = useState('Joyful');
  const [algorithm, setAlgorithm] = useState<AlgorithmChoice>('all');
  const [validationError, setValidationError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<WordAnalysisResponse | null>(null);
  const [showTechnical, setShowTechnical] = useState(false);

  useEffect(() => {
    if (restoredResult) {
      setWordMode('pair');
      setWordA(restoredResult.word_a);
      setWordB(restoredResult.word_b);
      setAlgorithm(restoredResult.algorithm);
      setResult(restoredResult);
      setValidationError(null);
      setApiError(null);
    }
  }, [restoredResult]);

  const runSimilarSearch = async (inputWord: string, limit: 5 | 10) => {
    if (!inputWord.trim()) {
      setSingleError('Please enter a word to find similar words.');
      return;
    }
    setSingleError(null);
    setSingleLoading(true);
    try {
      const response = await fetch('/api/similar-words', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word: inputWord, top_k: limit }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.error || 'Failed to find similar words.');
      }
      setSingleResult(payload as SimilarWordsResponse);
    } catch (err) {
      setSingleError(
        err instanceof Error ? err.message : 'Unable to reach similar-word engine.'
      );
    } finally {
      setSingleLoading(false);
    }
  };

  const runPairAnalysis = async (aVal: string, bVal: string, algoVal: AlgorithmChoice) => {
    const errA = validateField(aVal, 'Word A');
    const errB = validateField(bVal, 'Word B');
    if (errA || errB) {
      setValidationError(errA || errB);
      return;
    }

    setValidationError(null);
    setApiError(null);
    setLoading(true);

    try {
      const endpoint = algoVal === 'all' ? '/api/analyze/compare' : '/api/analyze/word';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          word_a: aVal.trim(),
          word_b: bVal.trim(),
          algorithm: algoVal,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.error || 'Failed to compute similarity.');
      }
      setResult(payload as WordAnalysisResponse);
      onAnalysisComplete(payload as WordAnalysisResponse);
    } catch (err) {
      setApiError(
        err instanceof Error ? err.message : 'Unable to reach the analysis service.'
      );
    } finally {
      setLoading(false);
    }
  };

  const orderedResults: AlgorithmResult[] = result
    ? (['transformer', 'wordnet', 'fasttext', 'word2vec', 'tfidf'] as const)
        .map((key) => result.results[key])
        .filter((r): r is AlgorithmResult => Boolean(r))
    : [];

  return (
    <section
      id="word-analyzer"
      className={embedded ? '' : 'max-w-[1200px] mx-auto px-4 sm:px-8 py-14 sm:py-20'}
    >
      {!embedded && (
        <div className="mb-8">
          <p className="font-mono text-[11px] tracking-[0.16em] text-[#5BA4D6] mb-2">
            SIMILAR WORD DETECTION
          </p>
          <h2 className="font-display font-bold text-2xl sm:text-4xl tracking-tight text-[#CFDDD8]">
            Find similar words &amp; detect meaning
          </h2>
          <p className="text-sm sm:text-base text-[#91A4B0] mt-2">
            Discover exact synonyms, compare words across five NLP algorithms, or disambiguate meaning from sentence context.
          </p>
        </div>
      )}

      {/* Clean Sub-Mode Switch */}
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div
          role="tablist"
          aria-label="Word analysis mode"
          className="inline-flex flex-wrap p-1 rounded-md bg-[#071522]/90 border border-[rgba(207,221,216,0.12)]"
        >
          <button
            type="button"
            role="tab"
            aria-selected={wordMode === 'single'}
            onClick={() => setWordMode('single')}
            className={`px-4 py-1.5 rounded text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
              wordMode === 'single'
                ? 'bg-[#143F62] text-[#CFDDD8]'
                : 'text-[#91A4B0] hover:text-[#CFDDD8]'
            }`}
          >
            Smart Synonym Finder
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={wordMode === 'pair'}
            onClick={() => setWordMode('pair')}
            className={`px-4 py-1.5 rounded text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
              wordMode === 'pair'
                ? 'bg-[#143F62] text-[#CFDDD8]'
                : 'text-[#91A4B0] hover:text-[#CFDDD8]'
            }`}
          >
            Compare Two Words
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={wordMode === 'context'}
            onClick={() => setWordMode('context')}
            className={`px-4 py-1.5 rounded text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
              wordMode === 'context'
                ? 'bg-[#143F62] text-[#CFDDD8]'
                : 'text-[#91A4B0] hover:text-[#CFDDD8]'
            }`}
          >
            Context Sense (WSD)
          </button>
        </div>
      </div>

      {/* MODE 1: Smart Synonym Finder */}
      {wordMode === 'single' && (
        <>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void runSimilarSearch(singleWord, topK);
            }}
            className="rounded-lg bg-[#0B1925]/90 border border-[rgba(207,221,216,0.12)] p-5 sm:p-8"
          >
            <div className="flex flex-col md:flex-row gap-4 items-end">
              <div className="flex-1 w-full">
                <label
                  htmlFor="single-word-input"
                  className="block font-mono text-[11px] tracking-[0.14em] text-[#91A4B0] mb-2"
                >
                  INPUT WORD
                </label>
                <div className="relative">
                  <Search className="w-4 h-4 text-[#647987] absolute left-4 top-1/2 -translate-y-1/2" />
                  <input
                    id="single-word-input"
                    type="text"
                    value={singleWord}
                    onChange={(e) => {
                      setSingleWord(e.target.value);
                      if (singleError) setSingleError(null);
                    }}
                    placeholder="Enter a word (e.g. Computer or Happy)"
                    className="w-full h-12 pl-11 pr-4 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-base font-display text-[#CFDDD8] placeholder:text-[#647987]"
                  />
                </div>
              </div>

              <div className="w-full md:w-36">
                <label className="block font-mono text-[11px] tracking-[0.14em] text-[#91A4B0] mb-2">
                  RESULTS
                </label>
                <div className="grid grid-cols-2 p-1 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] h-12 items-center">
                  <button
                    type="button"
                    onClick={() => {
                      setTopK(5);
                      if (singleResult) void runSimilarSearch(singleWord, 5);
                    }}
                    className={`h-full rounded text-xs font-medium transition-colors cursor-pointer ${
                      topK === 5 ? 'bg-[#143F62] text-[#CFDDD8]' : 'text-[#91A4B0] hover:text-[#CFDDD8]'
                    }`}
                  >
                    Top 5
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setTopK(10);
                      if (singleResult) void runSimilarSearch(singleWord, 10);
                    }}
                    className={`h-full rounded text-xs font-medium transition-colors cursor-pointer ${
                      topK === 10 ? 'bg-[#143F62] text-[#CFDDD8]' : 'text-[#91A4B0] hover:text-[#CFDDD8]'
                    }`}
                  >
                    Top 10
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={singleLoading}
                className="w-full md:w-auto h-12 px-6 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/40 text-sm font-medium transition-colors cursor-pointer disabled:opacity-60 whitespace-nowrap inline-flex items-center justify-center gap-2"
              >
                {singleLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Searching...
                  </>
                ) : (
                  'Find Similar Words'
                )}
              </button>
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-2">
              <span className="text-xs text-[#647987] mr-1">Try:</span>
              {SINGLE_WORD_EXAMPLES.map((w) => (
                <button
                  key={w}
                  type="button"
                  onClick={() => {
                    setSingleWord(w);
                    void runSimilarSearch(w, topK);
                  }}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
                    singleWord.trim().toLowerCase() === w.toLowerCase()
                      ? 'bg-[#143F62] border-[#5BA4D6]/60 text-[#CFDDD8]'
                      : 'bg-[#071522] border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8]'
                  }`}
                >
                  {w}
                </button>
              ))}
            </div>

            {singleError && (
              <div
                role="alert"
                className="mt-4 flex items-center gap-2 p-3 rounded-md bg-[#E17E7E]/10 border border-[#E17E7E]/30 text-sm text-[#E17E7E]"
              >
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{singleError}</span>
              </div>
            )}
          </form>

          {singleResult && !singleLoading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="mt-6 space-y-5"
            >
              <div className="rounded-lg bg-[#0B1925]/90 border border-[rgba(207,221,216,0.12)] p-6 sm:p-8 space-y-6">
                {/* Ranked Header */}
                <div className="flex flex-wrap items-start justify-between gap-3 pb-4 border-b border-[rgba(207,221,216,0.10)]">
                  <div>
                    <h3 className="font-display text-xl sm:text-2xl font-bold text-[#CFDDD8]">
                      Top {singleResult.topK} words similar to &ldquo;{singleResult.preprocessedWord}&rdquo;
                    </h3>
                    {singleResult.inputDefinition && (
                      <p className="text-xs text-[#91A4B0] mt-1">
                        Definition: {singleResult.inputDefinition}
                      </p>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-3 text-xs">
                    <span className="font-mono text-[#65C99A]">
                      {singleResult.exactSynonymCount} Exact Synonym{singleResult.exactSynonymCount === 1 ? '' : 's'}
                    </span>
                    <span className="text-[#647987]">·</span>
                    <span className="font-mono text-[#5BA4D6]">
                      {singleResult.relatedCount} Closely Related
                    </span>
                    {onExploreWord && (
                      <>
                        <span className="text-[#647987]">·</span>
                        <button
                          type="button"
                          onClick={() => onExploreWord(singleResult.preprocessedWord)}
                          className="text-[#5BA4D6] hover:underline cursor-pointer"
                        >
                          View network graph
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {/* Bar Chart Ranking (Cosine 0.xxx + Definitions & Examples) */}
                <div className="space-y-3">
                  {singleResult.results.map((item, idx) => (
                    <button
                      key={item.word}
                      type="button"
                      onClick={() => {
                        setSingleWord(item.word);
                        void runSimilarSearch(item.word, topK);
                      }}
                      className="w-full text-left p-4 rounded-md bg-[#071522]/90 hover:bg-[#143F62]/35 border border-[rgba(207,221,216,0.08)] hover:border-[#5BA4D6]/40 transition-colors cursor-pointer space-y-2"
                    >
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                        <div className="flex items-center gap-3 min-w-0">
                          <span className="font-mono text-sm font-semibold text-[#5BA4D6] w-6 shrink-0 tabular-nums">
                            {idx + 1}.
                          </span>
                          <span className="font-display text-base font-semibold text-[#CFDDD8] truncate">
                            {item.word.toLowerCase()}
                          </span>
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-mono border shrink-0 ${
                              item.matchType === 'Exact Synonym'
                                ? 'bg-[#65C99A]/15 border-[#65C99A]/40 text-[#65C99A]'
                                : 'bg-[#143F62]/60 border-[#5BA4D6]/35 text-[#91A4B0]'
                            }`}
                          >
                            {item.matchType}
                          </span>
                          <span className="font-mono text-[10px] text-[#647987] hidden md:inline">
                            {item.posLabel}
                          </span>
                        </div>

                        <div className="flex items-center gap-3 shrink-0 w-full sm:w-72">
                          <div className="flex-1 h-3 rounded bg-[#010A12] overflow-hidden">
                            <div
                              className={`h-full rounded ${
                                idx === 0
                                  ? 'bg-[#E9B96E]'
                                  : item.cosineSimilarity >= 0.6
                                    ? 'bg-[#65C99A]'
                                    : 'bg-[#5BA4D6]'
                              }`}
                              style={{ width: `${Math.max(4, item.cosineSimilarity * 100)}%` }}
                            />
                          </div>
                          <span
                            className={`font-mono text-sm font-bold tabular-nums w-14 text-right ${
                              idx === 0 ? 'text-[#E9B96E]' : 'text-[#65C99A]'
                            }`}
                          >
                            {item.cosineSimilarity.toFixed(3)}
                          </span>
                        </div>
                      </div>

                      <div className="pl-9 text-xs text-[#91A4B0] leading-relaxed">
                        <span>{item.definition}</span>
                        {item.exampleUsage && (
                          <span className="block text-[#647987] italic mt-0.5">
                            Example: &ldquo;{item.exampleUsage}&rdquo;
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>

                {/* Collapsible 2D Vector Angle Diagram & Pipeline Details */}
                {(singleResult.vectorGeometry || singleResult.pipelineStages) && (
                  <div className="pt-4 border-t border-[rgba(207,221,216,0.08)]">
                    <button
                      type="button"
                      onClick={() => setShowSingleDetails((s) => !s)}
                      className="inline-flex items-center gap-1.5 text-xs text-[#91A4B0] hover:text-[#CFDDD8] cursor-pointer"
                    >
                      {showSingleDetails ? (
                        <ChevronUp className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronDown className="w-3.5 h-3.5" />
                      )}
                      {showSingleDetails
                        ? 'Hide vector geometry & NLP pipeline'
                        : 'Show vector geometry & NLP pipeline'}
                    </button>

                    {showSingleDetails && (
                      <div className="mt-4 space-y-5">
                        {singleResult.vectorGeometry && (
                          <VectorAngleDiagram geometry={singleResult.vectorGeometry} />
                        )}
                        {singleResult.pipelineStages && (
                          <NlpPipelineSteps stages={singleResult.pipelineStages} />
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </>
      )}

      {/* MODE 2: Compare Two Words */}
      {wordMode === 'pair' && (
        <>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void runPairAnalysis(wordA, wordB, algorithm);
            }}
            noValidate
            className="rounded-lg bg-[#0B1925]/90 border border-[rgba(207,221,216,0.12)] p-5 sm:p-8"
          >
            <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-end">
              <div>
                <label
                  htmlFor="word-a-input"
                  className="block font-mono text-[11px] tracking-[0.14em] text-[#91A4B0] mb-2"
                >
                  WORD A
                </label>
                <input
                  id="word-a-input"
                  type="text"
                  value={wordA}
                  onChange={(e) => {
                    setWordA(e.target.value);
                    if (validationError) setValidationError(null);
                  }}
                  placeholder="e.g. Happy"
                  className="w-full h-12 px-4 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-base font-display text-[#CFDDD8] placeholder:text-[#647987]"
                />
              </div>

              <div className="flex justify-center md:pb-0.5">
                <button
                  type="button"
                  onClick={() => {
                    setWordA(wordB);
                    setWordB(wordA);
                  }}
                  aria-label="Swap words"
                  className="inline-flex items-center justify-center w-11 h-11 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.14)] hover:border-[#5BA4D6]/60 text-[#91A4B0] hover:text-[#CFDDD8] transition-colors cursor-pointer"
                >
                  <ArrowLeftRight className="w-4 h-4" />
                </button>
              </div>

              <div>
                <label
                  htmlFor="word-b-input"
                  className="block font-mono text-[11px] tracking-[0.14em] text-[#91A4B0] mb-2"
                >
                  WORD B
                </label>
                <input
                  id="word-b-input"
                  type="text"
                  value={wordB}
                  onChange={(e) => {
                    setWordB(e.target.value);
                    if (validationError) setValidationError(null);
                  }}
                  placeholder="e.g. Joyful"
                  className="w-full h-12 px-4 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-base font-display text-[#CFDDD8] placeholder:text-[#647987]"
                />
              </div>
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-2">
              <span className="text-xs text-[#647987] mr-1">Try:</span>
              {EXAMPLE_PAIRS.map((pair) => {
                const isSelected =
                  wordA.trim().toLowerCase() === pair.a.toLowerCase() &&
                  wordB.trim().toLowerCase() === pair.b.toLowerCase();
                return (
                  <button
                    key={`${pair.a}-${pair.b}`}
                    type="button"
                    onClick={() => {
                      setWordA(pair.a);
                      setWordB(pair.b);
                      void runPairAnalysis(pair.a, pair.b, algorithm);
                    }}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
                      isSelected
                        ? 'bg-[#143F62] border-[#5BA4D6]/60 text-[#CFDDD8]'
                        : 'bg-[#071522] border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8]'
                    }`}
                  >
                    {pair.a} – {pair.b}
                  </button>
                );
              })}
            </div>

            <div className="mt-6 pt-5 border-t border-[rgba(207,221,216,0.10)] flex flex-col sm:flex-row sm:items-end justify-between gap-4">
              <div className="w-full sm:w-72">
                <label
                  htmlFor="algorithm-select"
                  className="block font-mono text-[11px] tracking-[0.12em] text-[#91A4B0] mb-1.5"
                >
                  ALGORITHM
                </label>
                <select
                  id="algorithm-select"
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value as AlgorithmChoice)}
                  className="w-full h-10 px-3 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-sm text-[#CFDDD8] cursor-pointer"
                >
                  {ALGORITHM_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => {
                    setWordA('');
                    setWordB('');
                    setResult(null);
                    setValidationError(null);
                    setApiError(null);
                  }}
                  className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md bg-[#071522] hover:bg-[#010A12] text-[#91A4B0] hover:text-[#CFDDD8] border border-[rgba(207,221,216,0.14)] text-sm font-medium transition-colors cursor-pointer"
                >
                  <RotateCcw className="w-4 h-4" />
                  Clear
                </button>

                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/40 text-sm font-medium transition-colors cursor-pointer disabled:opacity-60"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    'Analyze Similarity'
                  )}
                </button>
              </div>
            </div>

            {validationError && (
              <div
                role="alert"
                className="mt-4 flex items-center gap-2 p-3 rounded-md bg-[#E17E7E]/10 border border-[#E17E7E]/30 text-sm text-[#E17E7E]"
              >
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{validationError}</span>
              </div>
            )}

            {apiError && (
              <div
                role="alert"
                className="mt-4 flex items-center gap-2 p-3 rounded-md bg-[#E17E7E]/10 border border-[#E17E7E]/30 text-sm text-[#E17E7E]"
              >
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{apiError}</span>
              </div>
            )}
          </form>

          {result && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="mt-6 space-y-5"
            >
              <div className="rounded-lg bg-[#0B1925]/90 border border-[rgba(207,221,216,0.12)] p-6 sm:p-8 space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-5 border-b border-[rgba(207,221,216,0.10)]">
                  <h3 className="font-display text-xl sm:text-2xl font-bold text-[#CFDDD8]">
                    {result.word_a} <span className="text-[#5BA4D6] mx-1">↔</span> {result.word_b}
                  </h3>
                  <div className="flex items-center gap-3 text-xs text-[#91A4B0]">
                    <span className="text-[#CFDDD8] font-medium">{result.relationship}</span>
                    {onExploreWord && (
                      <>
                        <span aria-hidden="true">·</span>
                        <button
                          type="button"
                          onClick={() => onExploreWord(result.word_a)}
                          className="text-[#5BA4D6] hover:underline cursor-pointer"
                        >
                          View in network
                        </button>
                      </>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
                  <div className="lg:col-span-4 flex flex-col items-center text-center">
                    <SimilarityGauge score={result.primaryScore} label={result.primaryAlgorithmUsed} />
                    {result.interpretationBand && (
                      <span className="text-sm font-medium text-[#CFDDD8] mt-3">
                        {result.interpretationBand.label} ({result.interpretationBand.range})
                      </span>
                    )}
                  </div>

                  <div className="lg:col-span-8 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                      <div className="p-3.5 rounded bg-[#071522]/80 border border-[rgba(207,221,216,0.08)]">
                        <div className="font-mono text-[10px] text-[#5BA4D6] mb-1">
                          BEGINNER EXPLANATION
                        </div>
                        <p className="text-[#CFDDD8] leading-relaxed">{result.beginnerExplanation}</p>
                      </div>
                      <div className="p-3.5 rounded bg-[#071522]/80 border border-[rgba(207,221,216,0.08)]">
                        <div className="font-mono text-[10px] text-[#91A4B0] mb-1">
                          TECHNICAL EXPLANATION
                        </div>
                        <p className="text-[#91A4B0] leading-relaxed">{result.technicalExplanation}</p>
                      </div>
                    </div>

                    {/* 5 Algorithm Score Bars + Timing + Category */}
                    <div className="space-y-3 pt-3 border-t border-[rgba(207,221,216,0.10)]">
                      {orderedResults.map((item) => (
                        <div key={item.algorithm} className="space-y-1">
                          <div className="flex items-center justify-between text-xs gap-2">
                            <span className="text-[#CFDDD8] font-medium">
                              {item.name}{' '}
                              <span className="text-[#647987] font-normal">
                                · {item.methodCategory}
                                {item.processingTimeMs !== undefined ? ` · ${item.processingTimeMs}ms` : ''}
                              </span>
                            </span>
                            <span className="font-mono tabular-nums text-[#CFDDD8]">
                              {item.score !== null ? `${item.score.toFixed(1)}%` : 'Unavailable'}
                            </span>
                          </div>
                          <div className="w-full h-1.5 rounded-full bg-[#071522] overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                item.score === null
                                  ? 'bg-transparent'
                                  : item.score >= 60
                                    ? 'bg-[#65C99A]'
                                    : item.score >= 40
                                      ? 'bg-[#5BA4D6]'
                                      : 'bg-[#E9B96E]'
                              }`}
                              style={{
                                width: `${item.score !== null ? Math.max(2, item.score) : 0}%`,
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {result.vectorGeometry && <VectorAngleDiagram geometry={result.vectorGeometry} />}

                {/* Unified Comparison Engine Details */}
                <div className="pt-4 border-t border-[rgba(207,221,216,0.08)]">
                  <button
                    type="button"
                    onClick={() => setShowTechnical((s) => !s)}
                    className="inline-flex items-center gap-1.5 text-xs text-[#91A4B0] hover:text-[#CFDDD8] cursor-pointer"
                  >
                    {showTechnical ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    {showTechnical
                      ? 'Hide algorithm breakdown & limitations'
                      : 'Show algorithm breakdown, methods & limitations'}
                  </button>

                  {showTechnical && (
                    <div className="mt-4 space-y-3">
                      <p className="text-[11px] text-[#91A4B0]">
                        Note: Scores from different algorithms measure different linguistic dimensions (lexical taxonomy, token frequency, distributional co-occurrence, subwords, and contextual attention) and are not directly equivalent scales.
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                        {orderedResults.map((item) => (
                          <div
                            key={item.algorithm}
                            className="p-3.5 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.08)] space-y-1.5"
                          >
                            <div className="flex justify-between font-medium text-[#CFDDD8]">
                              <span>
                                {item.name}{' '}
                                <span className="text-[#647987] font-normal">({item.methodCategory})</span>
                              </span>
                              <span className="font-mono text-[#5BA4D6]">
                                {item.rawCosineOrMetric !== null ? item.rawCosineOrMetric.toFixed(3) : 'N/A'}
                              </span>
                            </div>
                            {item.calculationMethod && (
                              <div className="font-mono text-[10px] text-[#5BA4D6]">
                                Method: {item.calculationMethod}
                              </div>
                            )}
                            <p className="text-[#CFDDD8] leading-relaxed">
                              {item.status === 'not_available' && item.reasonIfUnavailable
                                ? item.reasonIfUnavailable
                                : item.explanation}
                            </p>
                            {item.limitation && (
                              <p className="text-[11px] text-[#E9B96E]/90 leading-relaxed">
                                Limitation: {item.limitation}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {result.pipelineStages && <NlpPipelineSteps stages={result.pipelineStages} />}
            </motion.div>
          )}
        </>
      )}

      {/* MODE 3: Contextual Word Detection (WSD) */}
      {wordMode === 'context' && <ContextDetector embedded />}
    </section>
  );
}
