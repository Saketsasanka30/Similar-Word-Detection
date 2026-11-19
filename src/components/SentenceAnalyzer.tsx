import { AlertCircle, ChevronDown, ChevronUp, Loader2, RotateCcw } from 'lucide-react';
import { motion } from 'motion/react';
import { useEffect, useState } from 'react';
import type { AlgorithmResult, SentenceAnalysisResponse } from '../types/nlp';
import { NlpPipelineSteps } from './NlpPipelineSteps';
import { SimilarityGauge } from './SimilarityGauge';
import { VectorAngleDiagram } from './VectorAngleDiagram';

interface SentenceAnalyzerProps {
  onAnalysisComplete: (result: SentenceAnalysisResponse) => void;
  restoredResult?: SentenceAnalysisResponse | null;
  embedded?: boolean;
}

const SAMPLE_SENTENCES: Array<{ label: string; a: string; b: string }> = [
  {
    label: 'Healthcare AI (Paraphrase)',
    a: 'Artificial intelligence is transforming healthcare.',
    b: 'AI is changing the medical industry.',
  },
  {
    label: 'Climate & Oceans',
    a: 'Rising ocean temperatures disrupt marine ecosystems.',
    b: 'Warmer seas threaten aquatic biodiversity.',
  },
  {
    label: 'Autonomous Transit',
    a: 'Self-driving electric cars reduce urban traffic emissions.',
    b: 'Autonomous battery vehicles help lower city pollution.',
  },
  {
    label: 'Unrelated Topics',
    a: 'Astronomers detected water vapor on a distant exoplanet.',
    b: 'The orchestra rehearsed Beethoven before Friday evening.',
  },
];

export function SentenceAnalyzer({
  onAnalysisComplete,
  restoredResult,
  embedded = false,
}: SentenceAnalyzerProps) {
  const [sentenceA, setSentenceA] = useState(
    'Artificial intelligence is transforming healthcare.'
  );
  const [sentenceB, setSentenceB] = useState(
    'AI is changing the medical industry.'
  );
  const [validationError, setValidationError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SentenceAnalysisResponse | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    if (restoredResult) {
      setSentenceA(restoredResult.sentence_a);
      setSentenceB(restoredResult.sentence_b);
      setResult(restoredResult);
      setValidationError(null);
      setApiError(null);
    }
  }, [restoredResult]);

  const runCompare = async (aVal: string, bVal: string) => {
    const cleanA = aVal.trim();
    const cleanB = bVal.trim();
    if (!cleanA || cleanA.length < 3) {
      setValidationError('Sentence A must contain at least 3 characters.');
      return;
    }
    if (!cleanB || cleanB.length < 3) {
      setValidationError('Sentence B must contain at least 3 characters.');
      return;
    }

    setValidationError(null);
    setApiError(null);
    setLoading(true);

    try {
      const response = await fetch('/api/analyze/sentence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence_a: cleanA, sentence_b: cleanB }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.error || 'Failed to compare sentences.');
      }
      setResult(payload as SentenceAnalysisResponse);
      onAnalysisComplete(payload as SentenceAnalysisResponse);
    } catch (err) {
      setApiError(
        err instanceof Error
          ? err.message
          : 'Unable to reach sentence analysis service.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void runCompare(sentenceA, sentenceB);
  };

  const handleClear = () => {
    setSentenceA('');
    setSentenceB('');
    setResult(null);
    setValidationError(null);
    setApiError(null);
  };

  const algoList: AlgorithmResult[] = result
    ? [
        result.results.transformer,
        result.results.fasttext,
        result.results.word2vec,
        result.results.tfidf,
      ].filter((a): a is AlgorithmResult => Boolean(a))
    : [];

  return (
    <section
      className={embedded ? '' : 'max-w-[1200px] mx-auto px-4 sm:px-8 py-14 sm:py-20'}
    >
      {!embedded && (
        <div className="mb-8">
          <p className="font-mono text-[11px] tracking-[0.16em] text-[#5BA4D6] mb-2">
            SENTENCE SIMILARITY
          </p>
          <h2 className="font-display font-bold text-2xl sm:text-4xl tracking-tight text-[#CFDDD8]">
            Compare meaning, not just words.
          </h2>
          <p className="text-sm sm:text-base text-[#91A4B0] mt-2">
            See how closely two sentences align in meaning across contextual, subword, static, and statistical NLP algorithms.
          </p>
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-5 sm:p-8"
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <label
              htmlFor="sentence-a-input"
              className="block font-mono text-[11px] tracking-[0.14em] text-[#91A4B0] mb-2"
            >
              SENTENCE A
            </label>
            <textarea
              id="sentence-a-input"
              rows={3}
              value={sentenceA}
              onChange={(e) => {
                setSentenceA(e.target.value);
                if (validationError) setValidationError(null);
              }}
              placeholder="Enter the first sentence..."
              className="w-full p-4 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-sm sm:text-base text-[#CFDDD8] placeholder:text-[#647987] resize-y"
            />
          </div>

          <div>
            <label
              htmlFor="sentence-b-input"
              className="block font-mono text-[11px] tracking-[0.14em] text-[#91A4B0] mb-2"
            >
              SENTENCE B
            </label>
            <textarea
              id="sentence-b-input"
              rows={3}
              value={sentenceB}
              onChange={(e) => {
                setSentenceB(e.target.value);
                if (validationError) setValidationError(null);
              }}
              placeholder="Enter the second sentence..."
              className="w-full p-4 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-sm sm:text-base text-[#CFDDD8] placeholder:text-[#647987] resize-y"
            />
          </div>
        </div>

        {/* Sample Sentence Pairs */}
        <div className="mt-5 flex flex-wrap items-center gap-2">
          <span className="font-mono text-[11px] tracking-wider text-[#647987] mr-1">
            Sample pairs:
          </span>
          {SAMPLE_SENTENCES.map((sample) => (
            <button
              key={sample.label}
              type="button"
              onClick={() => {
                setSentenceA(sample.a);
                setSentenceB(sample.b);
                void runCompare(sample.a, sample.b);
              }}
              className="px-3 py-1.5 rounded-md text-xs font-medium bg-[#071522] border border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8] hover:border-[rgba(207,221,216,0.28)] transition-colors cursor-pointer"
            >
              {sample.label}
            </button>
          ))}
        </div>

        <div className="mt-6 pt-6 border-t border-[rgba(207,221,216,0.10)] flex flex-wrap items-center justify-end gap-3">
          <button
            type="button"
            onClick={handleClear}
            disabled={loading}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md bg-[#071522] hover:bg-[#010A12] text-[#91A4B0] hover:text-[#CFDDD8] border border-[rgba(207,221,216,0.14)] text-sm font-medium transition-colors cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" />
            Clear
          </button>

          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center gap-2.5 px-6 py-2.5 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/40 text-sm font-medium transition-colors cursor-pointer disabled:opacity-60"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Comparing...
              </>
            ) : (
              'Compare Sentences'
            )}
          </button>
        </div>

        {validationError && (
          <div
            role="alert"
            className="mt-4 flex items-start gap-2.5 p-3.5 rounded-md bg-[#E17E7E]/10 border border-[#E17E7E]/30 text-sm text-[#E17E7E]"
          >
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{validationError}</span>
          </div>
        )}

        {apiError && (
          <div
            role="alert"
            className="mt-4 flex items-start gap-2.5 p-3.5 rounded-md bg-[#E17E7E]/10 border border-[#E17E7E]/30 text-sm text-[#E17E7E]"
          >
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{apiError}</span>
          </div>
        )}
      </form>

      {result && !loading && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
          className="mt-8 space-y-6"
        >
          <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-6 sm:p-8 space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
              <div className="lg:col-span-4 flex flex-col items-center text-center">
                <SimilarityGauge
                  score={result.primaryScore}
                  label="Sentence-BERT Similarity"
                />
                {result.interpretationBand && (
                  <p className="text-sm font-medium text-[#CFDDD8] mt-3">
                    {result.interpretationBand.label} ({result.interpretationBand.range})
                  </p>
                )}
              </div>

              <div className="lg:col-span-8 space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <div className="p-3.5 rounded bg-[#071522]/80 border border-[rgba(207,221,216,0.08)]">
                    <div className="font-mono text-[10px] text-[#5BA4D6] mb-1">
                      BEGINNER EXPLANATION
                    </div>
                    <p className="text-[#CFDDD8] leading-relaxed">
                      {result.beginnerExplanation}
                    </p>
                  </div>
                  <div className="p-3.5 rounded bg-[#071522]/80 border border-[rgba(207,221,216,0.08)]">
                    <div className="font-mono text-[10px] text-[#65C99A] mb-1">
                      TECHNICAL EXPLANATION
                    </div>
                    <p className="text-[#91A4B0] leading-relaxed">
                      {result.technicalExplanation || result.explanation}
                    </p>
                  </div>
                </div>

                {/* Algorithm Comparison Bars */}
                <div className="space-y-3 pt-4 border-t border-[rgba(207,221,216,0.10)]">
                  {algoList.map((algo) => (
                    <div key={algo.algorithm} className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-[#CFDDD8]">
                          {algo.name}{' '}
                          <span className="text-[#647987] font-normal">
                            · {algo.methodCategory}
                            {algo.processingTimeMs !== undefined ? ` · ${algo.processingTimeMs}ms` : ''}
                          </span>
                        </span>
                        <span className="font-mono tabular-nums font-semibold text-[#CFDDD8]">
                          {algo.score !== null ? `${algo.score.toFixed(1)}%` : 'Unavailable'}
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-[#071522] overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            algo.algorithm === 'transformer'
                              ? 'bg-[#65C99A]'
                              : algo.algorithm === 'fasttext' || algo.algorithm === 'word2vec'
                                ? 'bg-[#5BA4D6]'
                                : 'bg-[#E9B96E]'
                          }`}
                          style={{
                            width: `${algo.score !== null ? Math.max(2, algo.score) : 0}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {result.vectorGeometry && <VectorAngleDiagram geometry={result.vectorGeometry} />}

            <div className="pt-4 border-t border-[rgba(207,221,216,0.08)]">
              <button
                type="button"
                onClick={() => setShowDetails((s) => !s)}
                className="inline-flex items-center gap-1.5 text-xs text-[#91A4B0] hover:text-[#CFDDD8] cursor-pointer"
              >
                {showDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                {showDetails ? 'Hide algorithm methods & limitations' : 'Show algorithm methods & limitations'}
              </button>

              {showDetails && (
                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  {algoList.map((algo) => (
                    <div
                      key={algo.algorithm}
                      className="p-3.5 rounded bg-[#071522] border border-[rgba(207,221,216,0.08)] space-y-1"
                    >
                      <div className="flex justify-between font-medium text-[#CFDDD8]">
                        <span>{algo.name}</span>
                        <span className="font-mono text-[#5BA4D6]">
                          {algo.rawCosineOrMetric !== null ? algo.rawCosineOrMetric.toFixed(3) : 'N/A'}
                        </span>
                      </div>
                      <p className="text-[#91A4B0]">{algo.explanation}</p>
                      {algo.limitation && (
                        <p className="text-[11px] text-[#E9B96E]/90">Limitation: {algo.limitation}</p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Shared Concepts & Wording Differences */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-5">
              <h4 className="font-mono text-xs tracking-wider text-[#65C99A] mb-3">
                SHARED NORMALIZED TOKENS
              </h4>
              {result.sharedConcepts.length > 0 ? (
                <p className="font-mono text-sm text-[#CFDDD8] leading-relaxed">
                  {result.sharedConcepts.join(' · ')}
                </p>
              ) : (
                <p className="text-xs text-[#91A4B0] leading-relaxed">
                  No identical surface content words shared — similarity is bridged semantically by Sentence-BERT.
                </p>
              )}
            </div>

            <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-5">
              <h4 className="font-mono text-xs tracking-wider text-[#91A4B0] mb-3">
                UNIQUE WORDING — SENTENCE A
              </h4>
              <p className="font-mono text-sm text-[#91A4B0] leading-relaxed">
                {result.uniqueToA.length > 0 ? result.uniqueToA.join(' · ') : 'None'}
              </p>
            </div>

            <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-5">
              <h4 className="font-mono text-xs tracking-wider text-[#91A4B0] mb-3">
                UNIQUE WORDING — SENTENCE B
              </h4>
              <p className="font-mono text-sm text-[#91A4B0] leading-relaxed">
                {result.uniqueToB.length > 0 ? result.uniqueToB.join(' · ') : 'None'}
              </p>
            </div>
          </div>

          {result.pipelineStages && <NlpPipelineSteps stages={result.pipelineStages} />}
        </motion.div>
      )}
    </section>
  );
}
