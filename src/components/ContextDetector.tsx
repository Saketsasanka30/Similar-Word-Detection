import { AlertCircle, Loader2, RotateCcw } from 'lucide-react';
import { motion } from 'motion/react';
import { useState } from 'react';
import type { ContextAnalysisResponse } from '../types/nlp';
import { NlpPipelineSteps } from './NlpPipelineSteps';

const CONTEXT_EXAMPLES = [
  {
    label: 'Bank (Finance vs River)',
    word: 'Bank',
    s1: 'I deposited money at the bank.',
    s2: 'We sat beside the river bank.',
  },
  {
    label: 'Apple (Fruit vs Tech)',
    word: 'Apple',
    s1: 'She sliced a crisp red apple for lunch.',
    s2: 'Apple announced a faster laptop processor.',
  },
  {
    label: 'Python (Snake vs Code)',
    word: 'Python',
    s1: 'The python coiled around a mossy tree branch.',
    s2: 'We wrote the NLP pipeline in Python.',
  },
  {
    label: 'Crane (Machine vs Bird)',
    word: 'Crane',
    s1: 'A tall crane lifted steel beams onto the roof.',
    s2: 'The white crane waded across the shallow lake.',
  },
];

export function ContextDetector({ embedded = false }: { embedded?: boolean }) {
  const [word, setWord] = useState('Bank');
  const [sentence1, setSentence1] = useState('I deposited money at the bank.');
  const [sentence2, setSentence2] = useState('We sat beside the river bank.');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ContextAnalysisResponse | null>(null);

  const runDetection = async (targetWord: string, s1: string, s2: string) => {
    if (!targetWord.trim() || !s1.trim()) {
      setError('Enter a target word and at least Sentence 1.');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await fetch('/api/analyze/context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          word: targetWord.trim(),
          sentence_a: s1.trim(),
          sentence_b: s2.trim() || undefined,
        }),
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload?.error || 'Contextual word detection failed.');
      setResult(payload as ContextAnalysisResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to detect contextual sense.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section
      className={embedded ? '' : 'max-w-[1200px] mx-auto px-4 sm:px-8 py-14 sm:py-20'}
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void runDetection(word, sentence1, sentence2);
        }}
        className="rounded-lg bg-[#0B1925]/90 border border-[rgba(207,221,216,0.12)] p-5 sm:p-8 space-y-5"
      >
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
          <div className="md:col-span-3">
            <label
              htmlFor="context-target-word"
              className="block font-mono text-[11px] tracking-[0.14em] text-[#91A4B0] mb-2"
            >
              TARGET WORD
            </label>
            <input
              id="context-target-word"
              type="text"
              value={word}
              onChange={(e) => setWord(e.target.value)}
              placeholder="e.g. Bank"
              className="w-full h-11 px-3.5 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-sm font-display text-[#CFDDD8]"
            />
          </div>
          <div className="md:col-span-9 flex flex-wrap items-end gap-2">
            <span className="text-xs text-[#647987] mr-1 mb-2">Examples:</span>
            {CONTEXT_EXAMPLES.map((ex) => (
              <button
                key={ex.label}
                type="button"
                onClick={() => {
                  setWord(ex.word);
                  setSentence1(ex.s1);
                  setSentence2(ex.s2);
                  void runDetection(ex.word, ex.s1, ex.s2);
                }}
                className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
                  word.trim().toLowerCase() === ex.word.toLowerCase()
                    ? 'bg-[#143F62] border-[#5BA4D6]/60 text-[#CFDDD8]'
                    : 'bg-[#071522] border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8]'
                }`}
              >
                {ex.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label
              htmlFor="context-sentence-1"
              className="block font-mono text-[11px] tracking-[0.14em] text-[#91A4B0] mb-2"
            >
              SENTENCE 1
            </label>
            <input
              id="context-sentence-1"
              type="text"
              value={sentence1}
              onChange={(e) => setSentence1(e.target.value)}
              placeholder="e.g. I deposited money at the bank."
              className="w-full h-11 px-3.5 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-sm text-[#CFDDD8]"
            />
          </div>
          <div>
            <label
              htmlFor="context-sentence-2"
              className="block font-mono text-[11px] tracking-[0.14em] text-[#91A4B0] mb-2"
            >
              SENTENCE 2 (OPTIONAL COMPARISON)
            </label>
            <input
              id="context-sentence-2"
              type="text"
              value={sentence2}
              onChange={(e) => setSentence2(e.target.value)}
              placeholder="e.g. We sat beside the river bank."
              className="w-full h-11 px-3.5 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-sm text-[#CFDDD8]"
            />
          </div>
        </div>

        <div className="flex justify-end gap-3 pt-2 border-t border-[rgba(207,221,216,0.10)]">
          <button
            type="button"
            onClick={() => {
              setWord('');
              setSentence1('');
              setSentence2('');
              setResult(null);
              setError(null);
            }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-[#071522] hover:bg-[#010A12] text-[#91A4B0] hover:text-[#CFDDD8] border border-[rgba(207,221,216,0.14)] text-xs font-medium cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Clear
          </button>
          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center gap-2 px-5 py-2 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/40 text-xs font-medium cursor-pointer disabled:opacity-60"
          >
            {loading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Disambiguating Sense...
              </>
            ) : (
              'Detect Contextual Meaning'
            )}
          </button>
        </div>

        {error && (
          <div
            role="alert"
            className="flex items-center gap-2 p-3 rounded-md bg-[#E17E7E]/10 border border-[#E17E7E]/30 text-sm text-[#E17E7E]"
          >
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
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
          <div className="rounded-lg bg-[#0B1925]/90 border border-[rgba(207,221,216,0.12)] p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-[rgba(207,221,216,0.10)]">
              <div>
                <p className="font-mono text-[10px] tracking-wider text-[#5BA4D6]">
                  CONTEXTUAL SENSE RESOLUTION · {result.totalSenses} WORDNET SENSES EVALUATED
                </p>
                <p className="text-sm sm:text-base text-[#CFDDD8] font-medium mt-1">
                  {result.senseComparisonSummary}
                </p>
              </div>
              {result.sameSenseDetected !== null && (
                <span
                  className={`px-3 py-1 rounded text-xs font-mono border ${
                    result.sameSenseDetected
                      ? 'bg-[#65C99A]/15 border-[#65C99A]/40 text-[#65C99A]'
                      : 'bg-[#E9B96E]/15 border-[#E9B96E]/40 text-[#E9B96E]'
                  }`}
                >
                  {result.sameSenseDetected ? 'Same Meaning' : 'Different Meanings'}
                </span>
              )}
            </div>

            {result.found && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                {[result.sentence1, result.sentence2]
                  .filter((item): item is NonNullable<typeof item> => item !== null)
                  .map((sentRes, idx) => (
                    <div
                      key={idx}
                      className="rounded-md bg-[#071522]/90 border border-[rgba(207,221,216,0.10)] p-5 space-y-4"
                    >
                      <div>
                        <div className="font-mono text-[10px] text-[#5BA4D6]">
                          SENTENCE {idx + 1}
                        </div>
                        <p className="text-sm text-[#CFDDD8] mt-1">&ldquo;{sentRes.sentence}&rdquo;</p>
                      </div>

                      <div className="p-3.5 rounded bg-[#010A12]/80 border border-[#65C99A]/35">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-[10px] text-[#65C99A]">
                            DETECTED SENSE ({sentRes.detectedSense.posLabel.toUpperCase()})
                          </span>
                          <span className="font-mono text-xs text-[#65C99A] tabular-nums">
                            {sentRes.detectedSense.combinedScore.toFixed(3)}
                          </span>
                        </div>
                        <p className="text-sm font-medium text-[#CFDDD8] mt-1">
                          {sentRes.detectedSense.gloss}
                        </p>
                        {sentRes.detectedSense.exampleUsage && (
                          <p className="text-xs text-[#91A4B0] italic mt-1">
                            e.g. &ldquo;{sentRes.detectedSense.exampleUsage}&rdquo;
                          </p>
                        )}
                      </div>

                      {sentRes.alternativeSenses.length > 0 && (
                        <div className="space-y-1.5">
                          <div className="font-mono text-[10px] text-[#647987]">
                            ALTERNATIVE MEANINGS
                          </div>
                          {sentRes.alternativeSenses.slice(0, 3).map((alt) => (
                            <div
                              key={alt.synsetId}
                              className="flex items-start justify-between gap-3 text-xs p-2 rounded bg-[#010A12]/50 border border-[rgba(207,221,216,0.06)]"
                            >
                              <span className="text-[#91A4B0] leading-snug">
                                <span className="text-[#CFDDD8] font-mono mr-1">
                                  [{alt.posLabel}]
                                </span>
                                {alt.gloss}
                              </span>
                              <span className="font-mono text-[#647987] tabular-nums shrink-0">
                                {alt.combinedScore.toFixed(3)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 text-xs">
              <div className="p-3.5 rounded bg-[#071522]/70 border border-[rgba(207,221,216,0.08)]">
                <div className="font-mono text-[10px] text-[#5BA4D6] mb-1">BEGINNER EXPLANATION</div>
                <p className="text-[#CFDDD8] leading-relaxed">{result.beginnerExplanation}</p>
              </div>
              <div className="p-3.5 rounded bg-[#071522]/70 border border-[rgba(207,221,216,0.08)]">
                <div className="font-mono text-[10px] text-[#91A4B0] mb-1">TECHNICAL EXPLANATION</div>
                <p className="text-[#91A4B0] leading-relaxed">{result.technicalExplanation}</p>
              </div>
            </div>
          </div>

          <NlpPipelineSteps stages={result.pipelineStages} />
        </motion.div>
      )}
    </section>
  );
}
