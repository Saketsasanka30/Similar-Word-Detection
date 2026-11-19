import { AlertCircle, Download, FileText, Loader2, Upload } from 'lucide-react';
import { motion } from 'motion/react';
import { useMemo, useRef, useState } from 'react';
import type { CorpusPredictionResponse, VectorGeometry2D } from '../types/nlp';
import { VectorAngleDiagram } from './VectorAngleDiagram';

const DEFAULT_TRAINING_CORPUS: string[] = [
  'computer technology software system data network',
  'computer software technology programming data system',
  'computer programming software code technology system',
  'computer network data internet technology system',
  'software programming code application computer technology',
  'data database analytics machine learning computer',
  'technology internet network computer digital system',
  'system software hardware computer network',
  'programming code software developer computer',
  'machine learning data analytics artificial intelligence',
  'database data storage system software',
  'internet network communication digital technology',
  'cybersecurity network system data security',
  'artificial intelligence machine learning data technology',
  'web application software programming computer system',
  'mobile application software technology computer',
  'cloud computing data storage network technology',
  'computer science programming algorithms data software',
];

function parseCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current.trim());
  return result;
}

function parseCorpusLinesFromCsv(content: string): string[] {
  const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return [];
  const firstCols = parseCsvLine(lines[0]);
  const hasHeader =
    firstCols.some((c) => /^(id|title|label|text|content|sentence|document|description)$/i.test(c)) &&
    lines.length > 1;
  const startIdx = hasHeader ? 1 : 0;

  let textColIdx = 0;
  if (hasHeader) {
    const foundText = firstCols.findIndex((c) =>
      /^(text|content|sentence|document|description|body)$/i.test(c)
    );
    if (foundText !== -1) textColIdx = foundText;
    else textColIdx = firstCols.length > 1 ? 1 : 0;
  } else if (firstCols.length >= 2) {
    textColIdx = 1;
  }

  return lines
    .slice(startIdx, startIdx + 500)
    .map((line) => {
      const cols = parseCsvLine(line);
      return (cols[textColIdx] ?? cols.join(' ')).trim();
    })
    .filter((t) => /[a-zA-Z]/.test(t));
}

function buildGeometryForSelectedMatch(
  baseGeometry: VectorGeometry2D | undefined,
  targetWord: string,
  matchWord: string,
  similarity: number
): VectorGeometry2D | undefined {
  if (!baseGeometry) return undefined;
  if (baseGeometry.secondaryLabel === matchWord) return baseGeometry;

  const clamped = Math.max(-1, Math.min(1, similarity));
  const thetaRad = Math.acos(clamped);
  const thetaDegrees = Math.round(((thetaRad * 180) / Math.PI) * 10) / 10;
  const baseAngle = (52 * Math.PI) / 180;
  const angleB = baseAngle - thetaRad;

  const primaryPoint = baseGeometry.points.find((p) => p.role === 'primary');
  const contrastPoint = baseGeometry.points.find((p) => p.role === 'contrast');

  return {
    ...baseGeometry,
    primaryLabel: targetWord,
    secondaryLabel: matchWord,
    cosineSimilarity: Math.round(clamped * 1000) / 1000,
    thetaDegrees,
    points: [
      primaryPoint ?? {
        label: targetWord,
        x: Math.round(Math.cos(baseAngle) * 1000) / 1000,
        y: Math.round(Math.sin(baseAngle) * 1000) / 1000,
        role: 'primary',
        cosineWithPrimary: 1,
      },
      {
        label: matchWord,
        x: Math.round(Math.cos(angleB) * 1000) / 1000,
        y: Math.round(Math.sin(angleB) * 1000) / 1000,
        role: 'secondary',
        cosineWithPrimary: Math.round(clamped * 1000) / 1000,
      },
      ...(contrastPoint ? [contrastPoint] : []),
    ],
  };
}

export function DatasetAnalyzer({ embedded = false }: { embedded?: boolean }) {
  const corpusFileRef = useRef<HTMLInputElement | null>(null);

  const [corpusSentences, setCorpusSentences] = useState<string[]>(DEFAULT_TRAINING_CORPUS);
  const [corpusFileName, setCorpusFileName] = useState<string>(
    'default_training_corpus.txt (18 sentences)'
  );
  const [corpusWord, setCorpusWord] = useState('computer');
  const [corpusTopK, setCorpusTopK] = useState<5 | 10>(5);
  const [corpusLoading, setCorpusLoading] = useState(false);
  const [corpusError, setCorpusError] = useState<string | null>(null);
  const [corpusResult, setCorpusResult] = useState<CorpusPredictionResponse | null>(null);
  const [selectedGraphWord, setSelectedGraphWord] = useState<string | null>(null);
  const [isCorpusDragging, setIsCorpusDragging] = useState(false);

  const handleCorpusFileSelected = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setCorpusError(null);
    setCorpusResult(null);
    setSelectedGraphWord(null);
    try {
      const file = files[0];
      const text = await file.text();
      const ext = file.name.split('.').pop()?.toLowerCase() ?? 'txt';
      let lines: string[] = [];

      if (ext === 'json') {
        const parsed = JSON.parse(text);
        const items = Array.isArray(parsed)
          ? parsed
          : Array.isArray(parsed?.documents)
            ? parsed.documents
            : Array.isArray(parsed?.corpus)
              ? parsed.corpus
              : [];
        lines = items
          .map((item: unknown) => {
            if (typeof item === 'string') return item.trim();
            if (item && typeof item === 'object') {
              const obj = item as Record<string, unknown>;
              const val = obj.text ?? obj.content ?? obj.sentence ?? obj.body ?? '';
              return typeof val === 'string' ? val.trim() : '';
            }
            return '';
          })
          .filter(Boolean);
      } else if (ext === 'csv') {
        lines = parseCorpusLinesFromCsv(text);
      } else {
        lines = text
          .split(/\r?\n/)
          .map((l) => l.trim())
          .filter((l) => /[a-zA-Z]/.test(l));
      }

      if (lines.length === 0) {
        setCorpusError(`No readable sentences found in "${file.name}".`);
        return;
      }

      const bounded = lines.slice(0, 500);
      setCorpusSentences(bounded);
      setCorpusFileName(`${file.name} (${bounded.length} lines)`);
    } catch {
      setCorpusError('Could not read file. Upload a valid UTF-8 .txt, .csv, or .json corpus.');
    }
  };

  const handleRunCorpusPredict = async (
    targetWord = corpusWord,
    limit: 5 | 10 = corpusTopK,
    sentences = corpusSentences
  ) => {
    if (!targetWord.trim()) {
      setCorpusError('Please enter a target input word.');
      return;
    }
    if (sentences.length === 0) {
      setCorpusError('Please upload a text corpus or load the default training corpus.');
      return;
    }
    setCorpusError(null);
    setCorpusLoading(true);
    try {
      const res = await fetch('/api/corpus/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          word: targetWord.trim(),
          corpus: sentences,
          top_k: limit,
          window: 2,
        }),
      });
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload?.error || 'Corpus model training failed.');
      }
      const typed = payload as CorpusPredictionResponse;
      setCorpusResult(typed);
      setSelectedGraphWord(typed.results[0]?.word ?? null);
    } catch (err) {
      setCorpusError(err instanceof Error ? err.message : 'Unable to train model.');
    } finally {
      setCorpusLoading(false);
    }
  };

  const handleExportCorpusCsv = () => {
    if (!corpusResult || !corpusResult.found) return;
    const header = 'Rank,Input Word,Similar Word,Similarity';
    const rows = corpusResult.results.map(
      (item) => `${item.rank},"${corpusResult.target}","${item.word}",${item.similarity.toFixed(3)}`
    );
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `semantiq_similar_${corpusResult.target}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const activeVectorGeometry = useMemo(() => {
    if (!corpusResult || !corpusResult.found || !corpusResult.vectorGeometry) return undefined;
    const selectedItem =
      corpusResult.results.find((r) => r.word === selectedGraphWord) ?? corpusResult.results[0];
    if (!selectedItem) return corpusResult.vectorGeometry;
    return buildGeometryForSelectedMatch(
      corpusResult.vectorGeometry,
      corpusResult.target,
      selectedItem.word,
      selectedItem.similarity
    );
  }, [corpusResult, selectedGraphWord]);

  const wrapperClass = embedded
    ? ''
    : 'max-w-[1200px] mx-auto px-4 sm:px-8 py-14 sm:py-20';

  return (
    <section className={wrapperClass}>
      {!embedded && (
        <div className="mb-8">
          <p className="font-mono text-[11px] tracking-[0.16em] text-[#5BA4D6] mb-2">
            UPLOAD DATASET
          </p>
          <h2 className="font-display font-bold text-2xl sm:text-4xl tracking-tight text-[#CFDDD8]">
            Upload text dataset &amp; predict similar words
          </h2>
          <p className="text-sm sm:text-base text-[#91A4B0] mt-2 max-w-2xl">
            Upload a text dataset or use the built-in training corpus to compute word-context vectors and rank similar words with cosine similarity.
          </p>
        </div>
      )}

      <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-5 sm:p-8 space-y-6">
        {/* Corpus Upload Dropzone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsCorpusDragging(true);
          }}
          onDragLeave={() => setIsCorpusDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsCorpusDragging(false);
            void handleCorpusFileSelected(e.dataTransfer.files);
          }}
          className={`rounded-lg border border-dashed p-6 text-center transition-colors ${
            isCorpusDragging
              ? 'border-[#5BA4D6] bg-[#143F62]/25'
              : 'border-[rgba(207,221,216,0.2)] bg-[#071522]/60'
          }`}
        >
          <input
            ref={corpusFileRef}
            type="file"
            accept=".txt,.csv,.json,.md"
            onChange={(e) => void handleCorpusFileSelected(e.target.files)}
            className="hidden"
          />
          <Upload className="w-6 h-6 text-[#5BA4D6] mx-auto mb-2.5" />
          <p className="font-display text-base font-medium text-[#CFDDD8]">
            Upload text corpus to train Word-Context Vector model (window = 2)
          </p>
          <p className="text-xs text-[#91A4B0] mt-1">
            Tokenises with <span className="font-mono text-[#CFDDD8]">[a-z]+</span>, builds co-occurrence vectors, and ranks Top-K words by cosine similarity.
          </p>

          <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              onClick={() => corpusFileRef.current?.click()}
              className="px-4 py-2 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/35 text-xs font-medium transition-colors cursor-pointer"
            >
              Upload Corpus File
            </button>
            <button
              type="button"
              onClick={() => {
                setCorpusSentences(DEFAULT_TRAINING_CORPUS);
                setCorpusFileName('default_training_corpus.txt (18 sentences)');
                setCorpusError(null);
                setCorpusResult(null);
                setSelectedGraphWord(null);
              }}
              className="px-4 py-2 rounded-md bg-[#0B1925] hover:bg-[#010A12] text-[#91A4B0] hover:text-[#CFDDD8] border border-[rgba(207,221,216,0.14)] text-xs font-medium transition-colors cursor-pointer"
            >
              Use Default 18-Sentence Corpus
            </button>
          </div>
        </div>

        {corpusSentences.length > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.10)] text-xs">
            <div className="flex items-center gap-2.5 text-[#CFDDD8]">
              <FileText className="w-4 h-4 text-[#5BA4D6]" />
              <span className="font-medium">{corpusFileName}</span>
            </div>
            <span className="font-mono text-[#91A4B0]">
              Window size: 2 · Cosine similarity
            </span>
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            void handleRunCorpusPredict();
          }}
          className="flex flex-col md:flex-row gap-4 items-end"
        >
          <div className="flex-1 w-full">
            <label
              htmlFor="corpus-word-input"
              className="block font-mono text-[11px] tracking-wider text-[#91A4B0] mb-2"
            >
              INPUT WORD
            </label>
            <input
              id="corpus-word-input"
              type="text"
              value={corpusWord}
              onChange={(e) => {
                setCorpusWord(e.target.value);
                if (corpusError) setCorpusError(null);
              }}
              placeholder="Enter a word (e.g. computer)"
              className="w-full h-11 px-4 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] focus:border-[#5BA4D6] text-sm font-display text-[#CFDDD8] placeholder:text-[#647987]"
            />
          </div>

          <div className="w-full md:w-36">
            <label className="block font-mono text-[11px] tracking-wider text-[#91A4B0] mb-2">
              TOP K
            </label>
            <div className="grid grid-cols-2 p-1 rounded-md bg-[#010A12] border border-[rgba(207,221,216,0.16)] h-11 items-center">
              <button
                type="button"
                onClick={() => {
                  setCorpusTopK(5);
                  if (corpusResult) void handleRunCorpusPredict(corpusWord, 5);
                }}
                className={`h-full rounded text-xs font-medium transition-colors cursor-pointer ${
                  corpusTopK === 5 ? 'bg-[#143F62] text-[#CFDDD8]' : 'text-[#91A4B0] hover:text-[#CFDDD8]'
                }`}
              >
                Top 5
              </button>
              <button
                type="button"
                onClick={() => {
                  setCorpusTopK(10);
                  if (corpusResult) void handleRunCorpusPredict(corpusWord, 10);
                }}
                className={`h-full rounded text-xs font-medium transition-colors cursor-pointer ${
                  corpusTopK === 10 ? 'bg-[#143F62] text-[#CFDDD8]' : 'text-[#91A4B0] hover:text-[#CFDDD8]'
                }`}
              >
                Top 10
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={corpusLoading}
            className="w-full md:w-auto h-11 px-6 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/40 text-sm font-medium transition-colors cursor-pointer disabled:opacity-50 whitespace-nowrap inline-flex items-center justify-center gap-2"
          >
            {corpusLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Training...
              </>
            ) : (
              'Predict Similar Words'
            )}
          </button>
        </form>

        {corpusError && (
          <div
            role="alert"
            className="flex items-start gap-2.5 p-3.5 rounded-md bg-[#E17E7E]/10 border border-[#E17E7E]/30 text-sm text-[#E17E7E]"
          >
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{corpusError}</span>
          </div>
        )}
      </div>

      {/* SIMILAR WORD PREDICTION + VECTOR GEOMETRY GRAPH */}
      {corpusResult && !corpusLoading && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="mt-6 rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-6 sm:p-8 space-y-6"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[rgba(207,221,216,0.10)]">
            <div>
              <p className="font-mono text-[11px] tracking-[0.16em] text-[#5BA4D6]">
                SIMILAR WORD PREDICTION
              </p>
              <h3 className="font-display text-lg sm:text-xl font-bold text-[#CFDDD8] mt-1">
                Input word: <span className="text-[#65C99A]">{corpusResult.target}</span>
              </h3>
              {corpusResult.found && (
                <p className="text-xs text-[#91A4B0] mt-0.5">
                  Top {corpusResult.results.length} similar words:
                </p>
              )}
            </div>

            {corpusResult.found && (
              <button
                type="button"
                onClick={handleExportCorpusCsv}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-md bg-[#071522] hover:bg-[#143F62] border border-[rgba(207,221,216,0.14)] text-xs font-medium text-[#CFDDD8] transition-colors cursor-pointer self-start sm:self-auto"
              >
                <Download className="w-3.5 h-3.5" />
                Export CSV
              </button>
            )}
          </div>

          {corpusResult.found ? (
            <>
              <div className="space-y-2.5">
                {corpusResult.results.map((item, idx) => (
                  <button
                    key={item.word}
                    type="button"
                    onClick={() => {
                      setCorpusWord(item.word);
                      void handleRunCorpusPredict(item.word, corpusTopK);
                    }}
                    className="w-full text-left px-4 py-3 rounded-md bg-[#071522] hover:bg-[#143F62]/35 border border-[rgba(207,221,216,0.08)] hover:border-[#5BA4D6]/40 transition-colors cursor-pointer flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="font-mono text-sm font-semibold text-[#5BA4D6] w-6 shrink-0 tabular-nums">
                        {item.rank}.
                      </span>
                      <span className="font-display text-base font-semibold text-[#CFDDD8]">
                        {item.word}
                      </span>
                    </div>

                    <div className="flex items-center gap-3 shrink-0 w-full sm:w-80">
                      <div className="flex-1 h-2.5 rounded bg-[#010A12] overflow-hidden">
                        <div
                          className={`h-full rounded ${
                            idx === 0 ? 'bg-[#E9B96E]' : 'bg-[#65C99A]'
                          }`}
                          style={{ width: `${Math.max(4, item.similarity * 100)}%` }}
                        />
                      </div>
                      <span
                        className={`font-mono text-xs sm:text-sm font-semibold tabular-nums whitespace-nowrap ${
                          idx === 0 ? 'text-[#E9B96E]' : 'text-[#65C99A]'
                        }`}
                      >
                        Similarity: {item.similarity.toFixed(3)}
                      </span>
                    </div>
                  </button>
                ))}
              </div>

              {/* Graphical Vector Representation wrt Target Word */}
              {activeVectorGeometry && (
                <div className="space-y-3 pt-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono text-[11px] tracking-wider text-[#91A4B0]">
                      COMPARE VECTOR ANGLE WITH TARGET &ldquo;{corpusResult.target}&rdquo;:
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {corpusResult.results.map((item) => {
                        const isPlotted =
                          (selectedGraphWord ?? corpusResult.results[0]?.word) === item.word;
                        return (
                          <button
                            key={`plot-${item.word}`}
                            type="button"
                            onClick={() => setSelectedGraphWord(item.word)}
                            className={`px-2.5 py-1 rounded text-xs font-mono border transition-colors cursor-pointer ${
                              isPlotted
                                ? 'bg-[#143F62] border-[#5BA4D6]/60 text-[#CFDDD8]'
                                : 'bg-[#071522] border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8]'
                            }`}
                          >
                            {item.word}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <VectorAngleDiagram geometry={activeVectorGeometry} />
                </div>
              )}
            </>
          ) : (
            <div className="space-y-4">
              <p className="font-mono text-sm text-[#E9B96E]">
                {corpusResult.message || 'Word not found in the vocabulary.'}
              </p>
              <div>
                <p className="text-xs text-[#91A4B0] mb-2">
                  Available vocabulary ({corpusResult.vocabularySize} words):
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {corpusResult.vocabularySample.map((vocabWord) => (
                    <button
                      key={vocabWord}
                      type="button"
                      onClick={() => {
                        setCorpusWord(vocabWord);
                        void handleRunCorpusPredict(vocabWord, corpusTopK);
                      }}
                      className="px-2.5 py-1 rounded bg-[#071522] hover:bg-[#143F62] border border-[rgba(207,221,216,0.12)] text-xs font-mono text-[#CFDDD8] cursor-pointer"
                    >
                      {vocabWord}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </section>
  );
}
