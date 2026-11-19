import {
  computeFastTextSimilarity,
  computeTransformerSimilarity,
  computeWord2VecSimilarity,
  encodeBatchStatic,
  encodeBatchTransformer,
  encodeBatchWord2Vec,
} from './embeddings.ts';
import { computeTfidfCosine } from './tfidf.ts';
import type {
  BenchmarkDatasetId,
  BenchmarkPairInput,
  CoreAlgorithmId,
  EvaluationAlgorithmMetrics,
  EvaluationPairRow,
  EvaluationResponse,
} from './types.ts';
import { analyzeWordNet } from './wordnet.ts';
import { cos_sim } from '@huggingface/transformers';

// Authentic WordSim-353 subset (Finkelstein et al., 2002; human ratings normalized 0-10 -> 0-100)
export const WORDSIM_353_SUBSET: BenchmarkPairInput[] = [
  { word_a: 'love', word_b: 'sex', human_score: 67.7 },
  { word_a: 'tiger', word_b: 'cat', human_score: 73.5 },
  { word_a: 'tiger', word_b: 'tiger', human_score: 100.0 },
  { word_a: 'book', word_b: 'paper', human_score: 74.6 },
  { word_a: 'computer', word_b: 'keyboard', human_score: 76.2 },
  { word_a: 'computer', word_b: 'internet', human_score: 75.8 },
  { word_a: 'planet', word_b: 'star', human_score: 84.5 },
  { word_a: 'doctor', word_b: 'nurse', human_score: 70.0 },
  { word_a: 'king', word_b: 'queen', human_score: 85.8 },
  { word_a: 'car', word_b: 'automobile', human_score: 89.4 },
  { word_a: 'money', word_b: 'bank', human_score: 85.0 },
  { word_a: 'dollar', word_b: 'buck', human_score: 92.2 },
  { word_a: 'journey', word_b: 'voyage', human_score: 92.9 },
  { word_a: 'coast', word_b: 'shore', human_score: 91.0 },
  { word_a: 'ocean', word_b: 'sea', human_score: 87.9 },
  { word_a: 'cup', word_b: 'coffee', human_score: 65.8 },
  { word_a: 'professor', word_b: 'cucumber', human_score: 3.1 },
  { word_a: 'chord', word_b: 'smile', human_score: 5.4 },
  { word_a: 'noon', word_b: 'string', human_score: 8.1 },
  { word_a: 'rooster', word_b: 'voyage', human_score: 6.2 },
];

// Authentic SimLex-999 subset (Hill et al., 2015; strict similarity ratings 0-10 -> 0-100)
export const SIMLEX_999_SUBSET: BenchmarkPairInput[] = [
  { word_a: 'old', word_b: 'new', human_score: 15.8 },
  { word_a: 'smart', word_b: 'intelligent', human_score: 92.0 },
  { word_a: 'hard', word_b: 'difficult', human_score: 87.7 },
  { word_a: 'happy', word_b: 'cheerful', human_score: 95.5 },
  { word_a: 'hard', word_b: 'easy', human_score: 9.5 },
  { word_a: 'fast', word_b: 'rapid', human_score: 87.5 },
  { word_a: 'stupid', word_b: 'dumb', human_score: 95.8 },
  { word_a: 'weird', word_b: 'strange', human_score: 89.3 },
  { word_a: 'wide', word_b: 'narrow', human_score: 10.3 },
  { word_a: 'bad', word_b: 'awful', human_score: 84.2 },
  { word_a: 'car', word_b: 'automobile', human_score: 98.0 },
  { word_a: 'cup', word_b: 'coffee', human_score: 7.5 },
  { word_a: 'doctor', word_b: 'nurse', human_score: 52.8 },
  { word_a: 'student', word_b: 'pupil', human_score: 93.5 },
  { word_a: 'coast', word_b: 'shore', human_score: 90.0 },
  { word_a: 'cat', word_b: 'dog', human_score: 62.5 },
  { word_a: 'book', word_b: 'novel', human_score: 79.3 },
  { word_a: 'movie', word_b: 'film', human_score: 96.5 },
  { word_a: 'sun', word_b: 'moon', human_score: 36.2 },
  { word_a: 'bread', word_b: 'butter', human_score: 13.8 },
];

function averageRanks(values: number[]): number[] {
  const indexed = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array<number>(values.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j + 1 < indexed.length && indexed[j + 1].v === indexed[i].v) {
      j++;
    }
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) {
      ranks[indexed[k].i] = avgRank;
    }
    i = j + 1;
  }
  return ranks;
}

function pearsonCorrelation(xs: number[], ys: number[]): number | null {
  if (xs.length < 3 || xs.length !== ys.length) return null;
  const n = xs.length;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let denX = 0;
  let denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  const den = Math.sqrt(denX * denY);
  if (den === 0) return null;
  return Math.round((num / den) * 1000) / 1000;
}

function spearmanCorrelation(xs: number[], ys: number[]): number | null {
  if (xs.length < 3 || xs.length !== ys.length) return null;
  return pearsonCorrelation(averageRanks(xs), averageRanks(ys));
}

export function getBenchmarkDatasetsMeta() {
  return [
    {
      id: 'wordsim353' as const,
      name: 'WordSim-353 (Benchmark Subset)',
      description:
        'Measures general semantic relatedness & association (Finkelstein et al., 2002). Includes related concepts like computer–keyboard and money–bank.',
      pairCount: WORDSIM_353_SUBSET.length,
    },
    {
      id: 'simlex999' as const,
      name: 'SimLex-999 (Benchmark Subset)',
      description:
        'Measures strict semantic similarity rather than topical association (Hill et al., 2015). Penalizes antonyms (old–new) and purely associated pairs (cup–coffee).',
      pairCount: SIMLEX_999_SUBSET.length,
    },
  ];
}

export async function runBenchmarkEvaluation(params: {
  datasetId: BenchmarkDatasetId;
  customPairs?: BenchmarkPairInput[];
}): Promise<EvaluationResponse> {
  const { datasetId, customPairs } = params;
  const rawPairs =
    datasetId === 'custom' && customPairs && customPairs.length >= 3
      ? customPairs.slice(0, 60)
      : datasetId === 'simlex999'
        ? SIMLEX_999_SUBSET
        : WORDSIM_353_SUBSET;

  const datasetName =
    datasetId === 'custom'
      ? `Custom Uploaded CSV (${rawPairs.length} pairs)`
      : datasetId === 'simlex999'
        ? 'SimLex-999 Benchmark Subset'
        : 'WordSim-353 Benchmark Subset';

  // Normalize human scores if given in [0, 1] or [0, 10]
  const maxHuman = Math.max(...rawPairs.map((p) => p.human_score), 1);
  const scaleFactor = maxHuman <= 1.01 ? 100 : maxHuman <= 10.01 ? 10 : 1;

  const pairs = rawPairs.map((p) => ({
    word_a: p.word_a.trim(),
    word_b: p.word_b.trim(),
    humanScore: Math.round(p.human_score * scaleFactor * 10) / 10,
  }));

  const uniqueWords = Array.from(
    new Set(pairs.flatMap((p) => [p.word_a, p.word_b]))
  );
  const wordIdx = new Map(uniqueWords.map((w, i) => [w, i]));

  // Time each algorithm batch honestly
  const wnStart = performance.now();
  const wnScores = pairs.map((p) => analyzeWordNet(p.word_a, p.word_b).score);
  const wnTime = Math.round((performance.now() - wnStart) * 10) / 10;

  const tfStart = performance.now();
  const tfScores = pairs.map(
    (p) => computeTfidfCosine(p.word_a, p.word_b).scorePercent
  );
  const tfTime = Math.round((performance.now() - tfStart) * 10) / 10;

  let w2vScores: Array<number | null> = pairs.map(() => null);
  let w2vTime = 0;
  try {
    const t0 = performance.now();
    const vecs = await encodeBatchWord2Vec(uniqueWords);
    w2vScores = pairs.map((p) => {
      const va = vecs[wordIdx.get(p.word_a)!];
      const vb = vecs[wordIdx.get(p.word_b)!];
      const c = Math.max(0, Math.min(1, cos_sim(va, vb)));
      return Math.round(c * 1000) / 10;
    });
    w2vTime = Math.round((performance.now() - t0) * 10) / 10;
  } catch {
    // Fallback per-pair if batch fails
    try {
      const t0 = performance.now();
      w2vScores = await Promise.all(
        pairs.map(async (p) => (await computeWord2VecSimilarity(p.word_a, p.word_b)).scorePercent)
      );
      w2vTime = Math.round((performance.now() - t0) * 10) / 10;
    } catch {
      w2vScores = pairs.map(() => null);
    }
  }

  let ftScores: Array<number | null> = pairs.map(() => null);
  let ftTime = 0;
  try {
    const t0 = performance.now();
    const vecs = await encodeBatchStatic(uniqueWords);
    ftScores = pairs.map((p) => {
      const va = vecs[wordIdx.get(p.word_a)!];
      const vb = vecs[wordIdx.get(p.word_b)!];
      const c = Math.max(0, Math.min(1, cos_sim(va, vb)));
      return Math.round(c * 1000) / 10;
    });
    ftTime = Math.round((performance.now() - t0) * 10) / 10;
  } catch {
    try {
      const t0 = performance.now();
      ftScores = await Promise.all(
        pairs.map(async (p) => (await computeFastTextSimilarity(p.word_a, p.word_b)).scorePercent)
      );
      ftTime = Math.round((performance.now() - t0) * 10) / 10;
    } catch {
      ftScores = pairs.map(() => null);
    }
  }

  let trScores: Array<number | null> = pairs.map(() => null);
  let trTime = 0;
  try {
    const t0 = performance.now();
    const vecs = await encodeBatchTransformer(uniqueWords);
    trScores = pairs.map((p) => {
      const va = vecs[wordIdx.get(p.word_a)!];
      const vb = vecs[wordIdx.get(p.word_b)!];
      const c = Math.max(0, Math.min(1, cos_sim(va, vb)));
      return Math.round(c * 1000) / 10;
    });
    trTime = Math.round((performance.now() - t0) * 10) / 10;
  } catch {
    try {
      const t0 = performance.now();
      trScores = await Promise.all(
        pairs.map(async (p) => (await computeTransformerSimilarity(p.word_a, p.word_b)).scorePercent)
      );
      trTime = Math.round((performance.now() - t0) * 10) / 10;
    } catch {
      trScores = pairs.map(() => null);
    }
  }

  const rows: EvaluationPairRow[] = pairs.map((p, idx) => ({
    word_a: p.word_a,
    word_b: p.word_b,
    humanScore: p.humanScore,
    predictions: {
      wordnet: wnScores[idx],
      tfidf: tfScores[idx],
      word2vec: w2vScores[idx],
      fasttext: ftScores[idx],
      transformer: trScores[idx],
    },
  }));

  const buildMetric = (
    algorithm: CoreAlgorithmId,
    name: string,
    category: string,
    scores: Array<number | null>,
    timeMs: number
  ): EvaluationAlgorithmMetrics => {
    const validHumans: number[] = [];
    const validPreds: number[] = [];
    for (let i = 0; i < pairs.length; i++) {
      const pred = scores[i];
      if (pred !== null && !Number.isNaN(pred)) {
        validHumans.push(pairs[i].humanScore);
        validPreds.push(pred);
      }
    }
    const mae =
      validPreds.length > 0
        ? Math.round(
            (validPreds.reduce((acc, v, i) => acc + Math.abs(v - validHumans[i]), 0) /
              validPreds.length) *
              10
          ) / 10
        : null;

    return {
      algorithm,
      name,
      category,
      spearmanRho: spearmanCorrelation(validHumans, validPreds),
      pearsonR: pearsonCorrelation(validHumans, validPreds),
      mae,
      evaluatedCount: validPreds.length,
      coveragePercent: Math.round((validPreds.length / pairs.length) * 1000) / 10,
      processingTimeMs: timeMs,
    };
  };

  const algorithmMetrics: EvaluationAlgorithmMetrics[] = [
    buildMetric('transformer', 'Sentence-BERT', 'Contextual embeddings', trScores, trTime),
    buildMetric('wordnet', 'WordNet (Wu-Palmer)', 'Lexical', wnScores, wnTime),
    buildMetric('fasttext', 'FastText (Subword)', 'Subword embeddings', ftScores, ftTime),
    buildMetric('word2vec', 'Word2Vec (Static)', 'Word embeddings', w2vScores, w2vTime),
    buildMetric('tfidf', 'TF-IDF + Cosine', 'Statistical', tfScores, tfTime),
  ];

  const bestBySpearman = [...algorithmMetrics]
    .filter((m) => m.spearmanRho !== null)
    .sort((a, b) => (b.spearmanRho ?? -1) - (a.spearmanRho ?? -1))[0];

  const summary = bestBySpearman
    ? `Evaluated ${pairs.length} pairs on ${datasetName}. Highest rank correlation with human judgments: ${bestBySpearman.name} (Spearman ρ = ${bestBySpearman.spearmanRho?.toFixed(3)}).`
    : `Evaluated ${pairs.length} pairs on ${datasetName}.`;

  return {
    datasetId,
    datasetName,
    pairCount: pairs.length,
    algorithmMetrics,
    pairs: rows,
    summary,
    timestamp: new Date().toISOString(),
  };
}
