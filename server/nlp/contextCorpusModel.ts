import type { VectorGeometry2D } from './types.ts';

export const DEFAULT_TRAINING_CORPUS: string[] = [
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

export interface CorpusSimilarWordItem {
  rank: number;
  word: string;
  similarity: number;
  scorePercent: number;
}

export interface CorpusPredictionResponse {
  found: boolean;
  target: string;
  topK: number;
  vocabularySize: number;
  corpusSentenceCount: number;
  windowSize: number;
  vocabularySample: string[];
  results: CorpusSimilarWordItem[];
  vectorGeometry?: VectorGeometry2D;
  message?: string;
  timestamp: string;
}

function cosineVectors(vecA: number[], vecB: number[]): number {
  let dot = 0;
  let normSqA = 0;
  let normSqB = 0;
  for (let i = 0; i < vecA.length; i++) {
    const a = vecA[i];
    const b = vecB[i];
    dot += a * b;
    normSqA += a * a;
    normSqB += b * b;
  }
  const denom = Math.sqrt(normSqA) * Math.sqrt(normSqB);
  if (denom === 0) return 0;
  return dot / denom;
}

export function trainAndPredictCorpusModel(params: {
  targetWord: string;
  corpus?: string[];
  topK?: number;
  windowSize?: number;
}): CorpusPredictionResponse {
  const window = params.windowSize ?? 2;
  const topK = params.topK === 10 ? 10 : 5;
  const rawCorpus =
    params.corpus && params.corpus.length > 0 ? params.corpus : DEFAULT_TRAINING_CORPUS;

  // STEP 2: Tokenisation — lowercase and extract [a-z]+ words
  const tokenized: string[][] = rawCorpus
    .map((sentence) => sentence.toLowerCase().match(/[a-z]+/g) ?? [])
    .filter((tokens) => tokens.length > 0);

  // STEP 3: Create Vocabulary (sorted unique tokens)
  const vocabSet = new Set<string>();
  for (const sentence of tokenized) {
    for (const word of sentence) {
      vocabSet.add(word);
    }
  }
  const vocab = Array.from(vocabSet).sort();
  const wordToIndex = new Map<string, number>();
  vocab.forEach((word, index) => wordToIndex.set(word, index));

  // STEP 4: Create Word-Context Vectors (window = 2)
  const vectors = new Map<string, number[]>();
  for (const word of vocab) {
    vectors.set(word, new Array<number>(vocab.length).fill(0));
  }

  for (const sentence of tokenized) {
    for (let i = 0; i < sentence.length; i++) {
      const word = sentence[i];
      const vec = vectors.get(word)!;
      const start = Math.max(0, i - window);
      const end = Math.min(sentence.length, i + window + 1);
      for (let j = start; j < end; j++) {
        if (i !== j) {
          const contextWord = sentence[j];
          const ctxIdx = wordToIndex.get(contextWord);
          if (ctxIdx !== undefined) {
            vec[ctxIdx] += 1;
          }
        }
      }
    }
  }

  // STEP 5: Take Input Word
  const targetTokens = params.targetWord.toLowerCase().match(/[a-z]+/g) ?? [];
  const target = targetTokens[0] ?? params.targetWord.trim().toLowerCase();

  if (!target || !vectors.has(target)) {
    return {
      found: false,
      target,
      topK,
      vocabularySize: vocab.length,
      corpusSentenceCount: tokenized.length,
      windowSize: window,
      vocabularySample: vocab.slice(0, 30),
      results: [],
      message: 'Word not found in the vocabulary.',
      timestamp: new Date().toISOString(),
    };
  }

  // STEP 6: Calculate Cosine Similarity
  const targetVec = vectors.get(target)!;
  const candidates = vocab.filter((word) => word !== target);
  const scored = candidates.map((word) => {
    const score = cosineVectors(targetVec, vectors.get(word)!);
    return { word, score };
  });

  // STEP 7: Rank Similar Words
  scored.sort((a, b) => b.score - a.score);

  // STEP 8: Display Top K Results
  const results: CorpusSimilarWordItem[] = scored.slice(0, topK).map((item, idx) => ({
    rank: idx + 1,
    word: item.word,
    similarity: Math.round(item.score * 1000) / 1000,
    scorePercent: Math.round(item.score * 1000) / 10,
  }));

  // Build 2D cosine angle geometry for target (A), top match (B), and lowest-similarity corpus word (C)
  let vectorGeometry: VectorGeometry2D | undefined;
  if (results.length > 0) {
    const topItem = scored[0];
    const contrastItem = scored[scored.length - 1];
    const clampedTop = Math.max(-1, Math.min(1, topItem.score));
    const clampedContrast = Math.max(-1, Math.min(1, contrastItem.score));

    const thetaRad = Math.acos(clampedTop);
    const thetaDegrees = Math.round(((thetaRad * 180) / Math.PI) * 10) / 10;

    // Position Vector A at 52° in Quadrant I, Vector B below it by θ, Vector C in Quadrant IV
    const baseAngle = (52 * Math.PI) / 180;
    const angleB = baseAngle - thetaRad;
    const contrastAngleRad = Math.acos(clampedContrast);
    const angleC = Math.min(-0.28, baseAngle - contrastAngleRad - 0.35);

    vectorGeometry = {
      primaryLabel: target,
      secondaryLabel: topItem.word,
      contrastLabel: contrastItem.word,
      cosineSimilarity: Math.round(clampedTop * 1000) / 1000,
      thetaDegrees,
      contrastCosine: Math.round(clampedContrast * 1000) / 1000,
      contrastThetaDegrees: Math.round(((contrastAngleRad * 180) / Math.PI) * 10) / 10,
      modelUsed: `Word-Context Vectors (window=${window})`,
      points: [
        {
          label: target,
          x: Math.round(Math.cos(baseAngle) * 1000) / 1000,
          y: Math.round(Math.sin(baseAngle) * 1000) / 1000,
          role: 'primary',
          cosineWithPrimary: 1,
        },
        {
          label: topItem.word,
          x: Math.round(Math.cos(angleB) * 1000) / 1000,
          y: Math.round(Math.sin(angleB) * 1000) / 1000,
          role: 'secondary',
          cosineWithPrimary: Math.round(clampedTop * 1000) / 1000,
        },
        {
          label: contrastItem.word,
          x: Math.round(Math.cos(angleC) * 1000) / 1000,
          y: Math.round(Math.sin(angleC) * 1000) / 1000,
          role: 'contrast',
          cosineWithPrimary: Math.round(clampedContrast * 1000) / 1000,
        },
      ],
    };
  }

  return {
    found: true,
    target,
    topK,
    vocabularySize: vocab.length,
    corpusSentenceCount: tokenized.length,
    windowSize: window,
    vocabularySample: vocab.slice(0, 30),
    results,
    vectorGeometry,
    timestamp: new Date().toISOString(),
  };
}
