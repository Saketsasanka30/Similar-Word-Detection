import fs from 'fs';
import path from 'path';
import {
  computeVectorGeometry2D,
  encodeBatchStatic,
  encodeBatchTransformer,
  FASTTEXT_MODEL_ID,
  TRANSFORMER_MODEL_ID,
} from './embeddings.ts';
import type {
  NlpPipelineStage,
  SimilarWordMatch,
  SimilarWordsResponse,
} from './types.ts';
import { trainAndPredictCorpusModel } from './contextCorpusModel.ts';
import {
  getSynset,
  getWordDefinitionAndSynonyms,
  lookupSynsets,
} from './wordnet.ts';

const dictDir = path.resolve(process.cwd(), 'node_modules/wordnet-db/dict');
const EMBEDDING_DIMS = 256;

interface VocabIndex {
  words: string[];
  wordSet: Set<string>;
  vectors: Float32Array;
}

let vocabIndexPromise: Promise<VocabIndex> | null = null;

function collectCommonEnglishWords(): string[] {
  const words = new Set<string>();
  const files = ['index.noun', 'index.verb', 'index.adj', 'index.adv'];

  for (const file of files) {
    const content = fs.readFileSync(path.join(dictDir, file), 'utf8');
    const lines = content.split('\n');
    for (const line of lines) {
      if (!line || line.startsWith('  ')) continue;
      const parts = line.trim().split(/\s+/);
      if (parts.length < 6) continue;
      const lemma = parts[0];
      if (!/^[a-z]{3,15}$/.test(lemma)) continue;
      const pCnt = parseInt(parts[3], 10);
      const tagSenseCnt = parseInt(parts[4 + pCnt + 1], 10) || 0;
      if (tagSenseCnt >= 2) {
        words.add(lemma);
      }
    }
  }
  return Array.from(words);
}

function normalizeL2(vec: number[]): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
  const norm = Math.sqrt(sumSq) || 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

async function buildVocabIndex(): Promise<VocabIndex> {
  const words = collectCommonEnglishWords();
  const vectors = new Float32Array(words.length * EMBEDDING_DIMS);

  const BATCH_SIZE = 512;
  for (let start = 0; start < words.length; start += BATCH_SIZE) {
    const chunk = words.slice(start, start + BATCH_SIZE);
    const chunkVecs = await encodeBatchStatic(chunk);
    for (let i = 0; i < chunkVecs.length; i++) {
      const normalized = normalizeL2(chunkVecs[i]);
      vectors.set(normalized, (start + i) * EMBEDDING_DIMS);
    }
  }

  return {
    words,
    wordSet: new Set(words),
    vectors,
  };
}

export function getVocabIndex(): Promise<VocabIndex> {
  if (!vocabIndexPromise) {
    vocabIndexPromise = buildVocabIndex().catch((err) => {
      vocabIndexPromise = null;
      throw err;
    });
  }
  return vocabIndexPromise;
}

export function warmUpSimilarWordsIndex(): void {
  void getVocabIndex().catch(() => {});
}

export function preprocessSingleWord(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/[^a-z\s'-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function pickContrastWord(query: string): string {
  const q = query.toLowerCase();
  if (['apple', 'banana', 'fruit', 'orange', 'mango', 'food'].includes(q)) return 'telescope';
  if (['ocean', 'sea', 'river', 'water', 'lake'].includes(q)) return 'keyboard';
  return 'apple';
}

export async function findSimilarWords(
  rawInput: string,
  topK: 5 | 10 = 5
): Promise<SimilarWordsResponse> {
  // Step 1 & 2: User Input -> Text Preprocessing
  const preprocessed = preprocessSingleWord(rawInput);
  if (!preprocessed) {
    throw new Error('Please enter a valid alphabetic word.');
  }

  const inputDetails = getWordDefinitionAndSynonyms(preprocessed);

  // Check exact Python Word-Context Vector (window=2) corpus model first when word is in the training vocabulary
  const corpusPrediction = trainAndPredictCorpusModel({
    targetWord: preprocessed,
    topK,
    windowSize: 2,
  });

  if (corpusPrediction.found && corpusPrediction.results.length > 0) {
    const matches: SimilarWordMatch[] = corpusPrediction.results.map((item) => {
      const details = getWordDefinitionAndSynonyms(item.word);
      const isExactSyn = inputDetails.exactSynonyms.has(item.word.toLowerCase());
      return {
        word: item.word.charAt(0).toUpperCase() + item.word.slice(1),
        cosineSimilarity: item.similarity,
        scorePercent: item.scorePercent,
        matchType: isExactSyn ? 'Exact Synonym' : 'Closely Related',
        posLabel: details.primaryPos,
        definition:
          details.primaryDefinition ??
          'Distributional neighbor identified via Word-Context Co-occurrence vectors (window = 2).',
        exampleUsage: details.primaryExample,
      };
    });

    const topMatchLabel = matches[0]?.word.toLowerCase() || 'code';
    const contrastLabel = pickContrastWord(preprocessed);
    const vectorGeometry = await computeVectorGeometry2D(
      preprocessed,
      topMatchLabel,
      contrastLabel
    );

    const exactSynonymCount = matches.filter((m) => m.matchType === 'Exact Synonym').length;
    const relatedCount = matches.length - exactSynonymCount;

    const pipelineStages: NlpPipelineStage[] = [
      {
        step: 1,
        name: 'Corpus & Input',
        summary: `Received input word "${rawInput.trim()}"`,
        details: [
          `Corpus sentences: ${corpusPrediction.corpusSentenceCount}`,
          `Target Top-K: ${topK}`,
        ],
      },
      {
        step: 2,
        name: 'Tokenisation',
        summary: `Normalized to "${preprocessed}" with [a-z]+ regex`,
        details: ['Lowercased text', 'Extracted alphabetic tokens'],
      },
      {
        step: 3,
        name: 'Vocabulary Build',
        summary: `Sorted vocabulary of ${corpusPrediction.vocabularySize} unique words`,
        details: [`Window size: ${corpusPrediction.windowSize} context tokens left/right`],
      },
      {
        step: 4,
        name: 'Word-Context Vectors',
        summary: `${corpusPrediction.vocabularySize}-dim co-occurrence count vectors`,
        details: ['Counted neighboring words within ±2 positions across corpus'],
      },
      {
        step: 5,
        name: 'Cosine Similarity',
        summary: `Compared vector "${preprocessed}" against ${corpusPrediction.vocabularySize - 1} words`,
        details: [
          'Formula: cos(A, B) = (A · B) / (||A|| × ||B||)',
          `Top angle θ: ${vectorGeometry.thetaDegrees}° (${topMatchLabel})`,
        ],
      },
      {
        step: 6,
        name: 'Ranked Prediction',
        summary: `Top match: ${matches[0]?.word.toLowerCase()} (${matches[0]?.cosineSimilarity.toFixed(3)})`,
        details: matches.map((m, i) => `${i + 1}. ${m.word.toLowerCase()} — ${m.cosineSimilarity.toFixed(3)}`),
      },
    ];

    return {
      rawInput,
      preprocessedWord: preprocessed,
      inputDefinition: inputDetails.primaryDefinition,
      topK,
      algorithm: 'Word-Context Co-occurrence Vectors (window=2) + Cosine Similarity',
      modelName: 'Corpus Co-occurrence Model (29-word vocab)',
      dimensions: corpusPrediction.vocabularySize,
      vocabularySize: corpusPrediction.vocabularySize,
      vectorPreview: [],
      exactSynonymCount,
      relatedCount,
      results: matches,
      vectorGeometry,
      pipelineStages,
      timestamp: new Date().toISOString(),
    };
  }

  const index = await getVocabIndex();

  // Gather WordNet synset & 1-hop related lemmas
  const wnCandidates = new Set<string>();
  const synsets = lookupSynsets(preprocessed);
  for (const syn of synsets.slice(0, 4)) {
    for (const lemma of syn.lemmas) {
      const clean = preprocessSingleWord(lemma);
      if (clean && clean !== preprocessed && /^[a-z]{3,18}$/.test(clean)) {
        wnCandidates.add(clean);
      }
    }
    for (const ptr of syn.pointers
      .filter((p) => ['&', '^', '@', '~', '+'].includes(p.symbol))
      .slice(0, 12)) {
      const target = getSynset(ptr.pos, ptr.offset);
      if (target) {
        for (const lemma of target.lemmas.slice(0, 3)) {
          const clean = preprocessSingleWord(lemma);
          if (clean && clean !== preprocessed && /^[a-z]{3,18}$/.test(clean)) {
            wnCandidates.add(clean);
          }
        }
      }
    }
  }

  // Step 3 (Stage 1 Retrieval): Vectorize query with FastText static embeddings
  const [queryStaticRaw] = await encodeBatchStatic([preprocessed]);
  const queryVec = normalizeL2(queryStaticRaw);

  const stage1Scored: Array<{ word: string; score: number }> = [];
  for (let i = 0; i < index.words.length; i++) {
    const candidate = index.words[i];
    if (candidate === preprocessed) continue;
    let dot = 0;
    const baseOffset = i * EMBEDDING_DIMS;
    for (let d = 0; d < EMBEDDING_DIMS; d++) {
      dot += queryVec[d] * index.vectors[baseOffset + d];
    }
    stage1Scored.push({ word: candidate, score: dot });
  }
  stage1Scored.sort((a, b) => b.score - a.score);

  const candidatePool = new Set<string>();
  for (const w of wnCandidates) {
    if (candidatePool.size < 25) candidatePool.add(w);
  }
  for (const item of stage1Scored.slice(0, 35)) {
    candidatePool.add(item.word);
  }

  const candidates = Array.from(candidatePool);

  // Step 4: Cosine Similarity Calculation via Sentence-BERT (all-MiniLM-L6-v2)
  const allTexts = [preprocessed, ...candidates];
  const vectors = await encodeBatchTransformer(allTexts);
  const qEmb = vectors[0];

  const matches: SimilarWordMatch[] = [];
  for (let i = 0; i < candidates.length; i++) {
    const candWord = candidates[i];
    const candEmb = vectors[i + 1];
    let dot = 0;
    for (let d = 0; d < qEmb.length; d++) {
      dot += qEmb[d] * candEmb[d];
    }
    const clamped = Math.max(0, Math.min(1, dot));
    const details = getWordDefinitionAndSynonyms(candWord);
    const isExactSyn = inputDetails.exactSynonyms.has(candWord.toLowerCase());

    matches.push({
      word: candWord.charAt(0).toUpperCase() + candWord.slice(1),
      cosineSimilarity: Math.round(clamped * 1000) / 1000,
      scorePercent: Math.round(clamped * 1000) / 10,
      matchType: isExactSyn ? 'Exact Synonym' : 'Closely Related',
      posLabel: details.primaryPos,
      definition:
        details.primaryDefinition ??
        'Semantically associated concept identified in neural vector space.',
      exampleUsage: details.primaryExample,
    });
  }

  // Step 5 & 6: Rank and return Top 5 or Top 10
  matches.sort((a, b) => b.cosineSimilarity - a.cosineSimilarity);
  const topMatches = matches.slice(0, topK);

  const topMatchLabel = topMatches[0]?.word.toLowerCase() || 'system';
  const contrastLabel = pickContrastWord(preprocessed);
  const vectorGeometry = await computeVectorGeometry2D(
    preprocessed,
    topMatchLabel,
    contrastLabel
  );

  const vectorPreview = Array.from(qEmb.slice(0, 6)).map(
    (v) => Math.round(v * 1000) / 1000
  );

  const exactSynonymCount = topMatches.filter((m) => m.matchType === 'Exact Synonym').length;
  const relatedCount = topMatches.length - exactSynonymCount;

  const pipelineStages: NlpPipelineStage[] = [
    {
      step: 1,
      name: 'Text Input',
      summary: `Received query "${rawInput.trim()}"`,
      details: [`Raw string: "${rawInput}"`, `Target Top-K: ${topK}`],
    },
    {
      step: 2,
      name: 'Text Preprocessing',
      summary: `Normalized to "${preprocessed}"`,
      details: [
        'Lowercased & trimmed whitespace',
        'Stripped punctuation & non-alphabetic symbols',
      ],
    },
    {
      step: 3,
      name: 'Tokenization',
      summary: `Extracted lexical synsets & subword tokens`,
      details: [
        `WordNet synsets matched: ${synsets.length}`,
        `Candidate pool: ${candidates.length} words`,
      ],
    },
    {
      step: 4,
      name: 'Vectorization',
      summary: `${qEmb.length}-dim dense neural vector`,
      details: [
        `Model: ${TRANSFORMER_MODEL_ID}`,
        `Vector slice: [${vectorPreview.join(', ')}, ...]`,
      ],
    },
    {
      step: 5,
      name: 'Similarity Calculation',
      summary: `Cosine similarity across ${(index.words.length + wnCandidates.size).toLocaleString()} words`,
      details: [
        'Formula: cos(A, B) = (A · B) / (||A|| × ||B||)',
        `Top angle θ: ${vectorGeometry.thetaDegrees}° (${topMatchLabel})`,
      ],
    },
    {
      step: 6,
      name: 'Result Interpretation',
      summary: `${exactSynonymCount} exact synonyms · ${relatedCount} related words`,
      details: [
        `Top match: ${topMatches[0]?.word ?? 'N/A'} (${topMatches[0]?.cosineSimilarity.toFixed(3) ?? '0.000'})`,
        'Distinguished dictionary synonyms from distributional associates',
      ],
    },
  ];

  return {
    rawInput,
    preprocessedWord: preprocessed,
    inputDefinition: inputDetails.primaryDefinition,
    topK,
    algorithm: 'Word Embeddings + Cosine Similarity',
    modelName: `${FASTTEXT_MODEL_ID} + ${TRANSFORMER_MODEL_ID}`,
    dimensions: qEmb.length,
    vocabularySize: index.words.length + wnCandidates.size,
    vectorPreview,
    exactSynonymCount,
    relatedCount,
    results: topMatches,
    vectorGeometry,
    pipelineStages,
    timestamp: new Date().toISOString(),
  };
}
