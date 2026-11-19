import { cos_sim } from '@huggingface/transformers';
import {
  computeFastTextSimilarity,
  computeTransformerSimilarity,
  computeVectorGeometry2D,
  computeWord2VecSimilarity,
  cosineDense,
  encodeBatchStatic,
  encodeBatchTransformer,
  encodeBatchWord2Vec,
  getEmbeddingModelsHealth,
  TRANSFORMER_MODEL_ID,
} from './embeddings.ts';
import { getVocabIndex } from './similarWords.ts';
import {
  computeTfidfCosine,
  computeTfidfVectors,
  cosineSparse,
  preprocessTokens,
} from './tfidf.ts';
import type {
  AlgorithmChoice,
  AlgorithmResult,
  BatchAnalysisResponse,
  BatchWordPairInput,
  BatchWordPairResult,
  DatasetAnalysisResponse,
  DatasetDocumentInput,
  DatasetMode,
  DatasetPairMatch,
  DatasetQueryMatch,
  EngineHealthStatus,
  NlpPipelineStage,
  RelationshipLabel,
  SemanticExploreResponse,
  SentenceAnalysisResponse,
  VectorGeometry2D,
  WordAnalysisResponse,
} from './types.ts';
import {
  analyzeWordNet,
  exploreWordNet,
  getWordDefinitionAndSynonyms,
} from './wordnet.ts';

export function getInterpretationBand(score: number | null): {
  label: string;
  range: string;
  tone: 'high' | 'moderate' | 'low';
} | null {
  if (score === null) return null;
  if (score >= 80) return { label: 'Very High Similarity', range: '80–100', tone: 'high' };
  if (score >= 60) return { label: 'High Similarity', range: '60–79', tone: 'high' };
  if (score >= 40) return { label: 'Moderate Similarity', range: '40–59', tone: 'moderate' };
  if (score >= 20) return { label: 'Low Similarity', range: '20–39', tone: 'low' };
  return { label: 'Very Low Similarity', range: '0–19', tone: 'low' };
}

function deriveOverallRelationship(
  wnRelation: RelationshipLabel,
  wnScore: number | null,
  transformerScore: number | null,
  wordA: string,
  wordB: string
): RelationshipLabel {
  if (wordA.trim().toLowerCase() === wordB.trim().toLowerCase()) {
    return 'Exact Match';
  }
  if (
    wnRelation === 'Synonym' ||
    wnRelation === 'Near Synonym' ||
    wnRelation === 'Hypernym' ||
    wnRelation === 'Hyponym'
  ) {
    return wnRelation;
  }
  if (wnScore !== null && wnScore >= 75) {
    return 'Related Concept';
  }
  if (transformerScore !== null && transformerScore >= 60) {
    return 'Contextually Related';
  }
  if (wnRelation === 'Related Concept' && wnScore !== null && wnScore >= 45) {
    return 'Related Concept';
  }
  return 'No Direct Relationship Detected';
}

function buildWordExplanations(
  wordA: string,
  wordB: string,
  relationship: RelationshipLabel,
  results: WordAnalysisResponse['results'],
  primaryScore: number | null,
  sharedHypernym?: string | null
): { explanation: string; technicalExplanation: string; beginnerExplanation: string } {
  if (relationship === 'Exact Match') {
    const msg = `"${wordA}" and "${wordB}" normalize to the same lexical form, yielding 100% alignment across lexical, statistical, and embedding models.`;
    return {
      explanation: msg,
      technicalExplanation: msg,
      beginnerExplanation: `Both inputs represent the exact same word ("${wordA.toLowerCase()}"), so every method measures maximum similarity.`,
    };
  }

  const techParts: string[] = [];
  if (results.transformer?.status === 'completed' && results.transformer.score !== null) {
    techParts.push(
      `Sentence-BERT (all-MiniLM-L6-v2) encodes both inputs into 384-dim L2-normalized embeddings with cosine similarity ${results.transformer.rawCosineOrMetric?.toFixed(3)} (${results.transformer.score}%).`
    );
  }
  if (results.wordnet?.status === 'completed' && results.wordnet.score !== null) {
    techParts.push(
      `WordNet reports ${results.wordnet.score}% Wu-Palmer taxonomy similarity (${relationship}${
        sharedHypernym ? `, LCS "${sharedHypernym}"` : ''
      }).`
    );
  } else if (results.wordnet?.status === 'not_available') {
    techParts.push(
      `WordNet could not compute a comparable noun/verb taxonomy path for this pair.`
    );
  }
  if (results.tfidf?.status === 'completed' && results.tfidf.score !== null) {
    techParts.push(
      `TF-IDF records ${results.tfidf.score}% because it evaluates exact normalized surface token overlap rather than synonymy or contextual meaning.`
    );
  }
  if (results.word2vec?.status === 'completed' && results.word2vec.score !== null) {
    techParts.push(
      `Word2Vec static embeddings score ${results.word2vec.score}%, while FastText subword embeddings score ${results.fasttext?.score ?? 'N/A'}%.`
    );
  }

  const scoreLabel =
    primaryScore !== null ? `${primaryScore.toFixed(1)}% (${getInterpretationBand(primaryScore)?.label})` : 'evaluated';
  let beginner = '';
  if (relationship === 'Synonym' || relationship === 'Near Synonym') {
    beginner = `"${wordA}" and "${wordB}" have nearly the same meaning (${scoreLabel}). Even though they are spelled differently, AI language models place them close together in meaning space.`;
  } else if (relationship === 'Hypernym' || relationship === 'Hyponym') {
    beginner = `"${wordA}" and "${wordB}" have a category relationship (${scoreLabel})—one is a broader or more specific kind of the other${
      sharedHypernym ? ` under "${sharedHypernym}"` : ''
    }.`;
  } else if (relationship === 'Contextually Related' || relationship === 'Related Concept') {
    beginner = `"${wordA}" and "${wordB}" often appear in related topics or situations (${scoreLabel}), even if they are not direct dictionary synonyms.`;
  } else {
    beginner = `"${wordA}" and "${wordB}" do not share a strong direct meaning connection (${scoreLabel}), because they belong to largely separate topics.`;
  }

  return {
    explanation: techParts.join(' ') || beginner,
    technicalExplanation: techParts.join(' '),
    beginnerExplanation: beginner,
  };
}

export async function analyzeWordPair(
  wordA: string,
  wordB: string,
  algorithm: AlgorithmChoice = 'all'
): Promise<WordAnalysisResponse> {
  const cleanA = wordA.trim();
  const cleanB = wordB.trim();
  const runAll = algorithm === 'all';

  const results: WordAnalysisResponse['results'] = {};

  // Always gather WordNet lexical metadata for sense & relationship transparency
  const wnStart = performance.now();
  const wn = analyzeWordNet(cleanA, cleanB);
  const wnTime = Math.round((performance.now() - wnStart) * 10) / 10;

  if (runAll || algorithm === 'wordnet') {
    results.wordnet = {
      algorithm: 'wordnet',
      name: 'WordNet (Wu-Palmer)',
      methodCategory: 'Lexical',
      score: wn.score,
      rawCosineOrMetric: wn.rawMetric,
      status: wn.status,
      processingTimeMs: wnTime,
      calculationMethod: 'Wu-Palmer Least Common Subsumer (LCS) taxonomy depth ratio',
      modelName: 'Princeton WordNet 3.1 (Wu-Palmer & Path)',
      explanation:
        wn.status === 'completed' && wn.bestPair
          ? `Computed via Wu-Palmer taxonomy similarity (WUP = ${wn.bestPair.wupSimilarity}, Path = ${wn.bestPair.pathSimilarity})${
              wn.bestPair.sharedHypernym
                ? ` through shared ancestor "${wn.bestPair.sharedHypernym}"`
                : ''
            }.`
          : wn.reasonIfUnavailable || 'WordNet could not establish a valid synset path.',
      limitation: 'Limited to dictionary-indexed words and affected by word-sense ambiguity or differing parts of speech.',
      ...(wn.reasonIfUnavailable ? { reasonIfUnavailable: wn.reasonIfUnavailable } : {}),
      technicalDetails: {
        synsetsFoundA: wn.synsetsA.length,
        synsetsFoundB: wn.synsetsB.length,
        wupSimilarity: wn.bestPair?.wupSimilarity ?? null,
        pathSimilarity: wn.bestPair?.pathSimilarity ?? null,
        sharedHypernym: wn.bestPair?.sharedHypernym ?? null,
      },
    };
  }

  if (runAll || algorithm === 'tfidf') {
    const t0 = performance.now();
    const tfidf = computeTfidfCosine(cleanA, cleanB);
    const tfTime = Math.round((performance.now() - t0) * 10) / 10;
    results.tfidf = {
      algorithm: 'tfidf',
      name: 'TF-IDF + Cosine',
      methodCategory: 'Statistical',
      score: tfidf.scorePercent,
      rawCosineOrMetric: tfidf.cosineSimilarity,
      status: 'completed',
      processingTimeMs: tfTime,
      calculationMethod: 'Cosine Similarity = (A · B) / (||A|| × ||B||) on sparse IDF vectors',
      modelName: 'TF-IDF Vectorizer (Smooth IDF + L2 Cosine)',
      explanation:
        tfidf.sharedTerms.length > 0
          ? `Found ${tfidf.sharedTerms.length} shared normalized token(s) (${tfidf.sharedTerms.join(', ')}).`
          : 'No identical surface tokens are shared after normalization, so sparse cosine similarity is 0%.',
      limitation: 'Measures lexical surface overlap only; TF-IDF does not understand synonyms or contextual meanings.',
      technicalDetails: {
        formula: 'cos(A, B) = (A · B) / (||A|| × ||B||)',
        dotProduct: tfidf.dotProduct,
        normA: tfidf.normA,
        normB: tfidf.normB,
        sharedTerms: tfidf.sharedTerms,
        tokensA: tfidf.tokensA,
        tokensB: tfidf.tokensB,
        corpusVocabulary: tfidf.vocabularySize,
      },
    };
  }

  if (runAll || algorithm === 'word2vec' || algorithm === 'embeddings') {
    const t0 = performance.now();
    try {
      const w2v = await computeWord2VecSimilarity(cleanA, cleanB);
      const w2vTime = Math.round((performance.now() - t0) * 10) / 10;
      const oovNotes = [...w2v.oovTokensA, ...w2v.oovTokensB];
      results.word2vec = {
        algorithm: 'word2vec',
        name: 'Word2Vec',
        methodCategory: 'Word embeddings',
        score: w2v.scorePercent,
        rawCosineOrMetric: w2v.cosineSimilarity,
        status: 'completed',
        processingTimeMs: w2vTime,
        calculationMethod: '256-dim static word vector extraction + L2 Cosine Similarity',
        modelName: w2v.modelName,
        explanation:
          oovNotes.length > 0
            ? `Mapped tokens into ${w2v.dimensions}-dim static word space (sub-piece OOV fallback applied for: ${oovNotes.join(', ')}).`
            : `Extracted ${w2v.dimensions}-dim static word vectors learned from distributional co-occurrence.`,
        limitation: 'Assigns one static vector per word regardless of surrounding sentence context and struggles with unseen whole words.',
        technicalDetails: {
          dimensions: w2v.dimensions,
          rawCosine: w2v.cosineSimilarity,
          tokensA: w2v.tokensA,
          tokensB: w2v.tokensB,
          oovHandledTokens: oovNotes,
          contextual: false,
        },
      };
    } catch (err) {
      results.word2vec = {
        algorithm: 'word2vec',
        name: 'Word2Vec',
        methodCategory: 'Word embeddings',
        score: null,
        rawCosineOrMetric: null,
        status: 'not_available',
        processingTimeMs: Math.round((performance.now() - t0) * 10) / 10,
        calculationMethod: 'Static word vector cosine similarity',
        modelName: 'Distilled Word Embeddings (minishlab/M2V_base_output)',
        explanation: 'Word2Vec model could not process this input.',
        limitation: 'Requires known vocabulary tokens.',
        reasonIfUnavailable: err instanceof Error ? err.message : 'Word2Vec inference failed.',
        technicalDetails: {},
      };
    }
  }

  if (runAll || algorithm === 'fasttext' || algorithm === 'embeddings') {
    const t0 = performance.now();
    try {
      const ft = await computeFastTextSimilarity(cleanA, cleanB);
      const ftTime = Math.round((performance.now() - t0) * 10) / 10;
      results.fasttext = {
        algorithm: 'fasttext',
        name: 'FastText',
        methodCategory: 'Subword embeddings',
        score: ft.scorePercent,
        rawCosineOrMetric: ft.cosineSimilarity,
        status: 'completed',
        processingTimeMs: ftTime,
        calculationMethod: '256-dim subword & character n-gram pooling + Cosine Similarity',
        modelName: ft.modelName,
        explanation: `Encodes morphological subwords and character n-grams into ${ft.dimensions}-dim vectors, handling rare or unfamiliar words gracefully.`,
        limitation: 'Static subword composition captures morphology well but does not disambiguate polysemous words in context.',
        technicalDetails: {
          dimensions: ft.dimensions,
          rawCosine: ft.cosineSimilarity,
          subwordTokensA: ft.subwordTokensA,
          subwordTokensB: ft.subwordTokensB,
          charNgramsA: ft.charNgramsA.slice(0, 8),
          charNgramsB: ft.charNgramsB.slice(0, 8),
        },
      };
      results.embeddings = results.fasttext;
    } catch (err) {
      const errRes: AlgorithmResult = {
        algorithm: 'fasttext',
        name: 'FastText',
        methodCategory: 'Subword embeddings',
        score: null,
        rawCosineOrMetric: null,
        status: 'not_available',
        processingTimeMs: Math.round((performance.now() - t0) * 10) / 10,
        calculationMethod: 'Subword vector cosine similarity',
        modelName: 'Subword FastText Embeddings (minishlab/potion-base-8M)',
        explanation: 'FastText subword model could not process this input.',
        limitation: 'Unavailable during model load failure.',
        reasonIfUnavailable: err instanceof Error ? err.message : 'Subword inference failed.',
        technicalDetails: {},
      };
      results.fasttext = errRes;
      results.embeddings = errRes;
    }
  }

  if (runAll || algorithm === 'transformer') {
    const t0 = performance.now();
    try {
      const trans = await computeTransformerSimilarity(cleanA, cleanB);
      const trTime = Math.round((performance.now() - t0) * 10) / 10;
      results.transformer = {
        algorithm: 'transformer',
        name: 'Sentence-BERT',
        methodCategory: 'Contextual embeddings',
        score: trans.scorePercent,
        rawCosineOrMetric: trans.cosineSimilarity,
        status: 'completed',
        processingTimeMs: trTime,
        calculationMethod: '384-dim contextual transformer mean-pooling + L2 Cosine Similarity',
        modelName: trans.modelName,
        explanation: `Encodes both inputs through multi-head transformer attention into ${trans.dimensions}-dim L2-normalized vectors.`,
        limitation: 'Higher computational footprint than lexical or static vector lookup.',
        technicalDetails: {
          dimensions: trans.dimensions,
          rawCosine: trans.cosineSimilarity,
          pooling: 'mean',
          normalized: true,
        },
      };
    } catch (err) {
      results.transformer = {
        algorithm: 'transformer',
        name: 'Sentence-BERT',
        methodCategory: 'Contextual embeddings',
        score: null,
        rawCosineOrMetric: null,
        status: 'not_available',
        processingTimeMs: Math.round((performance.now() - t0) * 10) / 10,
        calculationMethod: 'Contextual transformer cosine similarity',
        modelName: 'sentence-transformers/all-MiniLM-L6-v2',
        explanation: 'Sentence-BERT transformer model failed to execute inference.',
        limitation: 'Requires pretrained transformer weights.',
        reasonIfUnavailable: err instanceof Error ? err.message : 'Transformer inference failed.',
        technicalDetails: {},
      };
    }
  }

  let primaryScore: number | null = null;
  let primaryAlgorithmUsed = 'Sentence-BERT';

  if (algorithm === 'wordnet') {
    primaryScore = results.wordnet?.score ?? null;
    primaryAlgorithmUsed = 'WordNet (Wu-Palmer)';
  } else if (algorithm === 'tfidf') {
    primaryScore = results.tfidf?.score ?? null;
    primaryAlgorithmUsed = 'TF-IDF + Cosine';
  } else if (algorithm === 'word2vec') {
    primaryScore = results.word2vec?.score ?? null;
    primaryAlgorithmUsed = 'Word2Vec';
  } else if (algorithm === 'fasttext' || algorithm === 'embeddings') {
    primaryScore = results.fasttext?.score ?? null;
    primaryAlgorithmUsed = 'FastText';
  } else {
    if (results.transformer?.score !== null && results.transformer?.score !== undefined) {
      primaryScore = results.transformer.score;
      primaryAlgorithmUsed = 'Sentence-BERT (all-MiniLM-L6-v2)';
    } else if (results.fasttext?.score !== null && results.fasttext?.score !== undefined) {
      primaryScore = results.fasttext.score;
      primaryAlgorithmUsed = 'FastText';
    } else if (results.word2vec?.score !== null && results.word2vec?.score !== undefined) {
      primaryScore = results.word2vec.score;
      primaryAlgorithmUsed = 'Word2Vec';
    } else if (results.wordnet?.score !== null && results.wordnet?.score !== undefined) {
      primaryScore = results.wordnet.score;
      primaryAlgorithmUsed = 'WordNet (Wu-Palmer)';
    } else {
      primaryScore = results.tfidf?.score ?? null;
      primaryAlgorithmUsed = 'TF-IDF + Cosine';
    }
  }

  const relationship = deriveOverallRelationship(
    wn.relationship,
    wn.score,
    results.transformer?.score ?? results.fasttext?.score ?? results.word2vec?.score ?? null,
    cleanA,
    cleanB
  );

  const { explanation, technicalExplanation, beginnerExplanation } = buildWordExplanations(
    cleanA,
    cleanB,
    relationship,
    results,
    primaryScore,
    wn.bestPair?.sharedHypernym
  );

  let vectorGeometry;
  try {
    const contrast =
      ['apple', 'banana', 'orange', 'fruit'].includes(cleanA.toLowerCase()) ||
      ['apple', 'banana', 'orange', 'fruit'].includes(cleanB.toLowerCase())
        ? 'telescope'
        : 'apple';
    vectorGeometry = await computeVectorGeometry2D(cleanA, cleanB, contrast);
  } catch {
    vectorGeometry = undefined;
  }

  const normTokensA = preprocessTokens(cleanA, false);
  const normTokensB = preprocessTokens(cleanB, false);
  const band = getInterpretationBand(primaryScore);

  const pipelineStages: NlpPipelineStage[] = [
    {
      step: 1,
      name: 'Text Input',
      summary: `"${cleanA}" vs "${cleanB}"`,
      details: [`Algorithm scope: ${algorithm}`, `Multiword safe: yes`],
    },
    {
      step: 2,
      name: 'Text Preprocessing',
      summary: `"${cleanA.toLowerCase()}" ↔ "${cleanB.toLowerCase()}"`,
      details: [
        'Lowercased and trimmed boundaries',
        `WordNet synsets: A=${wn.synsetsA.length}, B=${wn.synsetsB.length}`,
      ],
    },
    {
      step: 3,
      name: 'Tokenization',
      summary: `${normTokensA.length} token(s) ↔ ${normTokensB.length} token(s)`,
      details: [
        `Tokens A: [${normTokensA.join(', ')}]`,
        `Tokens B: [${normTokensB.join(', ')}]`,
      ],
    },
    {
      step: 4,
      name: 'Vectorization',
      summary: '384-dim contextual + 256-dim static + sparse TF-IDF',
      details: [
        `Transformer: ${TRANSFORMER_MODEL_ID}`,
        `Subword/Word vectors: 256 dimensions`,
      ],
    },
    {
      step: 5,
      name: 'Similarity Calculation',
      summary: vectorGeometry
        ? `cos(θ) = ${vectorGeometry.cosineSimilarity.toFixed(3)} (θ = ${vectorGeometry.thetaDegrees}°)`
        : `Primary score: ${primaryScore ?? 'N/A'}%`,
      details: [
        'Cosine Similarity = (A · B) / (||A|| × ||B||)',
        `Primary engine: ${primaryAlgorithmUsed}`,
      ],
    },
    {
      step: 6,
      name: 'Result Interpretation',
      summary: band ? `${band.label} · ${relationship}` : relationship,
      details: [
        `Score band: ${band?.range ?? 'N/A'}`,
        `Detected relationship: ${relationship}`,
      ],
    },
  ];

  return {
    word_a: cleanA,
    word_b: cleanB,
    algorithm,
    primaryScore,
    primaryAlgorithmUsed,
    interpretationBand: band,
    relationship,
    explanation,
    technicalExplanation,
    beginnerExplanation,
    results,
    lexicalDetails: {
      synsetsA: wn.synsetsA,
      synsetsB: wn.synsetsB,
      ...(wn.bestPair ? { bestSynsetPair: wn.bestPair } : {}),
    },
    vectorGeometry,
    pipelineStages,
    timestamp: new Date().toISOString(),
  };
}

export async function analyzeSentencePair(
  sentenceA: string,
  sentenceB: string
): Promise<SentenceAnalysisResponse> {
  const cleanA = sentenceA.trim();
  const cleanB = sentenceB.trim();

  const tokensA = preprocessTokens(cleanA, true);
  const tokensB = preprocessTokens(cleanB, true);

  const setA = new Set(tokensA);
  const setB = new Set(tokensB);

  const sharedConcepts = Array.from(setA).filter((t) => setB.has(t));
  const uniqueToA = Array.from(setA).filter((t) => !setB.has(t));
  const uniqueToB = Array.from(setB).filter((t) => !setA.has(t));

  const tfStart = performance.now();
  const tfidf = computeTfidfCosine(cleanA, cleanB);
  const tfidfResult: AlgorithmResult = {
    algorithm: 'tfidf',
    name: 'TF-IDF + Cosine',
    methodCategory: 'Statistical',
    score: tfidf.scorePercent,
    rawCosineOrMetric: tfidf.cosineSimilarity,
    status: 'completed',
    processingTimeMs: Math.round((performance.now() - tfStart) * 10) / 10,
    calculationMethod: 'Sparse TF-IDF weighted vector cosine similarity',
    modelName: 'TF-IDF Vectorizer (Smooth IDF + L2 Cosine)',
    explanation:
      sharedConcepts.length > 0
        ? `Statistical lexical match across ${sharedConcepts.length} shared content word(s).`
        : 'Zero shared non-stopword tokens; TF-IDF cannot bridge paraphrased wording.',
    limitation: 'Cannot recognize paraphrased meanings when sentences use different synonyms.',
    technicalDetails: {
      dotProduct: tfidf.dotProduct,
      normA: tfidf.normA,
      normB: tfidf.normB,
      sharedTerms: tfidf.sharedTerms,
    },
  };

  let transformerResult: AlgorithmResult;
  const trStart = performance.now();
  try {
    const trans = await computeTransformerSimilarity(cleanA, cleanB);
    transformerResult = {
      algorithm: 'transformer',
      name: 'Sentence-BERT',
      methodCategory: 'Contextual embeddings',
      score: trans.scorePercent,
      rawCosineOrMetric: trans.cosineSimilarity,
      status: 'completed',
      processingTimeMs: Math.round((performance.now() - trStart) * 10) / 10,
      calculationMethod: 'Mean-pooled 384-dim contextual sentence embeddings + cosine similarity',
      modelName: trans.modelName,
      explanation: `Encodes full syntactic and contextual meaning into ${trans.dimensions}-dim embeddings using self-attention.`,
      limitation: 'Higher inference cost on large document batches than TF-IDF.',
      technicalDetails: {
        dimensions: trans.dimensions,
        rawCosine: trans.cosineSimilarity,
      },
    };
  } catch (err) {
    transformerResult = {
      algorithm: 'transformer',
      name: 'Sentence-BERT',
      methodCategory: 'Contextual embeddings',
      score: null,
      rawCosineOrMetric: null,
      status: 'not_available',
      processingTimeMs: Math.round((performance.now() - trStart) * 10) / 10,
      modelName: 'sentence-transformers/all-MiniLM-L6-v2',
      explanation: 'Sentence Transformer inference unavailable.',
      limitation: 'Requires pretrained transformer weights.',
      reasonIfUnavailable: err instanceof Error ? err.message : 'Inference failed',
      technicalDetails: {},
    };
  }

  let word2vecResult: AlgorithmResult;
  const w2vStart = performance.now();
  try {
    const w2v = await computeWord2VecSimilarity(cleanA, cleanB);
    word2vecResult = {
      algorithm: 'word2vec',
      name: 'Word2Vec (Mean Pooled)',
      methodCategory: 'Word embeddings',
      score: w2v.scorePercent,
      rawCosineOrMetric: w2v.cosineSimilarity,
      status: 'completed',
      processingTimeMs: Math.round((performance.now() - w2vStart) * 10) / 10,
      calculationMethod: 'Mean-pooled 256-dim static word vectors + cosine similarity',
      modelName: w2v.modelName,
      explanation: `Averages ${w2v.dimensions}-dim static word vectors across sentence tokens without word-order attention.`,
      limitation: 'Ignores word order, negation, and context-dependent word senses.',
      technicalDetails: {
        dimensions: w2v.dimensions,
        rawCosine: w2v.cosineSimilarity,
      },
    };
  } catch (err) {
    word2vecResult = {
      algorithm: 'word2vec',
      name: 'Word2Vec (Mean Pooled)',
      methodCategory: 'Word embeddings',
      score: null,
      rawCosineOrMetric: null,
      status: 'not_available',
      processingTimeMs: Math.round((performance.now() - w2vStart) * 10) / 10,
      modelName: 'Distilled Word Embeddings',
      explanation: 'Word2Vec static embeddings unavailable.',
      reasonIfUnavailable: err instanceof Error ? err.message : 'Inference failed',
      technicalDetails: {},
    };
  }

  let fasttextResult: AlgorithmResult;
  const ftStart = performance.now();
  try {
    const ft = await computeFastTextSimilarity(cleanA, cleanB);
    fasttextResult = {
      algorithm: 'fasttext',
      name: 'FastText (Subword Pooled)',
      methodCategory: 'Subword embeddings',
      score: ft.scorePercent,
      rawCosineOrMetric: ft.cosineSimilarity,
      status: 'completed',
      processingTimeMs: Math.round((performance.now() - ftStart) * 10) / 10,
      calculationMethod: 'Mean-pooled 256-dim subword character vectors + cosine similarity',
      modelName: ft.modelName,
      explanation: `Averages ${ft.dimensions}-dim subword vectors across tokens, preserving morphological similarity.`,
      limitation: 'Lacks cross-word contextual attention compared to Sentence-BERT.',
      technicalDetails: {
        dimensions: ft.dimensions,
        rawCosine: ft.cosineSimilarity,
      },
    };
  } catch (err) {
    fasttextResult = {
      algorithm: 'fasttext',
      name: 'FastText (Subword Pooled)',
      methodCategory: 'Subword embeddings',
      score: null,
      rawCosineOrMetric: null,
      status: 'not_available',
      processingTimeMs: Math.round((performance.now() - ftStart) * 10) / 10,
      modelName: 'Subword FastText Embeddings',
      explanation: 'FastText embeddings unavailable.',
      reasonIfUnavailable: err instanceof Error ? err.message : 'Inference failed',
      technicalDetails: {},
    };
  }

  const primaryScore =
    transformerResult.score ??
    fasttextResult.score ??
    word2vecResult.score ??
    tfidfResult.score;
  const band = getInterpretationBand(primaryScore);

  const tScore = transformerResult.score ?? 0;
  const lScore = tfidfResult.score ?? 0;
  let explanation = '';
  let beginnerExplanation = '';
  if (tScore >= 60 && lScore < 30) {
    explanation = `Both sentences express closely aligned meaning (${tScore}% Sentence-BERT similarity) despite minimal exact word overlap (${lScore}% TF-IDF). The transformer maps paraphrased expressions into nearby regions of vector space.`;
    beginnerExplanation = `Even though these two sentences use different words, they express a very similar idea (${tScore}% similarity).`;
  } else if (tScore >= 60 && lScore >= 30) {
    explanation = `Both contextual embeddings (${tScore}%) and lexical overlap (${lScore}%) indicate strong alignment, supported by shared concepts (${sharedConcepts.join(', ') || 'overlapping terms'}).`;
    beginnerExplanation = `These sentences share both similar wording and similar overall meaning (${tScore}% similarity).`;
  } else if (tScore >= 40) {
    explanation = `The sentences share moderate topical context (${tScore}% semantic similarity) while differing in focus or phrasing (${lScore}% lexical overlap).`;
    beginnerExplanation = `These sentences touch on related themes, though they make somewhat different points (${tScore}% similarity).`;
  } else {
    explanation = `Contextual embeddings (${tScore}%) and lexical vectorization (${lScore}%) indicate that these sentences address largely distinct topics or assertions.`;
    beginnerExplanation = `These sentences talk about different topics and do not convey the same meaning (${tScore}% similarity).`;
  }

  let vectorGeometry: VectorGeometry2D | undefined;
  try {
    const vecs = await encodeBatchTransformer([
      cleanA,
      cleanB,
      'Freshly baked bread cooled on the wooden kitchen counter.',
    ]);
    const cosAB = Math.max(-0.99, Math.min(0.9999, cos_sim(vecs[0], vecs[1])));
    const cosAC = Math.max(-0.99, Math.min(0.9999, cos_sim(vecs[0], vecs[2])));
    const thetaAB = (Math.acos(cosAB) * 180) / Math.PI;
    const thetaAC = (Math.acos(cosAC) * 180) / Math.PI;
    const baseRad = (48 * Math.PI) / 180;
    const secRad = ((48 - thetaAB) * Math.PI) / 180;
    const conRad = ((48 - Math.max(thetaAC, thetaAB + 18)) * Math.PI) / 180;
    vectorGeometry = {
      primaryLabel: 'Sentence A',
      secondaryLabel: 'Sentence B',
      contrastLabel: 'Unrelated topic',
      cosineSimilarity: Math.round(cosAB * 1000) / 1000,
      thetaDegrees: Math.round(thetaAB * 10) / 10,
      contrastCosine: Math.round(cosAC * 1000) / 1000,
      contrastThetaDegrees: Math.round(thetaAC * 10) / 10,
      modelUsed: TRANSFORMER_MODEL_ID,
      points: [
        { label: 'Sentence A', x: Math.cos(baseRad), y: Math.sin(baseRad), role: 'primary', cosineWithPrimary: 1 },
        { label: 'Sentence B', x: Math.cos(secRad), y: Math.sin(secRad), role: 'secondary', cosineWithPrimary: Math.round(cosAB * 1000) / 1000 },
        { label: 'Unrelated topic', x: Math.cos(conRad), y: Math.sin(conRad), role: 'contrast', cosineWithPrimary: Math.round(cosAC * 1000) / 1000 },
      ],
    };
  } catch {
    vectorGeometry = undefined;
  }

  const pipelineStages: NlpPipelineStage[] = [
    {
      step: 1,
      name: 'Text Input',
      summary: `${cleanA.length} chars ↔ ${cleanB.length} chars`,
      details: [`Sentence A: "${cleanA.slice(0, 60)}"`, `Sentence B: "${cleanB.slice(0, 60)}"`],
    },
    {
      step: 2,
      name: 'Text Preprocessing',
      summary: 'Lowercased & filtered stopwords',
      details: ['Removed punctuation & English stopwords', `Shared lemmas: ${sharedConcepts.length}`],
    },
    {
      step: 3,
      name: 'Tokenization',
      summary: `${tokensA.length} content tokens ↔ ${tokensB.length} content tokens`,
      details: [
        `Tokens A: ${tokensA.slice(0, 8).join(', ')}`,
        `Tokens B: ${tokensB.slice(0, 8).join(', ')}`,
      ],
    },
    {
      step: 4,
      name: 'Vectorization',
      summary: '384-dim Sentence-BERT + 256-dim Word2Vec/FastText + TF-IDF',
      details: [`Model: ${TRANSFORMER_MODEL_ID}`, 'L2-normalized embeddings'],
    },
    {
      step: 5,
      name: 'Similarity Calculation',
      summary: vectorGeometry
        ? `cos(θ) = ${vectorGeometry.cosineSimilarity.toFixed(3)} (θ = ${vectorGeometry.thetaDegrees}°)`
        : `Cosine score: ${primaryScore ?? 0}%`,
      details: [
        `Sentence-BERT: ${transformerResult.score ?? 'N/A'}%`,
        `TF-IDF lexical: ${tfidfResult.score}%`,
      ],
    },
    {
      step: 6,
      name: 'Result Interpretation',
      summary: band ? band.label : 'Evaluated',
      details: [`Band: ${band?.range ?? 'N/A'}`, explanation],
    },
  ];

  return {
    sentence_a: cleanA,
    sentence_b: cleanB,
    primaryScore,
    interpretationBand: band,
    results: {
      transformer: transformerResult,
      word2vec: word2vecResult,
      fasttext: fasttextResult,
      tfidf: tfidfResult,
      embeddings: fasttextResult,
    },
    sharedConcepts,
    uniqueToA,
    uniqueToB,
    preprocessedA: tokensA,
    preprocessedB: tokensB,
    explanation,
    technicalExplanation: explanation,
    beginnerExplanation,
    vectorGeometry,
    pipelineStages,
    timestamp: new Date().toISOString(),
  };
}

export function getHealthStatus(): EngineHealthStatus {
  const embHealth = getEmbeddingModelsHealth();
  const anyError =
    embHealth.word2vec.status === 'error' ||
    embHealth.fasttext.status === 'error' ||
    embHealth.transformer.status === 'error';
  const allReady =
    embHealth.word2vec.status === 'ready' &&
    embHealth.fasttext.status === 'ready' &&
    embHealth.transformer.status === 'ready';

  return {
    status: anyError ? 'degraded' : allReady ? 'ready' : 'initializing',
    engines: {
      wordnet: { status: 'ready', database: 'Princeton WordNet 3.1' },
      tfidf: { status: 'ready', corpusSize: 24 },
      word2vec: embHealth.word2vec,
      fasttext: embHealth.fasttext,
      embeddings: embHealth.embeddings,
      transformer: embHealth.transformer,
    },
  };
}

export function getAlgorithmsCatalog() {
  return [
    {
      id: 'wordnet' as const,
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
      id: 'tfidf' as const,
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
      id: 'word2vec' as const,
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
      id: 'fasttext' as const,
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
      id: 'transformer' as const,
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
}

export async function exploreWord(word: string): Promise<SemanticExploreResponse> {
  const base = exploreWordNet(word);
  try {
    const clean = word.trim().toLowerCase();
    const index = await getVocabIndex();
    const [qVec] = await encodeBatchStatic([clean]);
    const scored: Array<{ word: string; score: number }> = [];
    for (let i = 0; i < index.words.length; i++) {
      const cand = index.words[i];
      if (cand === clean) continue;
      let dot = 0;
      const off = i * 256;
      for (let d = 0; d < 256; d++) dot += qVec[d] * index.vectors[off + d];
      scored.push({ word: cand, score: dot });
    }
    scored.sort((a, b) => b.score - a.score);
    const topStage1 = scored.slice(0, 18).map((s) => s.word);
    const tVecs = await encodeBatchTransformer([clean, ...topStage1]);
    const qTrans = tVecs[0];
    const reranked = topStage1
      .map((w, idx) => {
        const cos = Math.max(0, Math.min(1, cos_sim(qTrans, tVecs[idx + 1])));
        return { word: w, cosineSimilarity: Math.round(cos * 1000) / 1000 };
      })
      .sort((a, b) => b.cosineSimilarity - a.cosineSimilarity)
      .slice(0, 6);

    const existingLabels = new Set(base.nodes.map((n) => n.label.toLowerCase()));
    const centerId = `center:${clean}`;
    const embeddingNeighbors: Array<{ word: string; cosineSimilarity: number; gloss: string }> = [];

    for (const item of reranked) {
      const details = getWordDefinitionAndSynonyms(item.word);
      const gloss = details.primaryDefinition || 'Related concept in neural vector space.';
      embeddingNeighbors.push({
        word: item.word.charAt(0).toUpperCase() + item.word.slice(1),
        cosineSimilarity: item.cosineSimilarity,
        gloss,
      });

      if (!existingLabels.has(item.word.toLowerCase()) && base.nodes.length < 26) {
        const nodeId = `emb:${item.word.toLowerCase()}`;
        existingLabels.add(item.word.toLowerCase());
        base.nodes.push({
          id: nodeId,
          label: item.word.charAt(0).toUpperCase() + item.word.slice(1),
          relation: 'embedding',
          pos: details.primaryPos,
          gloss,
          similarityScore: item.cosineSimilarity,
        });
        base.edges.push({
          source: centerId,
          target: nodeId,
          relation: 'embedding',
          label: `Embedding (${item.cosineSimilarity.toFixed(2)})`,
        });
      }
    }

    return {
      ...base,
      found: base.found || embeddingNeighbors.length > 0,
      embeddingNeighbors,
    };
  } catch {
    return base;
  }
}

export async function analyzeWordPairsBatch(params: {
  pairs: BatchWordPairInput[];
  algorithm: AlgorithmChoice;
}): Promise<BatchAnalysisResponse> {
  const startTotal = performance.now();
  const algorithm = params.algorithm || 'all';
  const cleanPairs = params.pairs
    .slice(0, 80)
    .map((p, idx) => ({
      id: p.id || `pair_${idx + 1}`,
      word_a: p.word_a.trim(),
      word_b: p.word_b.trim(),
    }))
    .filter((p) => p.word_a.length > 0 && p.word_b.length > 0);

  const uniqueWords = Array.from(
    new Set(cleanPairs.flatMap((p) => [p.word_a, p.word_b]))
  );
  const idxMap = new Map(uniqueWords.map((w, i) => [w, i]));

  const runTr = algorithm === 'all' || algorithm === 'transformer';
  const runW2v = algorithm === 'all' || algorithm === 'word2vec';
  const runFt = algorithm === 'all' || algorithm === 'fasttext' || algorithm === 'embeddings';

  const [trVecs, w2vVecs, ftVecs] = await Promise.all([
    runTr ? encodeBatchTransformer(uniqueWords).catch(() => null) : Promise.resolve(null),
    runW2v ? encodeBatchWord2Vec(uniqueWords).catch(() => null) : Promise.resolve(null),
    runFt ? encodeBatchStatic(uniqueWords).catch(() => null) : Promise.resolve(null),
  ]);

  const results: BatchWordPairResult[] = cleanPairs.map((pair) => {
    const t0 = performance.now();
    const iA = idxMap.get(pair.word_a)!;
    const iB = idxMap.get(pair.word_b)!;

    const wn = analyzeWordNet(pair.word_a, pair.word_b);
    const tf = computeTfidfCosine(pair.word_a, pair.word_b);

    const w2vScore = w2vVecs
      ? Math.round(Math.max(0, Math.min(1, cos_sim(w2vVecs[iA], w2vVecs[iB]))) * 1000) / 10
      : null;
    const ftScore = ftVecs
      ? Math.round(Math.max(0, Math.min(1, cos_sim(ftVecs[iA], ftVecs[iB]))) * 1000) / 10
      : null;
    const trScore = trVecs
      ? Math.round(Math.max(0, Math.min(1, cos_sim(trVecs[iA], trVecs[iB]))) * 1000) / 10
      : null;

    let primaryScore: number | null = null;
    if (algorithm === 'wordnet') primaryScore = wn.score;
    else if (algorithm === 'tfidf') primaryScore = tf.scorePercent;
    else if (algorithm === 'word2vec') primaryScore = w2vScore;
    else if (algorithm === 'fasttext' || algorithm === 'embeddings') primaryScore = ftScore;
    else primaryScore = trScore ?? ftScore ?? w2vScore ?? wn.score ?? tf.scorePercent;

    const rel = deriveOverallRelationship(
      wn.relationship,
      wn.score,
      trScore ?? ftScore ?? w2vScore,
      pair.word_a,
      pair.word_b
    );
    const band = getInterpretationBand(primaryScore);

    return {
      id: pair.id,
      word_a: pair.word_a,
      word_b: pair.word_b,
      primaryScore,
      rawMetric: primaryScore !== null ? Math.round((primaryScore / 100) * 1000) / 1000 : null,
      bandLabel: band?.label ?? 'Unavailable',
      relationship: rel,
      scores: {
        wordnet: wn.score,
        tfidf: tf.scorePercent,
        word2vec: w2vScore,
        fasttext: ftScore,
        transformer: trScore,
      },
      processingTimeMs: Math.round((performance.now() - t0) * 100) / 100,
    };
  });

  results.sort((a, b) => (b.primaryScore ?? -1) - (a.primaryScore ?? -1));
  const validScores = results
    .map((r) => r.primaryScore)
    .filter((s): s is number => s !== null);

  const averageScore =
    validScores.length > 0
      ? Math.round((validScores.reduce((a, b) => a + b, 0) / validScores.length) * 10) / 10
      : 0;
  const highestScore = validScores.length > 0 ? Math.max(...validScores) : 0;
  const lowestScore = validScores.length > 0 ? Math.min(...validScores) : 0;

  return {
    algorithm,
    pairCount: results.length,
    averageScore,
    highestScore,
    lowestScore,
    totalTimeMs: Math.round((performance.now() - startTotal) * 10) / 10,
    results,
    timestamp: new Date().toISOString(),
  };
}

export async function analyzeDataset(params: {
  documents: DatasetDocumentInput[];
  mode: DatasetMode;
  queryText?: string;
  algorithm: 'transformer' | 'tfidf' | 'embeddings' | 'word2vec' | 'fasttext';
}): Promise<DatasetAnalysisResponse> {
  const { documents, mode, queryText, algorithm } = params;
  const algoName =
    algorithm === 'transformer'
      ? 'Sentence-BERT (all-MiniLM-L6-v2)'
      : algorithm === 'word2vec'
        ? 'Word2Vec (Static Word Embeddings)'
        : algorithm === 'fasttext' || algorithm === 'embeddings'
          ? 'FastText (Subword Embeddings)'
          : 'TF-IDF + Cosine Similarity';

  const getBandLabel = (score: number) =>
    getInterpretationBand(score)?.label ?? 'Evaluated';

  const sharedWords = (a: string, b: string) => {
    const setA = new Set(preprocessTokens(a, true));
    const setB = new Set(preprocessTokens(b, true));
    return Array.from(setA).filter((t) => setB.has(t)).slice(0, 8);
  };

  if (mode === 'compare_two' && documents.length >= 2) {
    const docA = documents[0];
    const docB = documents[1];
    const tfidf = computeTfidfCosine(docA.text, docB.text);
    let transformerScore: number | null = null;
    let embeddingsScore: number | null = null;

    try {
      const tr = await computeTransformerSimilarity(
        docA.text.slice(0, 2000),
        docB.text.slice(0, 2000)
      );
      transformerScore = tr.scorePercent;
    } catch {
      transformerScore = null;
    }

    try {
      const emb =
        algorithm === 'word2vec'
          ? await computeWord2VecSimilarity(docA.text.slice(0, 2000), docB.text.slice(0, 2000))
          : await computeFastTextSimilarity(docA.text.slice(0, 2000), docB.text.slice(0, 2000));
      embeddingsScore = emb.scorePercent;
    } catch {
      embeddingsScore = null;
    }

    const activeScore =
      algorithm === 'transformer'
        ? (transformerScore ?? tfidf.scorePercent)
        : algorithm === 'word2vec' || algorithm === 'fasttext' || algorithm === 'embeddings'
          ? (embeddingsScore ?? tfidf.scorePercent)
          : tfidf.scorePercent;

    const shared = sharedWords(docA.text, docB.text);
    const roundedActive = Math.round(activeScore * 10) / 10;
    return {
      mode,
      algorithm,
      algorithmName: algoName,
      documentCount: 2,
      summary: `Compared "${docA.label}" and "${docB.label}" across ${algoName}, measuring ${activeScore.toFixed(1)}% similarity.`,
      averageScore: activeScore,
      topScore: activeScore,
      documentComparison: {
        docALabel: docA.label,
        docBLabel: docB.label,
        transformerScore,
        tfidfScore: tfidf.scorePercent,
        embeddingsScore,
        sharedKeywords: shared,
      },
      similarityMatrix: {
        labels: [docA.label, docB.label],
        values: [
          [100, roundedActive],
          [roundedActive, 100],
        ],
      },
      timestamp: new Date().toISOString(),
    };
  }

  const MAX_MATRIX_DOCS = 16;

  if (mode === 'query') {
    const cleanQuery = (queryText ?? '').trim();
    const texts = [cleanQuery, ...documents.map((d) => d.text.slice(0, 1200))];
    let scores: number[] = [];
    const matrixLimit = Math.min(documents.length, MAX_MATRIX_DOCS);
    const matrixLabels = ['Query', ...documents.slice(0, matrixLimit).map((d) => d.label)];
    const matrixValues: number[][] = Array.from({ length: matrixLimit + 1 }, () =>
      Array(matrixLimit + 1).fill(0)
    );

    if (algorithm === 'transformer') {
      const vecs = await encodeBatchTransformer(texts);
      const qVec = vecs[0];
      scores = vecs.slice(1).map((v) => Math.round(cosineDense(qVec, v) * 1000) / 10);
      for (let i = 0; i <= matrixLimit; i++) {
        matrixValues[i][i] = 100;
        for (let j = i + 1; j <= matrixLimit; j++) {
          const s = Math.round(cosineDense(vecs[i], vecs[j]) * 1000) / 10;
          matrixValues[i][j] = s;
          matrixValues[j][i] = s;
        }
      }
    } else if (algorithm === 'word2vec' || algorithm === 'fasttext' || algorithm === 'embeddings') {
      const vecs =
        algorithm === 'word2vec'
          ? await encodeBatchWord2Vec(texts)
          : await encodeBatchStatic(texts);
      const qVec = vecs[0];
      scores = vecs.slice(1).map((v) => Math.round(cosineDense(qVec, v) * 1000) / 10);
      for (let i = 0; i <= matrixLimit; i++) {
        matrixValues[i][i] = 100;
        for (let j = i + 1; j <= matrixLimit; j++) {
          const s = Math.round(cosineDense(vecs[i], vecs[j]) * 1000) / 10;
          matrixValues[i][j] = s;
          matrixValues[j][i] = s;
        }
      }
    } else {
      const { vectors, norms } = computeTfidfVectors(texts);
      const qVec = vectors[0];
      const qNorm = norms[0];
      scores = vectors
        .slice(1)
        .map((v, idx) => Math.round(cosineSparse(qVec, qNorm, v, norms[idx + 1]) * 1000) / 10);
      for (let i = 0; i <= matrixLimit; i++) {
        matrixValues[i][i] = 100;
        for (let j = i + 1; j <= matrixLimit; j++) {
          const s =
            Math.round(cosineSparse(vectors[i], norms[i], vectors[j], norms[j]) * 1000) / 10;
          matrixValues[i][j] = s;
          matrixValues[j][i] = s;
        }
      }
    }

    const queryMatches: DatasetQueryMatch[] = documents
      .map((doc, idx) => {
        const score = scores[idx] ?? 0;
        return {
          id: doc.id,
          label: doc.label,
          text: doc.text,
          score,
          bandLabel: getBandLabel(score),
          sharedKeywords: sharedWords(cleanQuery, doc.text),
        };
      })
      .sort((a, b) => b.score - a.score);

    const avg =
      queryMatches.length > 0
        ? Math.round(
            (queryMatches.reduce((acc, m) => acc + m.score, 0) / queryMatches.length) * 10
          ) / 10
        : 0;
    const top = queryMatches[0]?.score ?? 0;

    return {
      mode,
      algorithm,
      algorithmName: algoName,
      documentCount: documents.length,
      queryText: cleanQuery,
      summary: `Ranked ${documents.length} uploaded entries against your query using ${algoName}. Top match: "${queryMatches[0]?.label ?? ''}" (${top.toFixed(1)}%).`,
      averageScore: avg,
      topScore: top,
      queryMatches,
      similarityMatrix: {
        labels: matrixLabels,
        values: matrixValues,
      },
      timestamp: new Date().toISOString(),
    };
  }

  // Pairwise similarity detection inside the uploaded dataset
  const texts = documents.map((d) => d.text.slice(0, 1200));
  const pairs: DatasetPairMatch[] = [];
  const matrixLimit = Math.min(documents.length, MAX_MATRIX_DOCS);
  const matrixLabels = documents.slice(0, matrixLimit).map((d) => d.label);
  const matrixValues: number[][] = Array.from({ length: matrixLimit }, () =>
    Array(matrixLimit).fill(0)
  );

  if (algorithm === 'transformer') {
    const vecs = await encodeBatchTransformer(texts);
    for (let i = 0; i < documents.length; i++) {
      if (i < matrixLimit) matrixValues[i][i] = 100;
      for (let j = i + 1; j < documents.length; j++) {
        const score = Math.round(cosineDense(vecs[i], vecs[j]) * 1000) / 10;
        if (i < matrixLimit && j < matrixLimit) {
          matrixValues[i][j] = score;
          matrixValues[j][i] = score;
        }
        pairs.push({
          docA: { id: documents[i].id, label: documents[i].label, text: documents[i].text },
          docB: { id: documents[j].id, label: documents[j].label, text: documents[j].text },
          score,
          bandLabel: getBandLabel(score),
          sharedKeywords: sharedWords(documents[i].text, documents[j].text),
        });
      }
    }
  } else if (algorithm === 'word2vec' || algorithm === 'fasttext' || algorithm === 'embeddings') {
    const vecs =
      algorithm === 'word2vec'
        ? await encodeBatchWord2Vec(texts)
        : await encodeBatchStatic(texts);
    for (let i = 0; i < documents.length; i++) {
      if (i < matrixLimit) matrixValues[i][i] = 100;
      for (let j = i + 1; j < documents.length; j++) {
        const score = Math.round(cosineDense(vecs[i], vecs[j]) * 1000) / 10;
        if (i < matrixLimit && j < matrixLimit) {
          matrixValues[i][j] = score;
          matrixValues[j][i] = score;
        }
        pairs.push({
          docA: { id: documents[i].id, label: documents[i].label, text: documents[i].text },
          docB: { id: documents[j].id, label: documents[j].label, text: documents[j].text },
          score,
          bandLabel: getBandLabel(score),
          sharedKeywords: sharedWords(documents[i].text, documents[j].text),
        });
      }
    }
  } else {
    const { vectors, norms } = computeTfidfVectors(texts);
    for (let i = 0; i < documents.length; i++) {
      if (i < matrixLimit) matrixValues[i][i] = 100;
      for (let j = i + 1; j < documents.length; j++) {
        const score =
          Math.round(cosineSparse(vectors[i], norms[i], vectors[j], norms[j]) * 1000) / 10;
        if (i < matrixLimit && j < matrixLimit) {
          matrixValues[i][j] = score;
          matrixValues[j][i] = score;
        }
        pairs.push({
          docA: { id: documents[i].id, label: documents[i].label, text: documents[i].text },
          docB: { id: documents[j].id, label: documents[j].label, text: documents[j].text },
          score,
          bandLabel: getBandLabel(score),
          sharedKeywords: sharedWords(documents[i].text, documents[j].text),
        });
      }
    }
  }

  pairs.sort((a, b) => b.score - a.score);
  const topPairs = pairs.slice(0, 25);
  const avg =
    pairs.length > 0
      ? Math.round((pairs.reduce((acc, p) => acc + p.score, 0) / pairs.length) * 10) / 10
      : 0;
  const top = topPairs[0]?.score ?? 0;

  return {
    mode: 'pairwise',
    algorithm,
    algorithmName: algoName,
    documentCount: documents.length,
    summary: `Evaluated ${pairs.length} document pairs across ${documents.length} uploaded items using ${algoName}. Highest similarity found: ${top.toFixed(1)}%.`,
    averageScore: avg,
    topScore: top,
    pairMatches: topPairs,
    similarityMatrix: {
      labels: matrixLabels,
      values: matrixValues,
    },
    timestamp: new Date().toISOString(),
  };
}
