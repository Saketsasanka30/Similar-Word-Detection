export type AlgorithmChoice =
  | 'wordnet'
  | 'tfidf'
  | 'word2vec'
  | 'fasttext'
  | 'embeddings'
  | 'transformer'
  | 'all';

export type CoreAlgorithmId =
  | 'wordnet'
  | 'tfidf'
  | 'word2vec'
  | 'fasttext'
  | 'transformer';

export type ProcessingStatus = 'completed' | 'not_available' | 'error';

export type RelationshipLabel =
  | 'Exact Match'
  | 'Synonym'
  | 'Near Synonym'
  | 'Hypernym'
  | 'Hyponym'
  | 'Related Concept'
  | 'Contextually Related'
  | 'No Direct Relationship Detected';

export interface SynsetSummary {
  id: string;
  pos: 'n' | 'v' | 'a' | 'r';
  posLabel: string;
  lemmas: string[];
  gloss: string;
  examples?: string[];
}

export interface VectorPoint2D {
  label: string;
  x: number;
  y: number;
  role: 'primary' | 'secondary' | 'contrast';
  cosineWithPrimary: number;
}

export interface VectorGeometry2D {
  primaryLabel: string;
  secondaryLabel: string;
  contrastLabel: string;
  cosineSimilarity: number;
  thetaDegrees: number;
  contrastCosine: number;
  contrastThetaDegrees: number;
  points: VectorPoint2D[];
  modelUsed: string;
}

export interface NlpPipelineStage {
  step: number;
  name: string;
  summary: string;
  details: string[];
}

export interface AlgorithmResult {
  algorithm: CoreAlgorithmId | 'embeddings';
  name: string;
  methodCategory:
    | 'Lexical'
    | 'Statistical'
    | 'Word embeddings'
    | 'Subword embeddings'
    | 'Contextual embeddings'
    | 'Vector-based'
    | 'Contextual';
  score: number | null; // 0 to 100 scale, null if not available
  rawCosineOrMetric: number | null; // 0 to 1 scale
  status: ProcessingStatus;
  processingTimeMs?: number;
  calculationMethod?: string;
  modelName: string;
  explanation: string;
  limitation?: string;
  reasonIfUnavailable?: string;
  technicalDetails: Record<string, string | number | string[] | boolean | null>;
}

export interface InterpretationBand {
  label: string;
  range: string;
  tone: 'high' | 'moderate' | 'low';
}

export interface WordAnalysisResponse {
  word_a: string;
  word_b: string;
  algorithm: AlgorithmChoice;
  primaryScore: number | null;
  primaryAlgorithmUsed: string;
  interpretationBand: InterpretationBand | null;
  relationship: RelationshipLabel;
  explanation: string;
  technicalExplanation: string;
  beginnerExplanation: string;
  results: {
    wordnet?: AlgorithmResult;
    tfidf?: AlgorithmResult;
    word2vec?: AlgorithmResult;
    fasttext?: AlgorithmResult;
    embeddings?: AlgorithmResult;
    transformer?: AlgorithmResult;
  };
  lexicalDetails: {
    synsetsA: SynsetSummary[];
    synsetsB: SynsetSummary[];
    bestSynsetPair?: {
      synsetA: SynsetSummary;
      synsetB: SynsetSummary;
      wupSimilarity: number | null;
      pathSimilarity: number | null;
      sharedHypernym?: string | null;
    };
  };
  vectorGeometry?: VectorGeometry2D;
  pipelineStages?: NlpPipelineStage[];
  timestamp: string;
}

export interface SentenceAnalysisResponse {
  sentence_a: string;
  sentence_b: string;
  primaryScore: number | null;
  interpretationBand: InterpretationBand | null;
  results: {
    transformer: AlgorithmResult;
    word2vec: AlgorithmResult;
    fasttext: AlgorithmResult;
    tfidf: AlgorithmResult;
    embeddings: AlgorithmResult;
  };
  sharedConcepts: string[];
  uniqueToA: string[];
  uniqueToB: string[];
  preprocessedA: string[];
  preprocessedB: string[];
  explanation: string;
  technicalExplanation: string;
  beginnerExplanation: string;
  vectorGeometry?: VectorGeometry2D;
  pipelineStages?: NlpPipelineStage[];
  timestamp: string;
}

export interface SemanticGraphNode {
  id: string;
  label: string;
  relation: 'center' | 'synonym' | 'hypernym' | 'hyponym' | 'related' | 'embedding';
  pos: string;
  gloss: string;
  similarityScore?: number;
}

export interface SemanticGraphEdge {
  source: string;
  target: string;
  relation: 'synonym' | 'hypernym' | 'hyponym' | 'related' | 'embedding';
  label: string;
}

export interface SemanticExploreResponse {
  word: string;
  found: boolean;
  synsets: SynsetSummary[];
  embeddingNeighbors?: Array<{ word: string; cosineSimilarity: number; gloss: string }>;
  nodes: SemanticGraphNode[];
  edges: SemanticGraphEdge[];
  message?: string;
}

export interface EngineHealthStatus {
  status: 'ready' | 'initializing' | 'degraded';
  engines: {
    wordnet: { status: 'ready' | 'error'; database: string };
    tfidf: { status: 'ready'; corpusSize: number };
    word2vec: {
      status: 'uninitialized' | 'loading' | 'ready' | 'error';
      model: string;
      dimensions: number;
      error?: string;
    };
    fasttext: {
      status: 'uninitialized' | 'loading' | 'ready' | 'error';
      model: string;
      dimensions: number;
      error?: string;
    };
    embeddings: {
      status: 'uninitialized' | 'loading' | 'ready' | 'error';
      model: string;
      dimensions: number;
      error?: string;
    };
    transformer: {
      status: 'uninitialized' | 'loading' | 'ready' | 'error';
      model: string;
      dimensions: number;
      error?: string;
    };
  };
}

export type DatasetMode = 'query' | 'pairwise' | 'compare_two';

export interface DatasetDocumentInput {
  id: string;
  label: string;
  text: string;
}

export interface DatasetQueryMatch {
  id: string;
  label: string;
  text: string;
  score: number;
  bandLabel: string;
  sharedKeywords: string[];
}

export interface DatasetPairMatch {
  docA: { id: string; label: string; text: string };
  docB: { id: string; label: string; text: string };
  score: number;
  bandLabel: string;
  sharedKeywords: string[];
}

export interface DatasetAnalysisResponse {
  mode: DatasetMode;
  algorithm: 'transformer' | 'tfidf' | 'embeddings' | 'word2vec' | 'fasttext';
  algorithmName: string;
  documentCount: number;
  queryText?: string;
  summary: string;
  averageScore: number;
  topScore: number;
  queryMatches?: DatasetQueryMatch[];
  pairMatches?: DatasetPairMatch[];
  documentComparison?: {
    docALabel: string;
    docBLabel: string;
    transformerScore: number | null;
    tfidfScore: number;
    embeddingsScore: number | null;
    sharedKeywords: string[];
  };
  similarityMatrix?: {
    labels: string[];
    values: number[][];
  };
  timestamp: string;
}

export interface SimilarWordMatch {
  word: string;
  cosineSimilarity: number;
  scorePercent: number;
  matchType: 'Exact Synonym' | 'Closely Related';
  posLabel: string;
  definition: string;
  exampleUsage: string | null;
}

export interface SimilarWordsResponse {
  rawInput: string;
  preprocessedWord: string;
  inputDefinition: string | null;
  topK: 5 | 10;
  algorithm: string;
  modelName: string;
  dimensions: number;
  vocabularySize: number;
  vectorPreview: number[];
  exactSynonymCount: number;
  relatedCount: number;
  results: SimilarWordMatch[];
  vectorGeometry?: VectorGeometry2D;
  pipelineStages?: NlpPipelineStage[];
  timestamp: string;
}

export interface ContextSenseScore {
  synsetId: string;
  posLabel: string;
  lemmas: string[];
  gloss: string;
  exampleUsage: string | null;
  contextualSimilarity: number;
  leskOverlapCount: number;
  combinedScore: number;
}

export interface ContextSentenceResult {
  sentence: string;
  detectedSense: ContextSenseScore;
  alternativeSenses: ContextSenseScore[];
  contextTokens: string[];
}

export interface ContextAnalysisResponse {
  word: string;
  found: boolean;
  totalSenses: number;
  sentence1: ContextSentenceResult | null;
  sentence2: ContextSentenceResult | null;
  sameSenseDetected: boolean | null;
  senseComparisonSummary: string;
  technicalExplanation: string;
  beginnerExplanation: string;
  pipelineStages: NlpPipelineStage[];
  message?: string;
  timestamp: string;
}

export interface BatchWordPairInput {
  id?: string;
  word_a: string;
  word_b: string;
}

export interface BatchWordPairResult {
  id: string;
  word_a: string;
  word_b: string;
  primaryScore: number | null;
  rawMetric: number | null;
  bandLabel: string;
  relationship: RelationshipLabel;
  scores: {
    wordnet: number | null;
    tfidf: number | null;
    word2vec: number | null;
    fasttext: number | null;
    transformer: number | null;
  };
  processingTimeMs: number;
}

export interface BatchAnalysisResponse {
  algorithm: AlgorithmChoice;
  pairCount: number;
  averageScore: number;
  highestScore: number;
  lowestScore: number;
  totalTimeMs: number;
  results: BatchWordPairResult[];
  timestamp: string;
}

export type BenchmarkDatasetId = 'wordsim353' | 'simlex999' | 'custom';

export interface BenchmarkPairInput {
  word_a: string;
  word_b: string;
  human_score: number;
}

export interface EvaluationAlgorithmMetrics {
  algorithm: CoreAlgorithmId;
  name: string;
  category: string;
  spearmanRho: number | null;
  pearsonR: number | null;
  mae: number | null;
  evaluatedCount: number;
  coveragePercent: number;
  processingTimeMs: number;
}

export interface EvaluationPairRow {
  word_a: string;
  word_b: string;
  humanScore: number;
  predictions: Record<CoreAlgorithmId, number | null>;
}

export interface EvaluationResponse {
  datasetId: BenchmarkDatasetId;
  datasetName: string;
  pairCount: number;
  algorithmMetrics: EvaluationAlgorithmMetrics[];
  pairs: EvaluationPairRow[];
  summary: string;
  timestamp: string;
}
