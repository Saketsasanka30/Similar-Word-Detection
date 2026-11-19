// Real TF-IDF + Cosine Similarity engine with stopword removal and reference corpus IDF

export const ENGLISH_STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
  'has', 'have', 'he', 'her', 'his', 'in', 'is', 'it', 'its', 'of', 'on',
  'or', 'she', 'that', 'the', 'their', 'them', 'they', 'this', 'to', 'was',
  'were', 'will', 'with', 'you', 'your', 'can', 'could', 'do', 'does', 'did',
  'if', 'into', 'more', 'not', 'only', 'other', 'such', 'than', 'then',
  'there', 'these', 'those', 'through', 'under', 'up', 'very', 'what', 'when',
  'where', 'which', 'who', 'why', 'would', 'about', 'after', 'all', 'also',
]);

// Reference background corpus (24 domain-diverse sentences) so IDF weights
// reflect realistic term specificity rather than a degenerate 2-document corpus.
export const REFERENCE_CORPUS = [
  'artificial intelligence and machine learning are transforming modern healthcare diagnostics',
  'natural language processing analyzes syntax semantics and contextual word relationships',
  'automobiles trucks motorcycles and buses are common road transport vehicles',
  'joyful and happy emotions describe positive human psychological states',
  'a financial bank manages deposits loans investments and currency accounts',
  'a river bank forms the sloping earth boundary beside flowing water',
  'monarchs kings and queens historically governed sovereign royal realms',
  'fresh fruits such as apples bananas oranges and pears grow on trees',
  'domestic pets including dogs and cats share households with human companions',
  'deep neural networks learn distributed vector representations from large text corpora',
  'cosine similarity measures the angle between two nonzero feature vectors',
  'lexical databases organize words into synonym sets and hierarchical hypernym trees',
  'engineers design combustion and electric engines to power mechanical vehicles',
  'clinical researchers evaluate medical treatments and patient recovery outcomes',
  'semantic search retrieves documents based on conceptual meaning rather than exact keywords',
  'statistical models weight frequent terms using inverse document frequency',
  'software developers build scalable web applications and distributed APIs',
  'astronomers observe distant galaxies planets and interstellar dust clouds',
  'ocean currents regulate global climate patterns and marine ecosystems',
  'architects plan sustainable urban buildings with structural steel and glass',
  'musicians compose harmonic melodies across acoustic and electronic instruments',
  'economists study market supply demand inflation and international trade',
  'linguists compare morphological rules and phonetic structures across languages',
  'athletes train endurance strength and agility for competitive tournaments',
];

export function preprocessTokens(text: string, removeStopwords = true): string[] {
  const normalized = text
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (!normalized) return [];
  const rawTokens = normalized.split(' ').filter((t) => t.length > 0);

  if (!removeStopwords) return rawTokens;
  const filtered = rawTokens.filter((t) => !ENGLISH_STOPWORDS.has(t));
  // Fallback if all tokens were stopwords (e.g. comparing "to" and "for")
  return filtered.length > 0 ? filtered : rawTokens;
}

export interface TfidfResult {
  cosineSimilarity: number;
  scorePercent: number;
  tokensA: string[];
  tokensB: string[];
  sharedTerms: string[];
  dotProduct: number;
  normA: number;
  normB: number;
  vocabularySize: number;
  topWeightsA: Array<{ term: string; weight: number }>;
  topWeightsB: Array<{ term: string; weight: number }>;
}

export function computeTfidfCosine(textA: string, textB: string): TfidfResult {
  const tokensA = preprocessTokens(textA, true);
  const tokensB = preprocessTokens(textB, true);

  const corpusTokens = REFERENCE_CORPUS.map((doc) => preprocessTokens(doc, true));
  const allDocs = [tokensA, tokensB, ...corpusTokens];
  const N = allDocs.length;

  // Document frequency across corpus + inputs
  const df = new Map<string, number>();
  for (const doc of allDocs) {
    const unique = new Set(doc);
    for (const term of unique) {
      df.set(term, (df.get(term) ?? 0) + 1);
    }
  }

  // Smooth IDF matching scikit-learn's TfidfVectorizer (smooth_idf=True, sublinear_tf=False):
  // idf(t) = log((1 + N) / (1 + df(t))) + 1
  const idf = new Map<string, number>();
  for (const [term, freq] of df.entries()) {
    idf.set(term, Math.log((1 + N) / (1 + freq)) + 1);
  }

  const buildTfidfVector = (tokens: string[]): Map<string, number> => {
    const tf = new Map<string, number>();
    for (const t of tokens) {
      tf.set(t, (tf.get(t) ?? 0) + 1);
    }
    const vec = new Map<string, number>();
    for (const [term, count] of tf.entries()) {
      const weight = count * (idf.get(term) ?? 1);
      vec.set(term, weight);
    }
    return vec;
  };

  const vecA = buildTfidfVector(tokensA);
  const vecB = buildTfidfVector(tokensB);

  let dotProduct = 0;
  let sumSqA = 0;
  let sumSqB = 0;
  const sharedTerms: string[] = [];

  for (const [term, wA] of vecA.entries()) {
    sumSqA += wA * wA;
    const wB = vecB.get(term);
    if (wB !== undefined) {
      dotProduct += wA * wB;
      sharedTerms.push(term);
    }
  }

  for (const wB of vecB.values()) {
    sumSqB += wB * wB;
  }

  const normA = Math.sqrt(sumSqA);
  const normB = Math.sqrt(sumSqB);
  const cosine = normA > 0 && normB > 0 ? Math.min(1, Math.max(0, dotProduct / (normA * normB))) : 0;

  const topWeightsA = Array.from(vecA.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([term, weight]) => ({ term, weight: Math.round(weight * 1000) / 1000 }));

  const topWeightsB = Array.from(vecB.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([term, weight]) => ({ term, weight: Math.round(weight * 1000) / 1000 }));

  return {
    cosineSimilarity: Math.round(cosine * 10000) / 10000,
    scorePercent: Math.round(cosine * 1000) / 10,
    tokensA,
    tokensB,
    sharedTerms,
    dotProduct: Math.round(dotProduct * 10000) / 10000,
    normA: Math.round(normA * 10000) / 10000,
    normB: Math.round(normB * 10000) / 10000,
    vocabularySize: df.size,
    topWeightsA,
    topWeightsB,
  };
}

export function computeTfidfVectors(documents: string[]): {
  vectors: Array<Map<string, number>>;
  norms: number[];
  tokenSets: Array<Set<string>>;
} {
  const tokenizedDocs = documents.map((d) => preprocessTokens(d, true));
  const corpusDocs = REFERENCE_CORPUS.map((d) => preprocessTokens(d, true));
  const allDocs = [...tokenizedDocs, ...corpusDocs];
  const N = allDocs.length;

  const df = new Map<string, number>();
  for (const doc of allDocs) {
    for (const term of new Set(doc)) {
      df.set(term, (df.get(term) ?? 0) + 1);
    }
  }

  const idf = new Map<string, number>();
  for (const [term, freq] of df.entries()) {
    idf.set(term, Math.log((1 + N) / (1 + freq)) + 1);
  }

  const vectors: Array<Map<string, number>> = [];
  const norms: number[] = [];
  const tokenSets: Array<Set<string>> = [];

  for (const tokens of tokenizedDocs) {
    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    const vec = new Map<string, number>();
    let sumSq = 0;
    for (const [term, count] of tf.entries()) {
      const w = count * (idf.get(term) ?? 1);
      vec.set(term, w);
      sumSq += w * w;
    }
    vectors.push(vec);
    norms.push(Math.sqrt(sumSq));
    tokenSets.push(new Set(tokens));
  }

  return { vectors, norms, tokenSets };
}

export function cosineSparse(
  vecA: Map<string, number>,
  normA: number,
  vecB: Map<string, number>,
  normB: number
): number {
  if (normA === 0 || normB === 0) return 0;
  let dot = 0;
  for (const [term, wA] of vecA.entries()) {
    const wB = vecB.get(term);
    if (wB !== undefined) dot += wA * wB;
  }
  return Math.min(1, Math.max(0, dot / (normA * normB)));
}

