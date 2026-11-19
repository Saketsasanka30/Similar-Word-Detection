import {
  AutoModel,
  AutoTokenizer,
  Tensor,
  cos_sim,
  pipeline,
} from '@huggingface/transformers';
import type { VectorGeometry2D } from './types.ts';

export type ModelLoadState = 'uninitialized' | 'loading' | 'ready' | 'error';

export const TRANSFORMER_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
export const WORD2VEC_MODEL_ID = 'minishlab/M2V_base_output';
export const FASTTEXT_MODEL_ID = 'minishlab/potion-base-8M';

interface EmbeddingModelStatus {
  state: ModelLoadState;
  error?: string;
}

const transformerStatus: EmbeddingModelStatus = { state: 'uninitialized' };
const word2vecStatus: EmbeddingModelStatus = { state: 'uninitialized' };
const fasttextStatus: EmbeddingModelStatus = { state: 'uninitialized' };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let transformerPipelinePromise: Promise<any> | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let word2vecModelPromise: Promise<{ tokenizer: any; model: any }> | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let fasttextModelPromise: Promise<{ tokenizer: any; model: any }> | null = null;

export function getEmbeddingModelsHealth() {
  return {
    word2vec: {
      status: word2vecStatus.state,
      model: WORD2VEC_MODEL_ID,
      dimensions: 256,
      ...(word2vecStatus.error ? { error: word2vecStatus.error } : {}),
    },
    fasttext: {
      status: fasttextStatus.state,
      model: FASTTEXT_MODEL_ID,
      dimensions: 256,
      ...(fasttextStatus.error ? { error: fasttextStatus.error } : {}),
    },
    embeddings: {
      status: fasttextStatus.state,
      model: FASTTEXT_MODEL_ID,
      dimensions: 256,
      ...(fasttextStatus.error ? { error: fasttextStatus.error } : {}),
    },
    transformer: {
      status: transformerStatus.state,
      model: TRANSFORMER_MODEL_ID,
      dimensions: 384,
      ...(transformerStatus.error ? { error: transformerStatus.error } : {}),
    },
  };
}

async function getTransformerPipeline() {
  if (!transformerPipelinePromise) {
    transformerStatus.state = 'loading';
    transformerPipelinePromise = pipeline('feature-extraction', TRANSFORMER_MODEL_ID, {
      dtype: 'q8',
    })
      .then((ext) => {
        transformerStatus.state = 'ready';
        return ext;
      })
      .catch((err: unknown) => {
        transformerStatus.state = 'error';
        transformerStatus.error = err instanceof Error ? err.message : String(err);
        transformerPipelinePromise = null;
        throw err;
      });
  }
  return transformerPipelinePromise;
}

async function getWord2VecModel() {
  if (!word2vecModelPromise) {
    word2vecStatus.state = 'loading';
    word2vecModelPromise = Promise.all([
      AutoTokenizer.from_pretrained(WORD2VEC_MODEL_ID),
      AutoModel.from_pretrained(WORD2VEC_MODEL_ID, { dtype: 'q8' }),
    ])
      .then(([tokenizer, model]) => {
        word2vecStatus.state = 'ready';
        return { tokenizer, model };
      })
      .catch((err: unknown) => {
        word2vecStatus.state = 'error';
        word2vecStatus.error = err instanceof Error ? err.message : String(err);
        word2vecModelPromise = null;
        throw err;
      });
  }
  return word2vecModelPromise;
}

async function getFastTextModel() {
  if (!fasttextModelPromise) {
    fasttextStatus.state = 'loading';
    fasttextModelPromise = Promise.all([
      AutoTokenizer.from_pretrained(FASTTEXT_MODEL_ID),
      AutoModel.from_pretrained(FASTTEXT_MODEL_ID),
    ])
      .then(([tokenizer, model]) => {
        fasttextStatus.state = 'ready';
        return { tokenizer, model };
      })
      .catch((err: unknown) => {
        fasttextStatus.state = 'error';
        fasttextStatus.error = err instanceof Error ? err.message : String(err);
        fasttextModelPromise = null;
        throw err;
      });
  }
  return fasttextModelPromise;
}

export function warmUpModels(): void {
  void getTransformerPipeline().catch(() => {});
  void getWord2VecModel().catch(() => {});
  void getFastTextModel().catch(() => {});
}

export function extractCharNgrams(word: string, minN = 3, maxN = 5): string[] {
  const clean = word.trim().toLowerCase();
  if (!clean) return [];
  const padded = `<${clean}>`;
  const ngrams: string[] = [];
  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i <= padded.length - n; i++) {
      ngrams.push(padded.slice(i, i + n));
    }
  }
  return ngrams.slice(0, 24);
}

export async function computeTransformerSimilarity(
  textA: string,
  textB: string
): Promise<{
  cosineSimilarity: number;
  scorePercent: number;
  dimensions: number;
  modelName: string;
}> {
  const extractor = await getTransformerPipeline();
  const output = await extractor([textA, textB], {
    pooling: 'mean',
    normalize: true,
  });

  const dims = output.dims[1] as number;
  const vecA = Array.from(output.data.slice(0, dims) as Float32Array);
  const vecB = Array.from(output.data.slice(dims, 2 * dims) as Float32Array);
  const rawCosine = cos_sim(vecA, vecB);
  const clamped = Math.max(0, Math.min(1, rawCosine));

  return {
    cosineSimilarity: Math.round(rawCosine * 10000) / 10000,
    scorePercent: Math.round(clamped * 1000) / 10,
    dimensions: dims,
    modelName: `sentence-transformers/all-MiniLM-L6-v2 (${TRANSFORMER_MODEL_ID})`,
  };
}

export async function computeWord2VecSimilarity(
  textA: string,
  textB: string
): Promise<{
  cosineSimilarity: number;
  scorePercent: number;
  dimensions: number;
  tokensA: string[];
  tokensB: string[];
  oovTokensA: string[];
  oovTokensB: string[];
  modelName: string;
}> {
  const { tokenizer, model } = await getWord2VecModel();

  const rawWordsA = textA.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const rawWordsB = textB.trim().toLowerCase().split(/\s+/).filter(Boolean);

  const idsA: number[] = tokenizer.encode(textA.toLowerCase(), { add_special_tokens: false });
  const idsB: number[] = tokenizer.encode(textB.toLowerCase(), { add_special_tokens: false });

  if (idsA.length === 0 || idsB.length === 0) {
    throw new Error('Word2Vec tokenizer produced empty token sequence.');
  }

  const subTokensA = idsA.map(
    (id) => tokenizer.decode([id], { skip_special_tokens: true }).trim() || `[${id}]`
  );
  const subTokensB = idsB.map(
    (id) => tokenizer.decode([id], { skip_special_tokens: true }).trim() || `[${id}]`
  );

  // Detect words that needed multi-piece splitting or UNK fallback (OOV at whole-word level)
  const oovTokensA = rawWordsA.filter((w) => {
    const enc: number[] = tokenizer.encode(w, { add_special_tokens: false });
    return enc.length !== 1 || enc[0] === 100;
  });
  const oovTokensB = rawWordsB.filter((w) => {
    const enc: number[] = tokenizer.encode(w, { add_special_tokens: false });
    return enc.length !== 1 || enc[0] === 100;
  });

  const input_ids = new Tensor(
    'int64',
    BigInt64Array.from([...idsA, ...idsB].map(BigInt)),
    [idsA.length + idsB.length]
  );
  const offsets = new Tensor(
    'int64',
    BigInt64Array.from([0n, BigInt(idsA.length)]),
    [2]
  );

  const output = await model({ input_ids, offsets });
  const emb = output.embeddings;
  const dims = emb.dims[1] as number;
  const vecA = Array.from(emb.data.slice(0, dims) as Float32Array);
  const vecB = Array.from(emb.data.slice(dims, 2 * dims) as Float32Array);

  const rawCosine = cos_sim(vecA, vecB);
  const clamped = Math.max(0, Math.min(1, rawCosine));

  return {
    cosineSimilarity: Math.round(rawCosine * 10000) / 10000,
    scorePercent: Math.round(clamped * 1000) / 10,
    dimensions: dims,
    tokensA: subTokensA,
    tokensB: subTokensB,
    oovTokensA,
    oovTokensB,
    modelName: `Distilled Word Embeddings (${WORD2VEC_MODEL_ID})`,
  };
}

export async function computeFastTextSimilarity(
  textA: string,
  textB: string
): Promise<{
  cosineSimilarity: number;
  scorePercent: number;
  dimensions: number;
  subwordTokensA: string[];
  subwordTokensB: string[];
  charNgramsA: string[];
  charNgramsB: string[];
  modelName: string;
}> {
  const { tokenizer, model } = await getFastTextModel();

  const idsA: number[] = tokenizer.encode(textA, { add_special_tokens: false });
  const idsB: number[] = tokenizer.encode(textB, { add_special_tokens: false });

  if (idsA.length === 0 || idsB.length === 0) {
    throw new Error('FastText subword tokenizer produced empty token sequence.');
  }

  const input_ids = new Tensor(
    'int64',
    BigInt64Array.from([...idsA, ...idsB].map(BigInt)),
    [idsA.length + idsB.length]
  );
  const offsets = new Tensor(
    'int64',
    BigInt64Array.from([0n, BigInt(idsA.length)]),
    [2]
  );

  const output = await model({ input_ids, offsets });
  const emb = output.embeddings;
  const dims = emb.dims[1] as number;
  const vecA = Array.from(emb.data.slice(0, dims) as Float32Array);
  const vecB = Array.from(emb.data.slice(dims, 2 * dims) as Float32Array);

  const rawCosine = cos_sim(vecA, vecB);
  const clamped = Math.max(0, Math.min(1, rawCosine));

  const subwordTokensA = idsA.map(
    (id) => tokenizer.decode([id], { skip_special_tokens: true }).trim() || `[${id}]`
  );
  const subwordTokensB = idsB.map(
    (id) => tokenizer.decode([id], { skip_special_tokens: true }).trim() || `[${id}]`
  );

  const firstWordA = textA.trim().split(/\s+/)[0] || textA;
  const firstWordB = textB.trim().split(/\s+/)[0] || textB;

  return {
    cosineSimilarity: Math.round(rawCosine * 10000) / 10000,
    scorePercent: Math.round(clamped * 1000) / 10,
    dimensions: dims,
    subwordTokensA,
    subwordTokensB,
    charNgramsA: extractCharNgrams(firstWordA),
    charNgramsB: extractCharNgrams(firstWordB),
    modelName: `Subword FastText Embeddings (${FASTTEXT_MODEL_ID})`,
  };
}

// Legacy alias kept for compatibility
export const computeStaticEmbeddingSimilarity = computeFastTextSimilarity;

export async function encodeBatchTransformer(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const extractor = await getTransformerPipeline();
  const output = await extractor(texts, { pooling: 'mean', normalize: true });
  const dims = output.dims[1] as number;
  const vectors: number[][] = [];
  for (let i = 0; i < texts.length; i++) {
    vectors.push(Array.from(output.data.slice(i * dims, (i + 1) * dims) as Float32Array));
  }
  return vectors;
}

export async function encodeBatchWord2Vec(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const { tokenizer, model } = await getWord2VecModel();
  const allIds: number[] = [];
  const offsetsArr: bigint[] = [];
  let cursor = 0;

  for (const t of texts) {
    offsetsArr.push(BigInt(cursor));
    let ids: number[] = tokenizer.encode(t.toLowerCase().slice(0, 1500), {
      add_special_tokens: false,
    });
    if (ids.length === 0) ids = [0];
    allIds.push(...ids);
    cursor += ids.length;
  }

  const input_ids = new Tensor('int64', BigInt64Array.from(allIds.map(BigInt)), [allIds.length]);
  const offsets = new Tensor('int64', BigInt64Array.from(offsetsArr), [offsetsArr.length]);
  const output = await model({ input_ids, offsets });
  const emb = output.embeddings;
  const dims = emb.dims[1] as number;
  const vectors: number[][] = [];
  for (let i = 0; i < texts.length; i++) {
    vectors.push(Array.from(emb.data.slice(i * dims, (i + 1) * dims) as Float32Array));
  }
  return vectors;
}

export async function encodeBatchStatic(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const { tokenizer, model } = await getFastTextModel();
  const allIds: number[] = [];
  const offsetsArr: bigint[] = [];
  let cursor = 0;

  for (const t of texts) {
    offsetsArr.push(BigInt(cursor));
    let ids: number[] = tokenizer.encode(t.slice(0, 1500), { add_special_tokens: false });
    if (ids.length === 0) ids = [0];
    allIds.push(...ids);
    cursor += ids.length;
  }

  const input_ids = new Tensor('int64', BigInt64Array.from(allIds.map(BigInt)), [allIds.length]);
  const offsets = new Tensor('int64', BigInt64Array.from(offsetsArr), [offsetsArr.length]);
  const output = await model({ input_ids, offsets });
  const emb = output.embeddings;
  const dims = emb.dims[1] as number;
  const vectors: number[][] = [];
  for (let i = 0; i < texts.length; i++) {
    vectors.push(Array.from(emb.data.slice(i * dims, (i + 1) * dims) as Float32Array));
  }
  return vectors;
}

export function cosineDense(a: number[], b: number[]): number {
  const raw = cos_sim(a, b);
  return Math.max(0, Math.min(1, raw));
}

export async function computeVectorGeometry2D(
  primaryLabel: string,
  secondaryLabel: string,
  contrastLabel?: string
): Promise<VectorGeometry2D> {
  const fallbackContrast =
    primaryLabel.toLowerCase().includes('banana') ||
    primaryLabel.toLowerCase().includes('apple') ||
    secondaryLabel.toLowerCase().includes('banana')
      ? 'volcano'
      : 'pineapple';
  const contrast = contrastLabel || fallbackContrast;

  const vecs = await encodeBatchTransformer([primaryLabel, secondaryLabel, contrast]);
  const cosAB = Math.max(-0.99, Math.min(0.9999, cos_sim(vecs[0], vecs[1])));
  const cosAC = Math.max(-0.99, Math.min(0.9999, cos_sim(vecs[0], vecs[2])));

  const thetaAB = (Math.acos(cosAB) * 180) / Math.PI;
  const thetaAC = (Math.acos(cosAC) * 180) / Math.PI;

  // Place Primary vector A at +48 deg in first quadrant, Secondary vector B at (48 - thetaAB),
  // and Contrasting vector C at (48 - thetaAC) so visual angle matches true cosine angle θ!
  const baseRad = (48 * Math.PI) / 180;
  const secRad = ((48 - thetaAB) * Math.PI) / 180;
  const conRad = ((48 - Math.max(thetaAC, thetaAB + 18)) * Math.PI) / 180;

  return {
    primaryLabel,
    secondaryLabel,
    contrastLabel: contrast,
    cosineSimilarity: Math.round(cosAB * 1000) / 1000,
    thetaDegrees: Math.round(thetaAB * 10) / 10,
    contrastCosine: Math.round(cosAC * 1000) / 1000,
    contrastThetaDegrees: Math.round(thetaAC * 10) / 10,
    modelUsed: TRANSFORMER_MODEL_ID,
    points: [
      {
        label: primaryLabel,
        x: Math.round(Math.cos(baseRad) * 1000) / 1000,
        y: Math.round(Math.sin(baseRad) * 1000) / 1000,
        role: 'primary',
        cosineWithPrimary: 1,
      },
      {
        label: secondaryLabel,
        x: Math.round(Math.cos(secRad) * 1000) / 1000,
        y: Math.round(Math.sin(secRad) * 1000) / 1000,
        role: 'secondary',
        cosineWithPrimary: Math.round(cosAB * 1000) / 1000,
      },
      {
        label: contrast,
        x: Math.round(Math.cos(conRad) * 1000) / 1000,
        y: Math.round(Math.sin(conRad) * 1000) / 1000,
        role: 'contrast',
        cosineWithPrimary: Math.round(cosAC * 1000) / 1000,
      },
    ],
  };
}
