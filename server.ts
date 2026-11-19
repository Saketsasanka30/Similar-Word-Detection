import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  analyzeDataset,
  analyzeSentencePair,
  analyzeWordPair,
  analyzeWordPairsBatch,
  exploreWord,
  getAlgorithmsCatalog,
  getHealthStatus,
} from './server/nlp/analyzer.ts';
import { analyzeWordInContext } from './server/nlp/contextWsd.ts';
import { warmUpModels } from './server/nlp/embeddings.ts';
import {
  getBenchmarkDatasetsMeta,
  runBenchmarkEvaluation,
} from './server/nlp/evaluation.ts';
import { trainAndPredictCorpusModel } from './server/nlp/contextCorpusModel.ts';
import { findSimilarWords, warmUpSimilarWordsIndex } from './server/nlp/similarWords.ts';
import type {
  AlgorithmChoice,
  BatchWordPairInput,
  BenchmarkDatasetId,
  BenchmarkPairInput,
  DatasetDocumentInput,
  DatasetMode,
} from './server/nlp/types.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const VALID_ALGORITHMS = new Set<AlgorithmChoice>([
  'wordnet',
  'tfidf',
  'word2vec',
  'fasttext',
  'embeddings',
  'transformer',
  'all',
]);

function validateWordInput(
  value: unknown,
  fieldName: string
): { ok: true; value: string } | { ok: false; error: string } {
  if (typeof value !== 'string') {
    return { ok: false, error: `${fieldName} must be a text string.` };
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return { ok: false, error: `${fieldName} is required and cannot be empty.` };
  }
  if (trimmed.length > 80) {
    return { ok: false, error: `${fieldName} must be 80 characters or fewer.` };
  }
  if (!/^[a-zA-Z][a-zA-Z\s'-]*$/.test(trimmed)) {
    return {
      ok: false,
      error: `${fieldName} must contain alphabetic characters (spaces, hyphens, or apostrophes permitted for multiword expressions).`,
    };
  }
  return { ok: true, value: trimmed };
}

function validateSentenceInput(
  value: unknown,
  fieldName: string
): { ok: true; value: string } | { ok: false; error: string } {
  if (typeof value !== 'string') {
    return { ok: false, error: `${fieldName} must be a text string.` };
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return { ok: false, error: `${fieldName} is required and cannot be empty.` };
  }
  if (trimmed.length < 3) {
    return { ok: false, error: `${fieldName} must be at least 3 characters long.` };
  }
  if (trimmed.length > 600) {
    return { ok: false, error: `${fieldName} must be 600 characters or fewer.` };
  }
  return { ok: true, value: trimmed };
}

async function createServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;
  const isProd = process.env.NODE_ENV === 'production';

  app.use(express.json({ limit: '5mb' }));

  // Health & engine model readiness endpoint
  app.get('/api/health', (_req, res) => {
    res.json(getHealthStatus());
  });

  // Algorithm catalog endpoint
  app.get('/api/algorithms', (_req, res) => {
    res.json({ algorithms: getAlgorithmsCatalog() });
  });

  // Word similarity analysis
  app.post('/api/analyze/word', async (req, res) => {
    try {
      const checkA = validateWordInput(req.body?.word_a, 'Word A');
      if (!checkA.ok) {
        return res.status(400).json({ error: checkA.error, code: 'INVALID_WORD_A' });
      }
      const checkB = validateWordInput(req.body?.word_b, 'Word B');
      if (!checkB.ok) {
        return res.status(400).json({ error: checkB.error, code: 'INVALID_WORD_B' });
      }

      const rawAlgo = req.body?.algorithm ?? 'all';
      const algorithm: AlgorithmChoice = VALID_ALGORITHMS.has(rawAlgo) ? rawAlgo : 'all';

      const result = await analyzeWordPair(checkA.value, checkB.value, algorithm);
      return res.json(result);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected server error during word analysis.';
      return res.status(500).json({ error: message, code: 'WORD_ANALYSIS_FAILED' });
    }
  });

  // Unified 5-algorithm comparison endpoint (supports words or sentences)
  app.post('/api/analyze/compare', async (req, res) => {
    try {
      if (typeof req.body?.sentence_a === 'string' && typeof req.body?.sentence_b === 'string') {
        const checkSA = validateSentenceInput(req.body.sentence_a, 'Sentence A');
        if (!checkSA.ok) {
          return res.status(400).json({ error: checkSA.error, code: 'INVALID_SENTENCE_A' });
        }
        const checkSB = validateSentenceInput(req.body.sentence_b, 'Sentence B');
        if (!checkSB.ok) {
          return res.status(400).json({ error: checkSB.error, code: 'INVALID_SENTENCE_B' });
        }
        const result = await analyzeSentencePair(checkSA.value, checkSB.value);
        return res.json(result);
      }

      const checkA = validateWordInput(req.body?.word_a, 'Word A');
      if (!checkA.ok) {
        return res.status(400).json({ error: checkA.error, code: 'INVALID_WORD_A' });
      }
      const checkB = validateWordInput(req.body?.word_b, 'Word B');
      if (!checkB.ok) {
        return res.status(400).json({ error: checkB.error, code: 'INVALID_WORD_B' });
      }

      const result = await analyzeWordPair(checkA.value, checkB.value, 'all');
      return res.json(result);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Unexpected server error during comparative analysis.';
      return res.status(500).json({ error: message, code: 'COMPARE_ANALYSIS_FAILED' });
    }
  });

  // Sentence similarity analysis
  app.post('/api/analyze/sentence', async (req, res) => {
    try {
      const checkA = validateSentenceInput(req.body?.sentence_a, 'Sentence A');
      if (!checkA.ok) {
        return res.status(400).json({ error: checkA.error, code: 'INVALID_SENTENCE_A' });
      }
      const checkB = validateSentenceInput(req.body?.sentence_b, 'Sentence B');
      if (!checkB.ok) {
        return res.status(400).json({ error: checkB.error, code: 'INVALID_SENTENCE_B' });
      }

      const result = await analyzeSentencePair(checkA.value, checkB.value);
      return res.json(result);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected server error during sentence analysis.';
      return res.status(500).json({ error: message, code: 'SENTENCE_ANALYSIS_FAILED' });
    }
  });

  // Contextual Word Detection (Word Sense Disambiguation)
  app.post('/api/analyze/context', async (req, res) => {
    try {
      const wordCheck = validateWordInput(req.body?.word, 'Target Word');
      if (!wordCheck.ok) {
        return res.status(400).json({ error: wordCheck.error, code: 'INVALID_CONTEXT_WORD' });
      }
      const s1Check = validateSentenceInput(req.body?.sentence_a, 'Sentence 1');
      if (!s1Check.ok) {
        return res.status(400).json({ error: s1Check.error, code: 'INVALID_SENTENCE_1' });
      }

      let sentenceB: string | undefined;
      if (typeof req.body?.sentence_b === 'string' && req.body.sentence_b.trim().length > 0) {
        const s2Check = validateSentenceInput(req.body.sentence_b, 'Sentence 2');
        if (!s2Check.ok) {
          return res.status(400).json({ error: s2Check.error, code: 'INVALID_SENTENCE_2' });
        }
        sentenceB = s2Check.value;
      }

      const result = await analyzeWordInContext({
        word: wordCheck.value,
        sentence_a: s1Check.value,
        sentence_b: sentenceB,
      });
      return res.json(result);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected error during contextual word detection.';
      return res.status(500).json({ error: message, code: 'CONTEXT_ANALYSIS_FAILED' });
    }
  });

  // Batch CSV word-pair similarity analysis
  app.post('/api/analyze/batch', async (req, res) => {
    try {
      const rawPairs = Array.isArray(req.body?.pairs) ? req.body.pairs : [];
      const pairs: BatchWordPairInput[] = rawPairs
        .slice(0, 80)
        .map((p: unknown, idx: number) => {
          const item = p as Partial<BatchWordPairInput>;
          return {
            id: typeof item?.id === 'string' && item.id ? item.id : `pair_${idx + 1}`,
            word_a: typeof item?.word_a === 'string' ? item.word_a.trim() : '',
            word_b: typeof item?.word_b === 'string' ? item.word_b.trim() : '',
          };
        })
        .filter((p: BatchWordPairInput) => p.word_a.length > 0 && p.word_b.length > 0);

      if (pairs.length < 1) {
        return res.status(400).json({
          error: 'Please provide at least 1 valid word pair (word_a, word_b).',
          code: 'EMPTY_BATCH_PAIRS',
        });
      }

      const rawAlgo = req.body?.algorithm ?? 'all';
      const algorithm: AlgorithmChoice = VALID_ALGORITHMS.has(rawAlgo) ? rawAlgo : 'all';
      const result = await analyzeWordPairsBatch({ pairs, algorithm });
      return res.json(result);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected server error during batch analysis.';
      return res.status(500).json({ error: message, code: 'BATCH_ANALYSIS_FAILED' });
    }
  });

  // Dataset & Document Upload similarity analysis
  app.post('/api/analyze/dataset', async (req, res) => {
    try {
      const rawDocs = Array.isArray(req.body?.documents) ? req.body.documents : [];
      const documents: DatasetDocumentInput[] = rawDocs
        .slice(0, 80)
        .map((d: unknown, idx: number) => {
          const item = d as Partial<DatasetDocumentInput>;
          return {
            id: typeof item?.id === 'string' && item.id ? item.id : `doc_${idx + 1}`,
            label:
              typeof item?.label === 'string' && item.label.trim()
                ? item.label.trim().slice(0, 120)
                : `Document ${idx + 1}`,
            text: typeof item?.text === 'string' ? item.text.trim() : '',
          };
        })
        .filter((d: DatasetDocumentInput) => d.text.length >= 2);

      if (documents.length < 2) {
        return res.status(400).json({
          error: 'Please provide at least 2 non-empty text entries or documents to analyze.',
          code: 'INSUFFICIENT_DOCUMENTS',
        });
      }

      const mode: DatasetMode = ['query', 'pairwise', 'compare_two'].includes(req.body?.mode)
        ? req.body.mode
        : 'pairwise';

      if (mode === 'query') {
        const q = typeof req.body?.queryText === 'string' ? req.body.queryText.trim() : '';
        if (q.length < 2) {
          return res.status(400).json({
            error: 'Please enter a search query or target text to compare against the dataset.',
            code: 'INVALID_QUERY_TEXT',
          });
        }
      }

      const algo = ['transformer', 'tfidf', 'embeddings', 'word2vec', 'fasttext'].includes(
        req.body?.algorithm
      )
        ? req.body.algorithm
        : 'transformer';

      const result = await analyzeDataset({
        documents,
        mode,
        queryText: req.body?.queryText,
        algorithm: algo,
      });
      return res.json(result);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected server error during dataset analysis.';
      return res.status(500).json({ error: message, code: 'DATASET_ANALYSIS_FAILED' });
    }
  });

  // Python-equivalent Word-Context Vector (window=2) + Cosine Similarity Corpus Trainer & Predictor
  app.post('/api/corpus/predict', (req, res) => {
    try {
      const rawWord = typeof req.body?.word === 'string' ? req.body.word : '';
      if (!rawWord.trim()) {
        return res.status(400).json({ error: 'Please enter an input word.', code: 'EMPTY_WORD' });
      }
      if (rawWord.trim().length > 60) {
        return res
          .status(400)
          .json({ error: 'Word must be 60 characters or fewer.', code: 'WORD_TOO_LONG' });
      }

      let customCorpus: string[] | undefined;
      if (Array.isArray(req.body?.corpus)) {
        const cleaned = req.body.corpus
          .filter((item: unknown): item is string => typeof item === 'string' && item.trim().length > 0)
          .map((s: string) => s.trim())
          .slice(0, 500);
        if (cleaned.length > 0) {
          customCorpus = cleaned;
        }
      } else if (typeof req.body?.corpus === 'string' && req.body.corpus.trim()) {
        const lines = req.body.corpus
          .split(/\r?\n/)
          .map((l: string) => l.trim())
          .filter(Boolean)
          .slice(0, 500);
        if (lines.length > 0) {
          customCorpus = lines;
        }
      }

      const topK = req.body?.top_k === 10 ? 10 : 5;
      const windowSize =
        typeof req.body?.window === 'number' && req.body.window >= 1 && req.body.window <= 5
          ? Math.floor(req.body.window)
          : 2;

      const result = trainAndPredictCorpusModel({
        targetWord: rawWord,
        corpus: customCorpus,
        topK,
        windowSize,
      });

      return res.json(result);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unable to train corpus model or predict similar words.';
      return res.status(500).json({ error: message, code: 'CORPUS_PREDICT_FAILED' });
    }
  });

  // Single-word Similar Word Detection & Smart Synonym Finder
  app.post('/api/similar-words', async (req, res) => {
    try {
      const rawWord = typeof req.body?.word === 'string' ? req.body.word : '';
      if (!rawWord.trim()) {
        return res.status(400).json({ error: 'Please enter a word to search.', code: 'EMPTY_WORD' });
      }
      if (rawWord.trim().length > 60) {
        return res
          .status(400)
          .json({ error: 'Word must be 60 characters or fewer.', code: 'WORD_TOO_LONG' });
      }
      const topK = req.body?.top_k === 10 ? 10 : 5;
      const result = await findSimilarWords(rawWord, topK);
      return res.json(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to find similar words.';
      return res.status(400).json({ error: message, code: 'SIMILAR_WORDS_FAILED' });
    }
  });

  // Semantic Explorer graph endpoint (WordNet + Embedding neighbors)
  app.get('/api/explore', async (req, res) => {
    try {
      const word = typeof req.query.word === 'string' ? req.query.word : '';
      const check = validateWordInput(word, 'Query word');
      if (!check.ok) {
        return res.status(400).json({ error: check.error, code: 'INVALID_EXPLORE_WORD' });
      }
      const graph = await exploreWord(check.value);
      return res.json(graph);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected error exploring WordNet graph.';
      return res.status(500).json({ error: message, code: 'EXPLORE_FAILED' });
    }
  });

  // Semantic related endpoint matching requested REST path
  app.get('/api/semantic/related/:word', async (req, res) => {
    try {
      const check = validateWordInput(req.params.word, 'Query word');
      if (!check.ok) {
        return res.status(400).json({ error: check.error, code: 'INVALID_EXPLORE_WORD' });
      }
      const [graph, similar] = await Promise.all([
        exploreWord(check.value),
        findSimilarWords(check.value, 10).catch(() => null),
      ]);
      return res.json({
        ...graph,
        rankedSuggestions: similar?.results ?? [],
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected error retrieving semantic relations.';
      return res.status(500).json({ error: message, code: 'SEMANTIC_RELATED_FAILED' });
    }
  });

  // Algorithm Evaluation Lab endpoints
  app.get('/api/evaluate/benchmarks', (_req, res) => {
    res.json({ datasets: getBenchmarkDatasetsMeta() });
  });

  app.post('/api/evaluate', async (req, res) => {
    try {
      const rawId = req.body?.datasetId;
      const datasetId: BenchmarkDatasetId =
        rawId === 'simlex999' || rawId === 'custom' ? rawId : 'wordsim353';

      let customPairs: BenchmarkPairInput[] | undefined;
      if (datasetId === 'custom') {
        const raw = Array.isArray(req.body?.pairs) ? req.body.pairs : [];
        customPairs = raw
          .map((item: unknown): BenchmarkPairInput | null => {
            const r = item as Partial<BenchmarkPairInput>;
            const score = Number(r?.human_score);
            if (
              typeof r?.word_a === 'string' &&
              r.word_a.trim() &&
              typeof r?.word_b === 'string' &&
              r.word_b.trim() &&
              Number.isFinite(score)
            ) {
              return {
                word_a: r.word_a.trim(),
                word_b: r.word_b.trim(),
                human_score: score,
              };
            }
            return null;
          })
          .filter((x: BenchmarkPairInput | null): x is BenchmarkPairInput => x !== null);

        if (!customPairs || customPairs.length < 3) {
          return res.status(400).json({
            error:
              'Custom benchmark CSV must contain at least 3 valid rows with word_a, word_b, and human_score.',
            code: 'INVALID_CUSTOM_BENCHMARK',
          });
        }
      }

      const result = await runBenchmarkEvaluation({ datasetId, customPairs });
      return res.json(result);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unexpected server error during evaluation.';
      return res.status(500).json({ error: message, code: 'EVALUATION_FAILED' });
    }
  });

  // Unknown API routes return structured JSON 404
  app.all('/api/*', (_req, res) => {
    res.status(404).json({ error: 'API endpoint not found.', code: 'NOT_FOUND' });
  });

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      const indexFile = path.join(distDir, 'index.html');
      if (fs.existsSync(indexFile)) {
        res.sendFile(indexFile);
      } else {
        res.status(404).send('Build output not found. Run npm run build first.');
      }
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`SEMANTIQ server running on http://0.0.0.0:${PORT}`);
    warmUpModels();
    warmUpSimilarWordsIndex();
  });
}

createServer().catch((err) => {
  console.error('Failed to start SEMANTIQ server:', err);
  process.exit(1);
});
