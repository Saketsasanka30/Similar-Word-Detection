import type { HistoryEntry, SentenceAnalysisResponse, WordAnalysisResponse } from '../types/nlp';

const STORAGE_KEY = 'semantiq_history_v1';
const MAX_ENTRIES = 60;

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function persistHistory(entries: HistoryEntry[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
  } catch {
    // Ignore storage quota errors gracefully
  }
}

export function addWordHistoryEntry(payload: WordAnalysisResponse): HistoryEntry[] {
  const entry: HistoryEntry = {
    id: `w_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    type: 'word',
    inputA: payload.word_a,
    inputB: payload.word_b,
    algorithm: payload.algorithm,
    primaryScore: payload.primaryScore,
    scoresSnapshot: {
      wordnet: payload.results.wordnet?.score ?? null,
      tfidf: payload.results.tfidf?.score ?? null,
      word2vec: payload.results.word2vec?.score ?? null,
      fasttext: payload.results.fasttext?.score ?? null,
      embeddings: payload.results.embeddings?.score ?? null,
      transformer: payload.results.transformer?.score ?? null,
    },
    timestamp: payload.timestamp,
    wordPayload: payload,
  };
  const current = loadHistory();
  const next = [entry, ...current].slice(0, MAX_ENTRIES);
  persistHistory(next);
  return next;
}

export function addSentenceHistoryEntry(payload: SentenceAnalysisResponse): HistoryEntry[] {
  const entry: HistoryEntry = {
    id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    type: 'sentence',
    inputA: payload.sentence_a,
    inputB: payload.sentence_b,
    algorithm: 'transformer',
    primaryScore: payload.primaryScore,
    scoresSnapshot: {
      tfidf: payload.results.tfidf.score,
      word2vec: payload.results.word2vec?.score ?? null,
      fasttext: payload.results.fasttext?.score ?? null,
      embeddings: payload.results.embeddings.score,
      transformer: payload.results.transformer.score,
    },
    timestamp: payload.timestamp,
    sentencePayload: payload,
  };
  const current = loadHistory();
  const next = [entry, ...current].slice(0, MAX_ENTRIES);
  persistHistory(next);
  return next;
}

export function deleteHistoryEntry(id: string): HistoryEntry[] {
  const next = loadHistory().filter((item) => item.id !== id);
  persistHistory(next);
  return next;
}

export function clearAllHistory(): HistoryEntry[] {
  persistHistory([]);
  return [];
}
