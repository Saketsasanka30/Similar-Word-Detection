import { ExternalLink, Search, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { AlgorithmChoice, HistoryEntry } from '../types/nlp';

interface HistoryViewProps {
  entries: HistoryEntry[];
  onOpenEntry: (entry: HistoryEntry) => void;
  onDeleteEntry: (id: string) => void;
  onClearAll: () => void;
  onStartNew: () => void;
}

const FILTER_OPTIONS: Array<{ value: 'all_filter' | AlgorithmChoice; label: string }> = [
  { value: 'all_filter', label: 'All Entries' },
  { value: 'all', label: 'Compare All' },
  { value: 'transformer', label: 'Sentence-BERT' },
  { value: 'wordnet', label: 'WordNet' },
  { value: 'word2vec', label: 'Word2Vec' },
  { value: 'fasttext', label: 'FastText' },
  { value: 'tfidf', label: 'TF-IDF' },
];

export function HistoryView({
  entries,
  onOpenEntry,
  onDeleteEntry,
  onClearAll,
  onStartNew,
}: HistoryViewProps) {
  const [search, setSearch] = useState('');
  const [algoFilter, setAlgoFilter] = useState<'all_filter' | AlgorithmChoice>('all_filter');

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return entries.filter((item) => {
      const matchesAlgo = algoFilter === 'all_filter' || item.algorithm === algoFilter;
      const matchesText =
        !q ||
        item.inputA.toLowerCase().includes(q) ||
        item.inputB.toLowerCase().includes(q);
      return matchesAlgo && matchesText;
    });
  }, [entries, search, algoFilter]);

  return (
    <section className="max-w-[1440px] mx-auto px-4 sm:px-8 py-14 sm:py-20">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-8">
        <div>
          <p className="font-mono text-[11px] tracking-[0.16em] text-[#5BA4D6] mb-2">
            ANALYSIS HISTORY
          </p>
          <h2 className="font-display font-bold text-2xl sm:text-4xl tracking-tight text-[#CFDDD8]">
            Saved analyses
          </h2>
          <p className="text-sm text-[#91A4B0] mt-2">
            Stored locally in your browser. Reopen any comparison to inspect its scores and breakdown.
          </p>
        </div>

        {entries.length > 0 && (
          <button
            type="button"
            onClick={onClearAll}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-[#071522] hover:bg-[#E17E7E]/15 text-[#E17E7E] border border-[#E17E7E]/30 text-xs font-medium transition-colors cursor-pointer self-start sm:self-auto"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear All History
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-12 text-center">
          <p className="font-display text-lg font-semibold text-[#CFDDD8]">
            No analyses recorded yet
          </p>
          <p className="text-sm text-[#91A4B0] mt-1.5 max-w-md mx-auto">
            Run a word or sentence similarity analysis and it will automatically appear here for future reference.
          </p>
          <button
            type="button"
            onClick={onStartNew}
            className="mt-6 px-5 py-2.5 rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/35 text-sm font-medium transition-colors cursor-pointer"
          >
            Run First Analysis
          </button>
        </div>
      ) : (
        <div className="space-y-5">
          {/* Search & Filter Controls */}
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="w-4 h-4 text-[#647987] absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search words or sentences..."
                aria-label="Search analysis history"
                className="w-full h-10 pl-10 pr-4 rounded-md bg-[#0B1925] border border-[rgba(207,221,216,0.14)] focus:border-[#5BA4D6] text-sm text-[#CFDDD8] placeholder:text-[#647987]"
              />
            </div>

            <div className="flex flex-wrap items-center gap-1.5">
              {FILTER_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setAlgoFilter(opt.value)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors cursor-pointer ${
                    algoFilter === opt.value
                      ? 'bg-[#143F62] border-[#5BA4D6]/60 text-[#CFDDD8]'
                      : 'bg-[#071522] border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#CFDDD8]'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {filtered.length === 0 ? (
            <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] p-10 text-center text-sm text-[#91A4B0]">
              No matching entries found for your current filter.
            </div>
          ) : (
            <div className="rounded-lg bg-[#0B1925] border border-[rgba(207,221,216,0.12)] overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-[rgba(207,221,216,0.10)] bg-[#071522]/70 text-[11px] font-mono tracking-wider text-[#91A4B0]">
                      <th className="py-3 px-6 font-medium">INPUTS</th>
                      <th className="py-3 px-4 font-medium">TYPE</th>
                      <th className="py-3 px-4 font-medium">ALGORITHM</th>
                      <th className="py-3 px-4 font-medium">PRIMARY SCORE</th>
                      <th className="py-3 px-4 font-medium">DATE</th>
                      <th className="py-3 px-6 font-medium text-right">ACTIONS</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[rgba(207,221,216,0.08)] text-sm">
                    {filtered.map((entry) => (
                      <tr key={entry.id} className="hover:bg-[#071522]/40 transition-colors">
                        <td className="py-4 px-6 max-w-md">
                          <div className="font-display font-medium text-[#CFDDD8] truncate">
                            {entry.inputA} <span className="text-[#5BA4D6] mx-1">↔</span> {entry.inputB}
                          </div>
                          <div className="font-mono text-[11px] text-[#647987] mt-0.5 flex flex-wrap gap-x-3">
                            {entry.scoresSnapshot.wordnet !== undefined && (
                              <span>WN: {entry.scoresSnapshot.wordnet ?? 'N/A'}%</span>
                            )}
                            {entry.scoresSnapshot.tfidf !== undefined && (
                              <span>TF-IDF: {entry.scoresSnapshot.tfidf ?? 'N/A'}%</span>
                            )}
                            {entry.scoresSnapshot.word2vec !== undefined && (
                              <span>W2V: {entry.scoresSnapshot.word2vec ?? 'N/A'}%</span>
                            )}
                            {entry.scoresSnapshot.fasttext !== undefined && (
                              <span>FT: {entry.scoresSnapshot.fasttext ?? 'N/A'}%</span>
                            )}
                            {entry.scoresSnapshot.transformer !== undefined && (
                              <span>S-BERT: {entry.scoresSnapshot.transformer ?? 'N/A'}%</span>
                            )}
                          </div>
                        </td>
                        <td className="py-4 px-4 font-mono text-xs text-[#91A4B0] capitalize">
                          {entry.type}
                        </td>
                        <td className="py-4 px-4 font-mono text-xs text-[#91A4B0] uppercase">
                          {entry.algorithm}
                        </td>
                        <td className="py-4 px-4 font-mono tabular-nums font-semibold text-[#CFDDD8]">
                          {entry.primaryScore !== null ? `${entry.primaryScore.toFixed(1)}%` : 'N/A'}
                        </td>
                        <td className="py-4 px-4 font-mono text-xs text-[#647987] whitespace-nowrap">
                          {new Date(entry.timestamp).toLocaleString()}
                        </td>
                        <td className="py-4 px-6 text-right whitespace-nowrap">
                          <div className="inline-flex items-center gap-2">
                            <button
                              type="button"
                              onClick={() => onOpenEntry(entry)}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#143F62] hover:bg-[#1B5278] text-xs font-medium text-[#CFDDD8] cursor-pointer"
                            >
                              <ExternalLink className="w-3.5 h-3.5" />
                              Open
                            </button>
                            <button
                              type="button"
                              onClick={() => onDeleteEntry(entry.id)}
                              aria-label="Delete history entry"
                              className="inline-flex items-center justify-center p-1.5 rounded bg-[#071522] border border-[rgba(207,221,216,0.12)] text-[#91A4B0] hover:text-[#E17E7E] hover:border-[#E17E7E]/40 cursor-pointer"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
