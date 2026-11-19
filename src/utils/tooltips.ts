export interface GlossaryEntry {
  term: string;
  short: string;
}

export const NLP_GLOSSARY: Record<string, GlossaryEntry> = {
  'Cosine Similarity': {
    term: 'Cosine Similarity',
    short:
      'Measures the cosine of the angle between two vectors: 1.0 means identical direction, 0.0 means orthogonal (unrelated). It is a geometric alignment metric, not a probability.',
  },
  'Word Embeddings': {
    term: 'Word Embeddings',
    short:
      'Dense numerical vectors learned from large text corpora so that words appearing in similar linguistic contexts occupy nearby coordinates in vector space.',
  },
  'TF-IDF': {
    term: 'TF-IDF',
    short:
      'Term Frequency–Inverse Document Frequency weights words by how often they appear in an input relative to a background corpus. It measures exact lexical overlap, not synonymy.',
  },
  WordNet: {
    term: 'WordNet',
    short:
      'A curated lexical database of English nouns, verbs, adjectives, and adverbs grouped into synonym sets (synsets) linked by hierarchical hypernym/hyponym relations.',
  },
  'Semantic Similarity': {
    term: 'Semantic Similarity',
    short:
      'How closely two words or sentences align in conceptual meaning, even when they use completely different surface spelling or wording.',
  },
  'Contextual Embeddings': {
    term: 'Contextual Embeddings',
    short:
      'Neural transformer vectors generated dynamically with self-attention so the representation adapts to surrounding words and sentence phrasing.',
  },
};
