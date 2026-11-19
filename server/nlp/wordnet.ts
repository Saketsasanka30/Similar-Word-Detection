import fs from 'fs';
import path from 'path';
import type {
  RelationshipLabel,
  SemanticExploreResponse,
  SemanticGraphEdge,
  SemanticGraphNode,
  SynsetSummary,
} from './types.ts';

type PosCode = 'n' | 'v' | 'a' | 'r';

const POS_FILES: Record<PosCode, { index: string; data: string; label: string }> = {
  n: { index: 'index.noun', data: 'data.noun', label: 'Noun' },
  v: { index: 'index.verb', data: 'data.verb', label: 'Verb' },
  a: { index: 'index.adj', data: 'data.adj', label: 'Adjective' },
  r: { index: 'index.adv', data: 'data.adv', label: 'Adverb' },
};

interface RawPointer {
  symbol: string;
  offset: number;
  pos: PosCode;
}

interface ParsedSynset {
  offset: number;
  pos: PosCode;
  id: string;
  lemmas: string[];
  pointers: RawPointer[];
  gloss: string;
}

const dictDir = path.resolve(process.cwd(), 'node_modules/wordnet-db/dict');

const indexCache = new Map<PosCode, Map<string, number[]>>();
const synsetCache = new Map<string, ParsedSynset>();
const fdCache = new Map<PosCode, number>();

function getDataFd(pos: PosCode): number {
  let fd = fdCache.get(pos);
  if (fd === undefined) {
    fd = fs.openSync(path.join(dictDir, POS_FILES[pos].data), 'r');
    fdCache.set(pos, fd);
  }
  return fd;
}

function loadIndex(pos: PosCode): Map<string, number[]> {
  const existing = indexCache.get(pos);
  if (existing) return existing;

  const map = new Map<string, number[]>();
  const filePath = path.join(dictDir, POS_FILES[pos].index);
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');

  for (const line of lines) {
    if (!line || line.startsWith('  ')) continue;
    const parts = line.trim().split(/\s+/);
    if (parts.length < 6) continue;
    const lemma = parts[0];
    const synsetCnt = parseInt(parts[2], 10);
    const pCnt = parseInt(parts[3], 10);
    const offsetsStart = 4 + pCnt + 2;
    const offsets: number[] = [];
    for (let i = 0; i < synsetCnt && offsetsStart + i < parts.length; i++) {
      const off = parseInt(parts[offsetsStart + i], 10);
      if (!Number.isNaN(off)) offsets.push(off);
    }
    if (offsets.length > 0) {
      map.set(lemma, offsets);
    }
  }

  indexCache.set(pos, map);
  return map;
}

function readLineAtOffset(pos: PosCode, offset: number): string {
  const fd = getDataFd(pos);
  const buf = Buffer.alloc(4096);
  let total = '';
  let currentOffset = offset;

  while (true) {
    const bytesRead = fs.readSync(fd, buf, 0, buf.length, currentOffset);
    if (bytesRead === 0) break;
    const chunk = buf.toString('utf8', 0, bytesRead);
    const nlIndex = chunk.indexOf('\n');
    if (nlIndex !== -1) {
      total += chunk.slice(0, nlIndex);
      break;
    }
    total += chunk;
    currentOffset += bytesRead;
  }
  return total;
}

export function getSynset(pos: PosCode, offset: number): ParsedSynset | null {
  const key = `${pos}:${offset}`;
  const cached = synsetCache.get(key);
  if (cached) return cached;

  try {
    const line = readLineAtOffset(pos, offset);
    if (!line) return null;
    const pipeIdx = line.indexOf(' | ');
    const metaPart = pipeIdx !== -1 ? line.slice(0, pipeIdx) : line;
    const gloss = pipeIdx !== -1 ? line.slice(pipeIdx + 3).trim() : '';
    const tokens = metaPart.trim().split(/\s+/);
    if (tokens.length < 6) return null;

    const wCnt = parseInt(tokens[3], 16);
    const lemmas: string[] = [];
    let cursor = 4;
    for (let i = 0; i < wCnt && cursor + 1 < tokens.length; i++) {
      lemmas.push(tokens[cursor].replace(/_/g, ' '));
      cursor += 2;
    }

    const pCnt = parseInt(tokens[cursor], 10);
    cursor += 1;
    const pointers: RawPointer[] = [];
    for (let i = 0; i < pCnt && cursor + 3 < tokens.length; i++) {
      const symbol = tokens[cursor];
      const ptrOffset = parseInt(tokens[cursor + 1], 10);
      const ptrPosRaw = tokens[cursor + 2];
      const ptrPos: PosCode =
        ptrPosRaw === 's' ? 'a' : (['n', 'v', 'a', 'r'].includes(ptrPosRaw) ? (ptrPosRaw as PosCode) : pos);
      if (!Number.isNaN(ptrOffset)) {
        pointers.push({ symbol, offset: ptrOffset, pos: ptrPos });
      }
      cursor += 4;
    }

    const parsed: ParsedSynset = {
      offset,
      pos,
      id: `${lemmas[0]?.toLowerCase().replace(/\s+/g, '_') || offset}.${pos}.${String(offset).slice(-4)}`,
      lemmas,
      pointers,
      gloss,
    };
    synsetCache.set(key, parsed);
    return parsed;
  } catch {
    return null;
  }
}

function normalizeWordForIndex(word: string): string[] {
  const cleaned = word.trim().toLowerCase().replace(/\s+/g, '_');
  const candidates = new Set<string>([cleaned]);
  // Simple English morphological normalization fallback (e.g., plural/tense)
  if (cleaned.endsWith('ies') && cleaned.length > 4) {
    candidates.add(cleaned.slice(0, -3) + 'y');
  } else if (cleaned.endsWith('es') && cleaned.length > 3) {
    candidates.add(cleaned.slice(0, -2));
  } else if (cleaned.endsWith('s') && cleaned.length > 3 && !cleaned.endsWith('ss')) {
    candidates.add(cleaned.slice(0, -1));
  }
  if (cleaned.endsWith('ing') && cleaned.length > 5) {
    candidates.add(cleaned.slice(0, -3));
    candidates.add(cleaned.slice(0, -3) + 'e');
  }
  if (cleaned.endsWith('ed') && cleaned.length > 4) {
    candidates.add(cleaned.slice(0, -2));
    candidates.add(cleaned.slice(0, -1));
  }
  return Array.from(candidates);
}

export function lookupSynsets(word: string): ParsedSynset[] {
  const candidates = normalizeWordForIndex(word);
  const results: ParsedSynset[] = [];
  const seen = new Set<string>();

  for (const pos of ['n', 'v', 'a', 'r'] as PosCode[]) {
    const idx = loadIndex(pos);
    for (const candidate of candidates) {
      const offsets = idx.get(candidate);
      if (offsets) {
        for (const off of offsets) {
          const key = `${pos}:${off}`;
          if (!seen.has(key)) {
            seen.add(key);
            const syn = getSynset(pos, off);
            if (syn) results.push(syn);
          }
        }
        break; // found exact or primary morphological match for this POS
      }
    }
  }
  return results;
}

function getHypernymAncestors(synset: ParsedSynset): Map<string, number> {
  // BFS upward along @ (hypernym) and @i (instance hypernym)
  const distances = new Map<string, number>();
  const queue: Array<{ syn: ParsedSynset; depth: number }> = [{ syn: synset, depth: 0 }];
  distances.set(`${synset.pos}:${synset.offset}`, 0);

  while (queue.length > 0) {
    const current = queue.shift()!;
    if (current.depth >= 18) continue;

    for (const ptr of current.syn.pointers) {
      if (ptr.symbol === '@' || ptr.symbol === '@i') {
        const key = `${ptr.pos}:${ptr.offset}`;
        if (!distances.has(key)) {
          const parent = getSynset(ptr.pos, ptr.offset);
          if (parent) {
            distances.set(key, current.depth + 1);
            queue.push({ syn: parent, depth: current.depth + 1 });
          }
        }
      }
    }
  }
  return distances;
}

function getMinRootDepth(synset: ParsedSynset): number {
  const ancestors = getHypernymAncestors(synset);
  let maxDist = 0;
  for (const d of ancestors.values()) {
    if (d > maxDist) maxDist = d;
  }
  // +1 to represent depth from root (root has depth 1, matching NLTK Wu-Palmer convention)
  return maxDist + 1;
}

export function computePairMetrics(
  synA: ParsedSynset,
  synB: ParsedSynset
): {
  wup: number | null;
  path: number | null;
  lcsKey: string | null;
  relationHint: RelationshipLabel | null;
} {
  if (synA.pos === synB.pos && synA.offset === synB.offset) {
    return { wup: 1.0, path: 1.0, lcsKey: `${synA.pos}:${synA.offset}`, relationHint: 'Synonym' };
  }

  // Direct lexical/semantic links for adjectives/adverbs or cross-links
  const directPtrA = synA.pointers.find((p) => p.pos === synB.pos && p.offset === synB.offset);
  if (directPtrA) {
    if (directPtrA.symbol === '&' || directPtrA.symbol === '^') {
      return { wup: 0.82, path: 0.5, lcsKey: null, relationHint: 'Near Synonym' };
    }
    if (directPtrA.symbol === '@' || directPtrA.symbol === '@i') {
      return { wup: 0.88, path: 0.5, lcsKey: `${synB.pos}:${synB.offset}`, relationHint: 'Hyponym' };
    }
    if (directPtrA.symbol === '~' || directPtrA.symbol === '~i') {
      return { wup: 0.88, path: 0.5, lcsKey: `${synA.pos}:${synA.offset}`, relationHint: 'Hypernym' };
    }
    if (directPtrA.symbol === '!') {
      return { wup: 0.2, path: 0.2, lcsKey: null, relationHint: 'Related Concept' };
    }
  }

  // Adjective 2-hop satellite cluster check (& -> &)
  if (synA.pos === 'a' && synB.pos === 'a') {
    const neighborsA = new Set(
      synA.pointers.filter((p) => p.symbol === '&' || p.symbol === '^').map((p) => `${p.pos}:${p.offset}`)
    );
    const neighborsB = synB.pointers
      .filter((p) => p.symbol === '&' || p.symbol === '^')
      .map((p) => `${p.pos}:${p.offset}`);
    const sharedCluster = neighborsB.find((k) => neighborsA.has(k));
    if (sharedCluster) {
      return { wup: 0.76, path: 0.3333, lcsKey: sharedCluster, relationHint: 'Near Synonym' };
    }
  }

  // Hypernym taxonomy Wu-Palmer & Path similarity (for nouns and verbs of same POS)
  if ((synA.pos === 'n' || synA.pos === 'v') && synA.pos === synB.pos) {
    const ancA = getHypernymAncestors(synA);
    const ancB = getHypernymAncestors(synB);

    let bestPathLen = Infinity;
    let bestWup = 0;
    let bestLcs: string | null = null;

    for (const [key, distA] of ancA.entries()) {
      const distB = ancB.get(key);
      if (distB !== undefined) {
        const pathLen = distA + distB;
        if (pathLen < bestPathLen) {
          bestPathLen = pathLen;
        }
        const [lcsPos, lcsOffStr] = key.split(':');
        const lcsSyn = getSynset(lcsPos as PosCode, parseInt(lcsOffStr, 10));
        if (lcsSyn) {
          const depthLcs = getMinRootDepth(lcsSyn);
          const depthA = depthLcs + distA;
          const depthB = depthLcs + distB;
          const wup = (2 * depthLcs) / (depthA + depthB);
          if (wup > bestWup) {
            bestWup = wup;
            bestLcs = key;
          }
        }
      }
    }

    if (bestLcs && bestPathLen < Infinity) {
      const pathSim = 1 / (bestPathLen + 1);
      let relationHint: RelationshipLabel = 'Related Concept';
      if (ancA.get(bestLcs) === 0 && ancB.get(bestLcs)! > 0) {
        relationHint = 'Hypernym'; // A is hypernym of B
      } else if (ancB.get(bestLcs) === 0 && ancA.get(bestLcs)! > 0) {
        relationHint = 'Hyponym'; // A is hyponym of B
      } else if (bestWup >= 0.75) {
        relationHint = 'Near Synonym';
      } else if (bestWup < 0.35) {
        relationHint = 'No Direct Relationship Detected';
      }
      return {
        wup: Math.min(1, bestWup),
        path: Math.min(1, pathSim),
        lcsKey: bestLcs,
        relationHint,
      };
    }
  }

  return { wup: null, path: null, lcsKey: null, relationHint: null };
}

export function parseGlossAndExamples(rawGloss: string): {
  definition: string;
  examples: string[];
} {
  const quoteMatches = Array.from(rawGloss.matchAll(/"([^"]+)"/g)).map((m) => m[1].trim());
  const firstQuoteIdx = rawGloss.indexOf('"');
  let definition = rawGloss;
  if (firstQuoteIdx !== -1) {
    definition = rawGloss.slice(0, firstQuoteIdx).replace(/;\s*$/, '').trim();
  }
  return {
    definition: definition || rawGloss,
    examples: quoteMatches,
  };
}

function toSummary(syn: ParsedSynset): SynsetSummary {
  const parsed = parseGlossAndExamples(syn.gloss);
  return {
    id: syn.id,
    pos: syn.pos,
    posLabel: POS_FILES[syn.pos].label,
    lemmas: syn.lemmas,
    gloss: parsed.definition,
    examples: parsed.examples,
  };
}

export function getWordDefinitionAndSynonyms(word: string): {
  primaryDefinition: string | null;
  primaryExample: string | null;
  primaryPos: string;
  exactSynonyms: Set<string>;
} {
  const synsets = lookupSynsets(word);
  const exactSynonyms = new Set<string>();
  const normInput = word.trim().toLowerCase().replace(/_/g, ' ');

  for (const syn of synsets) {
    for (const lemma of syn.lemmas) {
      const clean = lemma.toLowerCase();
      if (clean !== normInput) {
        exactSynonyms.add(clean);
      }
    }
  }

  if (synsets.length === 0) {
    return {
      primaryDefinition: null,
      primaryExample: null,
      primaryPos: 'Word',
      exactSynonyms,
    };
  }

  const parsed = parseGlossAndExamples(synsets[0].gloss);
  return {
    primaryDefinition: parsed.definition,
    primaryExample: parsed.examples[0] ?? null,
    primaryPos: POS_FILES[synsets[0].pos].label,
    exactSynonyms,
  };
}

export function analyzeWordNet(wordA: string, wordB: string) {
  const normA = wordA.trim().toLowerCase();
  const normB = wordB.trim().toLowerCase();

  const synsetsA = lookupSynsets(normA);
  const synsetsB = lookupSynsets(normB);

  if (synsetsA.length === 0 || synsetsB.length === 0) {
    const missing = [
      synsetsA.length === 0 ? `"${wordA}"` : null,
      synsetsB.length === 0 ? `"${wordB}"` : null,
    ]
      .filter(Boolean)
      .join(' and ');

    return {
      score: null,
      rawMetric: null,
      status: 'not_available' as const,
      reasonIfUnavailable: `WordNet lexical lookup found no synset entries for ${missing}. WordNet primarily indexes standard English nouns, verbs, adjectives, and adverbs.`,
      synsetsA: synsetsA.slice(0, 6).map(toSummary),
      synsetsB: synsetsB.slice(0, 6).map(toSummary),
      bestPair: null,
      relationship: 'No Direct Relationship Detected' as RelationshipLabel,
    };
  }

  if (normA === normB) {
    return {
      score: 100,
      rawMetric: 1.0,
      status: 'completed' as const,
      synsetsA: synsetsA.slice(0, 6).map(toSummary),
      synsetsB: synsetsB.slice(0, 6).map(toSummary),
      bestPair: {
        synsetA: toSummary(synsetsA[0]),
        synsetB: toSummary(synsetsB[0]),
        wupSimilarity: 1.0,
        pathSimilarity: 1.0,
        sharedHypernym: synsetsA[0].lemmas[0],
      },
      relationship: 'Exact Match' as RelationshipLabel,
    };
  }

  let bestWup = -1;
  let bestPath = 0;
  let bestSynA: ParsedSynset | null = null;
  let bestSynB: ParsedSynset | null = null;
  let bestLcsKey: string | null = null;
  let bestHint: RelationshipLabel = 'No Direct Relationship Detected';

  for (const sA of synsetsA) {
    for (const sB of synsetsB) {
      const metrics = computePairMetrics(sA, sB);
      if (metrics.wup !== null && metrics.wup > bestWup) {
        bestWup = metrics.wup;
        bestPath = metrics.path ?? 0;
        bestSynA = sA;
        bestSynB = sB;
        bestLcsKey = metrics.lcsKey;
        bestHint = metrics.relationHint || 'Related Concept';
      }
    }
  }

  if (bestWup < 0 || !bestSynA || !bestSynB) {
    return {
      score: null,
      rawMetric: null,
      status: 'not_available' as const,
      reasonIfUnavailable: `Both words exist in WordNet, but their available parts of speech or lexical graphs do not share a comparable hypernym path or direct relation.`,
      synsetsA: synsetsA.slice(0, 6).map(toSummary),
      synsetsB: synsetsB.slice(0, 6).map(toSummary),
      bestPair: null,
      relationship: 'No Direct Relationship Detected' as RelationshipLabel,
    };
  }

  let sharedHypernymName: string | null = null;
  if (bestLcsKey) {
    const [p, o] = bestLcsKey.split(':');
    const lcsSyn = getSynset(p as PosCode, parseInt(o, 10));
    if (lcsSyn) {
      sharedHypernymName = lcsSyn.lemmas.join(', ');
    }
  }

  return {
    score: Math.round(bestWup * 1000) / 10,
    rawMetric: Math.round(bestWup * 10000) / 10000,
    status: 'completed' as const,
    synsetsA: synsetsA.slice(0, 6).map(toSummary),
    synsetsB: synsetsB.slice(0, 6).map(toSummary),
    bestPair: {
      synsetA: toSummary(bestSynA),
      synsetB: toSummary(bestSynB),
      wupSimilarity: Math.round(bestWup * 10000) / 10000,
      pathSimilarity: Math.round(bestPath * 10000) / 10000,
      sharedHypernym: sharedHypernymName,
    },
    relationship: bestHint,
  };
}

export function exploreWordNet(word: string): SemanticExploreResponse {
  const cleanWord = word.trim();
  const synsets = lookupSynsets(cleanWord);

  if (synsets.length === 0) {
    return {
      word: cleanWord,
      found: false,
      synsets: [],
      nodes: [],
      edges: [],
      message: `No WordNet synsets were found for "${cleanWord}". Try a standard English noun, verb, adjective, or adverb (e.g., vehicle, happy, algorithm, ocean).`,
    };
  }

  const centerId = `center:${cleanWord.toLowerCase()}`;
  const nodesMap = new Map<string, SemanticGraphNode>();
  const edges: SemanticGraphEdge[] = [];
  const edgeKeys = new Set<string>();

  const primarySyn = synsets[0];
  nodesMap.set(centerId, {
    id: centerId,
    label: cleanWord,
    relation: 'center',
    pos: POS_FILES[primarySyn.pos].label,
    gloss: primarySyn.gloss,
  });

  const addNodeAndEdge = (
    lemma: string,
    relation: 'synonym' | 'hypernym' | 'hyponym' | 'related',
    posLabel: string,
    gloss: string,
    edgeLabel: string
  ) => {
    const norm = lemma.toLowerCase();
    if (norm === cleanWord.toLowerCase()) return;
    const nodeId = `node:${norm}`;
    if (!nodesMap.has(nodeId) && nodesMap.size < 22) {
      nodesMap.set(nodeId, {
        id: nodeId,
        label: lemma,
        relation,
        pos: posLabel,
        gloss,
      });
    }
    if (nodesMap.has(nodeId)) {
      const eKey = `${centerId}->${nodeId}:${relation}`;
      if (!edgeKeys.has(eKey)) {
        edgeKeys.add(eKey);
        edges.push({
          source: centerId,
          target: nodeId,
          relation,
          label: edgeLabel,
        });
      }
    }
  };

  // Collect across top 3 synsets for rich authentic relationships
  for (const syn of synsets.slice(0, 3)) {
    const posLabel = POS_FILES[syn.pos].label;

    // Synonyms from the synset lemmas
    for (const lemma of syn.lemmas) {
      addNodeAndEdge(lemma, 'synonym', posLabel, syn.gloss, 'Synonym');
    }

    // Hypernyms (@, @i)
    for (const ptr of syn.pointers.filter((p) => p.symbol === '@' || p.symbol === '@i').slice(0, 3)) {
      const target = getSynset(ptr.pos, ptr.offset);
      if (target && target.lemmas[0]) {
        addNodeAndEdge(target.lemmas[0], 'hypernym', POS_FILES[target.pos].label, target.gloss, 'Hypernym (broader)');
      }
    }

    // Hyponyms (~, ~i)
    for (const ptr of syn.pointers.filter((p) => p.symbol === '~' || p.symbol === '~i').slice(0, 6)) {
      const target = getSynset(ptr.pos, ptr.offset);
      if (target && target.lemmas[0]) {
        addNodeAndEdge(target.lemmas[0], 'hyponym', POS_FILES[target.pos].label, target.gloss, 'Hyponym (narrower)');
      }
    }

    // Related concepts (& similar_to, ^ also_see, + derivationally related)
    for (const ptr of syn.pointers
      .filter((p) => p.symbol === '&' || p.symbol === '^' || p.symbol === '+')
      .slice(0, 4)) {
      const target = getSynset(ptr.pos, ptr.offset);
      if (target && target.lemmas[0]) {
        addNodeAndEdge(target.lemmas[0], 'related', POS_FILES[target.pos].label, target.gloss, 'Related Concept');
      }
    }
  }

  return {
    word: cleanWord,
    found: true,
    synsets: synsets.slice(0, 6).map(toSummary),
    nodes: Array.from(nodesMap.values()),
    edges,
  };
}
