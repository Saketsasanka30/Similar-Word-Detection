import { cos_sim } from '@huggingface/transformers';
import { encodeBatchTransformer, TRANSFORMER_MODEL_ID } from './embeddings.ts';
import { preprocessTokens } from './tfidf.ts';
import type {
  ContextAnalysisResponse,
  ContextSenseScore,
  ContextSentenceResult,
  NlpPipelineStage,
} from './types.ts';
import { lookupSynsets, parseGlossAndExamples } from './wordnet.ts';

const POS_LABELS: Record<string, string> = {
  n: 'Noun',
  v: 'Verb',
  a: 'Adjective',
  r: 'Adverb',
};

function scoreSentenceAgainstSenses(
  sentence: string,
  sentVec: number[],
  targetWord: string,
  senseItems: Array<{
    synsetId: string;
    posLabel: string;
    lemmas: string[];
    definition: string;
    exampleUsage: string | null;
    senseTokens: Set<string>;
    vec: number[];
  }>
): ContextSentenceResult {
  const targetLower = targetWord.trim().toLowerCase();
  const contextTokens = preprocessTokens(sentence, true).filter((t) => t !== targetLower);
  const contextSet = new Set(contextTokens);

  const scored: ContextSenseScore[] = senseItems.map((sense) => {
    const rawCos = Math.max(0, Math.min(1, cos_sim(sentVec, sense.vec)));
    let leskOverlapCount = 0;
    for (const tok of contextSet) {
      if (sense.senseTokens.has(tok)) {
        leskOverlapCount += 1;
      }
    }
    const leskBonus = Math.min(0.12, leskOverlapCount * 0.04);
    const combined = Math.min(1, rawCos * 0.88 + leskBonus);

    return {
      synsetId: sense.synsetId,
      posLabel: sense.posLabel,
      lemmas: sense.lemmas,
      gloss: sense.definition,
      exampleUsage: sense.exampleUsage,
      contextualSimilarity: Math.round(rawCos * 1000) / 1000,
      leskOverlapCount,
      combinedScore: Math.round(combined * 1000) / 1000,
    };
  });

  scored.sort((a, b) => b.combinedScore - a.combinedScore);

  return {
    sentence,
    detectedSense: scored[0],
    alternativeSenses: scored.slice(1, 5),
    contextTokens,
  };
}

export async function analyzeWordInContext(params: {
  word: string;
  sentence_a: string;
  sentence_b?: string;
}): Promise<ContextAnalysisResponse> {
  const cleanWord = params.word.trim();
  const sentA = params.sentence_a.trim();
  const sentB = params.sentence_b?.trim() || '';

  const synsets = lookupSynsets(cleanWord).slice(0, 12);
  if (synsets.length === 0) {
    return {
      word: cleanWord,
      found: false,
      totalSenses: 0,
      sentence1: null,
      sentence2: null,
      sameSenseDetected: null,
      senseComparisonSummary: `No WordNet senses were found for "${cleanWord}".`,
      technicalExplanation: `Princeton WordNet 3.1 returned zero synsets for "${cleanWord}", so sense disambiguation cannot map it to dictionary senses.`,
      beginnerExplanation: `We could not find "${cleanWord}" in the dictionary database. Try a standard English word like "Bank", "Apple", "Python", or "Crane".`,
      pipelineStages: [],
      message: `No WordNet senses found for "${cleanWord}".`,
      timestamp: new Date().toISOString(),
    };
  }

  const parsedSenses = synsets.map((syn) => {
    const { definition, examples } = parseGlossAndExamples(syn.gloss);
    const posLabel = POS_LABELS[syn.pos] ?? 'Word';
    const profileText = `${cleanWord} (${posLabel}): ${definition}${
      examples.length > 0 ? `. Example: ${examples[0]}` : ''
    }`;
    const senseTokens = new Set([
      ...preprocessTokens(definition, true),
      ...syn.lemmas.flatMap((l) => preprocessTokens(l, true)),
    ]);
    return {
      synsetId: syn.id,
      posLabel,
      lemmas: syn.lemmas,
      definition,
      exampleUsage: examples[0] ?? null,
      profileText,
      senseTokens,
    };
  });

  const textsToEncode = [sentA];
  if (sentB) textsToEncode.push(sentB);
  const senseOffset = textsToEncode.length;
  for (const s of parsedSenses) textsToEncode.push(s.profileText);

  const embeddings = await encodeBatchTransformer(textsToEncode);
  const vecA = embeddings[0];
  const vecB = sentB ? embeddings[1] : null;
  const senseItems = parsedSenses.map((s, idx) => ({
    ...s,
    vec: embeddings[senseOffset + idx],
  }));

  const result1 = scoreSentenceAgainstSenses(sentA, vecA, cleanWord, senseItems);
  const result2 = vecB ? scoreSentenceAgainstSenses(sentB, vecB, cleanWord, senseItems) : null;

  let sameSenseDetected: boolean | null = null;
  let senseComparisonSummary = `Detected primary sense "${result1.detectedSense.gloss}" (${result1.detectedSense.posLabel}) for "${cleanWord}" in the given context.`;

  if (result2) {
    sameSenseDetected =
      result1.detectedSense.synsetId === result2.detectedSense.synsetId;
    if (sameSenseDetected) {
      senseComparisonSummary = `Both sentences invoke the same meaning of "${cleanWord}": ${result1.detectedSense.gloss}.`;
    } else {
      senseComparisonSummary = `Polysemy detected! Sentence 1 uses "${cleanWord}" as "${result1.detectedSense.gloss}", whereas Sentence 2 uses "${cleanWord}" as "${result2.detectedSense.gloss}".`;
    }
  }

  const pipelineStages: NlpPipelineStage[] = [
    {
      step: 1,
      name: 'Text Input',
      summary: `Target "${cleanWord}" across ${result2 ? '2 sentences' : '1 sentence'}`,
      details: [
        `Sentence 1: "${sentA}"`,
        ...(sentB ? [`Sentence 2: "${sentB}"`] : []),
      ],
    },
    {
      step: 2,
      name: 'Text Preprocessing',
      summary: `Extracted surrounding context tokens`,
      details: [
        `Context 1 tokens: ${result1.contextTokens.join(', ') || '(none)'}`,
        ...(result2
          ? [`Context 2 tokens: ${result2.contextTokens.join(', ') || '(none)'}`]
          : []),
      ],
    },
    {
      step: 3,
      name: 'Tokenization',
      summary: `Retrieved ${synsets.length} WordNet senses for "${cleanWord}"`,
      details: parsedSenses
        .slice(0, 3)
        .map((s) => `${s.synsetId} (${s.posLabel}): ${s.definition}`),
    },
    {
      step: 4,
      name: 'Vectorization',
      summary: `Encoded sentences & glosses with ${TRANSFORMER_MODEL_ID}`,
      details: [
        '384-dim contextual sentence vectors',
        '384-dim WordNet sense-definition vectors',
      ],
    },
    {
      step: 5,
      name: 'Similarity Calculation',
      summary: 'Contextual cosine similarity + Lesk gloss token overlap',
      details: [
        `Sentence 1 top sense score: ${result1.detectedSense.combinedScore.toFixed(3)}`,
        ...(result2
          ? [`Sentence 2 top sense score: ${result2.detectedSense.combinedScore.toFixed(3)}`]
          : []),
      ],
    },
    {
      step: 6,
      name: 'Result Interpretation',
      summary: sameSenseDetected === false ? 'Different senses identified' : 'Word sense resolved',
      details: [senseComparisonSummary],
    },
  ];

  return {
    word: cleanWord,
    found: true,
    totalSenses: synsets.length,
    sentence1: result1,
    sentence2: result2,
    sameSenseDetected,
    senseComparisonSummary,
    technicalExplanation: `Evaluated ${synsets.length} WordNet synsets for "${cleanWord}" by embedding each sense gloss with ${TRANSFORMER_MODEL_ID} and scoring contextual cosine alignment plus Lesk content-word overlap against surrounding sentence tokens.`,
    beginnerExplanation: `Words like "${cleanWord}" can mean different things depending on the words around them. By looking at the rest of the sentence, SEMANTIQ matches "${cleanWord}" to the dictionary meaning that fits best.`,
    pipelineStages,
    timestamp: new Date().toISOString(),
  };
}
