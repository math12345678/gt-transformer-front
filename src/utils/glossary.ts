export interface GlossaryEntry {
  key: string;
  term: string;
  category: string;
  summary: string;
  formula?: string;
}

export const PIPELINE_GLOSSARY: Record<string, GlossaryEntry> = {
  'parsimony-informative': {
    key: 'parsimony-informative',
    term: 'Parsimony-Informative Sites',
    category: 'Stage 1 · Alignment Filtering',
    summary:
      'Alignment columns where at least two distinct nucleotides each appear in at least two sequences. Invariant columns and singleton mutations carry no phylogenetic signal for recombination and are filtered while preserving 1-based genomic coordinates.',
    formula: 'count(base_a) ≥ 2 ∧ count(base_b) ≥ 2',
  },
  's-invariant': {
    key: 's-invariant',
    term: 'S-Invariant Representation',
    category: 'Stage 2 · Tensorization',
    summary:
      'An encoding whose tensor shape and neural runtime are independent of alignment depth S (whether 5 or 500 sequences). Summarizing pairwise discordance and 4-gamete incompatibilities across fixed circular windows keeps rare 1/64 singletons as sharp as large clade switches.',
    formula: 'dim(X) = [W, C] independent of S ∈ [5, 500]',
  },
  'circular-window': {
    key: 'circular-window',
    term: 'Circular Window Representation',
    category: 'Stage 2 · Window Geometry',
    summary:
      'A fixed-width sliding window (W = 314 informative sites) wrapped smoothly at alignment boundaries so edge-adjacent features do not suffer zero-padding artifacts during self-attention.',
    formula: 'W = 314 informative sites',
  },
  'window-transformer': {
    key: 'window-transformer',
    term: '4-Layer Window Transformer',
    category: 'Stage 3 · Neural Branch',
    summary:
      'A 570,881-parameter pre-norm Transformer trained on 40,000 SantaSim viral alignments. It scores local multi-site linkage breaks to emit calibrated per-site boundary logits (p_TF) with sub-window spatial precision.',
    formula: 'ckpt_N40000 · 570,881 params · 4 layers',
  },
  'flanking-kl': {
    key: 'flanking-kl',
    term: 'Flanking-KL Divergence',
    category: 'Stage 4 · Statistical Branch',
    summary:
      'Measures the Kullback–Leibler divergence between soft parental affinity distributions immediately left [b − w, b) and immediately right [b, b + w) of candidate site b across multiple window scales. True recombination produces a sharp symmetric peak where parental ancestry switches.',
    formula: 'D_KL( P_left[b−w, b) ∥ P_right[b, b+w) )',
  },
  'column-shuffled': {
    key: 'column-shuffled',
    term: '48 Column-Shuffled Permutations',
    category: 'Stage 5 · Empirical Null Calibration',
    summary:
      'Shuffles informative columns across genomic coordinates 48 times per alignment. Because column shuffling preserves per-site diversity and mutation rate while destroying spatial linkage, it yields an exact null distribution that suppresses false positives (0 / 30 FP at z ≥ 6.0).',
    formula: 'z(b) = (KL(b) − μ_null) / σ_null ≥ 6.0',
  },
  'co-descendants': {
    key: 'co-descendants',
    term: 'Co-Descendant Exclusion',
    category: 'Stage 6 · Parental Attribution',
    summary:
      'When an ancestral recombination event is inherited by multiple descendant genomes (e.g., Rec11..Rec13), those siblings share the exact same mosaic structure. Stage 6 clusters the carrier clade and excludes all co-descendants from the donor pool before calling external parents.',
    formula: 'donors(c) = S \\ carrier_clade(c)',
  },
  'level-1-consolidation': {
    key: 'level-1-consolidation',
    term: 'Level-1 Peak Consolidation (≤75 bp)',
    category: 'Stage 6 · Deduplication',
    summary:
      'Groups proximal candidate peaks within 75 bp via connected components into single biological recombination events, cutting redundant calls by >20× compared to raw 7-method RDP5 triplet scans.',
    formula: '|b_i − b_j| ≤ 75 bp → merge component',
  },
  'native-rdp5': {
    key: 'native-rdp5',
    term: 'Native RDP5 (.rdp) Export',
    category: 'Stage 6 · Interoperability',
    summary:
      'Writes consolidated breakpoint coordinates, 95% confidence intervals, carrier clades, and external parent attributions directly into binary-compatible RDP5 project files alongside GFF3, BED, FASTA tracts, JSON, and TSV.',
    formula: '.rdp · GFF3 · BED · FASTA · JSON · TSV',
  },
};

export function getGlossaryEntry(key: string): GlossaryEntry | undefined {
  return PIPELINE_GLOSSARY[key];
}
