# Refined Approach: Scientific Classification First

## Overview

This document describes the **refined, validated approach** for semantic classification and hypothesis generation, emphasizing:

1. **PubMed MCP Server integration** for real data access
2. **Scientific domain-specific classification** with rigorous justification
3. **Step-by-step validation** before full pipeline
4. **Classification quality assurance** using scientific criteria

---

## Key Improvements Over Initial Implementation

### Before (Initial POC)
❌ Implemented full pipeline without testing components
❌ Generic semantic clustering without scientific validation
❌ No justification for cluster assignments
❌ Assumed dataset availability without MCP server option
❌ No quality checks on classifications

### After (Refined Approach)
✅ Step-by-step component testing
✅ **Scientific classification with explicit justification**
✅ **Domain-specific validation criteria** (methodology, framework, temporal)
✅ **MCP server integration** for real PubMed access
✅ **Classification quality scores** and recommended actions

---

## Architecture: Classification-First Approach

```
┌─────────────────────────────────────────────────────────┐
│  DATA SOURCE                                            │
│  Option A: PubMed MCP Server (real-time queries)       │
│  Option B: aiscientist dataset (20k BCH papers)        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  SEMANTIC EMBEDDING                                     │
│  - Transform papers to vector space                     │
│  - Capture semantic meaning                             │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  INITIAL CLUSTERING                                     │
│  - HDBSCAN/DBSCAN for initial groupings                │
│  - Identify potential semantic clusters                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  🎯 SCIENTIFIC CLASSIFICATION (NEW - KEY COMPONENT)     │
│                                                         │
│  For each cluster:                                      │
│  1. Identify methodology (RCT, cohort, etc.)           │
│  2. Determine domain (clinical, basic, etc.)           │
│  3. Extract framework (mechanistic, interventional)     │
│  4. Validate temporal coherence                         │
│  5. Check MeSH term consistency                         │
│                                                         │
│  Output: JUSTIFIED classifications                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  CLASSIFICATION VALIDATION                              │
│                                                         │
│  Scores (0-1):                                          │
│  - Methodological coherence (0.35 weight)              │
│  - Conceptual framework (0.25 weight)                  │
│  - Temporal coherence (0.15 weight)                    │
│  - Internal consistency (0.15 weight)                  │
│  - MeSH coherence (0.10 weight)                        │
│                                                         │
│  Action: ACCEPT / REVIEW / SPLIT / RECLASSIFY          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  REFINED CLUSTERS                                       │
│  - Scientifically valid groupings                       │
│  - Explicit justifications                              │
│  - Quality-checked                                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  GAP ANALYSIS & HYPOTHESIS GENERATION                   │
│  (Only proceed with validated clusters)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Scientific Classification Criteria

### Dimension 1: Research Methodology (35% weight)

**Clinical Trials**
- Keywords: randomized, controlled trial, phase I/II/III, RCT
- Evidence: intervention, control group, blinding
- Score: Percentage of papers showing these features

**Cohort Studies**
- Keywords: cohort, prospective, retrospective, follow-up
- Evidence: exposure tracking, temporal sequence
- Score: Dominance of observational longitudinal design

**Laboratory Research**
- Keywords: in vitro, in vivo, cell culture, animal model
- Evidence: experimental manipulation, controlled conditions
- Score: Laboratory-based methods prevalence

**Computational**
- Keywords: bioinformatics, machine learning, modeling
- Evidence: algorithmic methods, large-scale data
- Score: Computational approach dominance

**Systematic Reviews**
- Keywords: systematic review, meta-analysis, PRISMA
- Evidence: literature synthesis, quality assessment
- Score: Review methodology prevalence

### Dimension 2: Conceptual Framework (25% weight)

**Mechanistic** (How does it work?)
- Keywords: mechanism, pathway, molecular, signaling
- Focus: Understanding biological processes

**Etiological** (What causes it?)
- Keywords: risk factor, etiology, cause, pathogenesis
- Focus: Disease origins and causation

**Descriptive** (What is happening?)
- Keywords: prevalence, incidence, epidemiology
- Focus: Characterization and patterns

**Predictive** (What will happen?)
- Keywords: prognosis, prediction, risk score
- Focus: Future outcomes

**Interventional** (How to change it?)
- Keywords: treatment, therapy, intervention, efficacy
- Focus: Changing outcomes

### Dimension 3: Temporal Coherence (15% weight)

**Scoring**:
- ≤10 year span: 0.8-1.0 (excellent)
- 10-15 years: 0.6-0.8 (good)
- >15 years: <0.6 (concerning - methods may have evolved)

### Dimension 4: Internal Consistency (15% weight)

- Common vocabulary across papers
- Similar research questions
- Compatible outcome measures

### Dimension 5: MeSH Terms (10% weight)

- Top MeSH terms appear in >50% of papers
- Cluster has identifiable MeSH signature
- Medical domain is clear

---

## Validation Thresholds

### Overall Score ≥0.8: EXCELLENT ✅
- Strong scientific coherence
- Clear cluster identity
- **Action**: ACCEPT as-is

### Overall Score 0.6-0.8: GOOD ✅
- Acceptable coherence
- Minor heterogeneity
- **Action**: ACCEPT with manual review of edge cases

### Overall Score 0.4-0.6: REVIEW ⚠️
- Moderate heterogeneity
- Mixed signals
- **Action**: Manual inspection required, consider refinement

### Overall Score <0.4: RECLASSIFY 🚩
- Poor coherence
- Multiple issues
- **Action**: SPLIT by methodology/domain or MERGE with another cluster

---

## Step-by-Step Workflow

### Phase 1: Setup (One-time)

```bash
# 1. Clone project
cd /Users/e.baena/CascadeProjects/research-semantic-poc

# 2. Install dependencies
./setup.sh

# 3. Setup PubMed MCP Server (optional but recommended)
# Follow: config/mcp_pubmed_setup.md

# 4. Get dataset (if not using MCP server)
cd data
git clone https://github.com/sergeicu/aiscientist
cd ..
```

### Phase 2: Component Testing

```bash
# Test each component independently
python scripts/test_step_by_step.py

# This will:
# 1. Test data loading ✓
# 2. Test embeddings ✓
# 3. Test clustering ✓
# 4. Test validation ✓
```

**Expected Output**:
- All components pass tests
- Validation scores ≥0.6 for most clusters
- Clear cluster identities

**If tests fail**:
- Check data quality
- Adjust clustering parameters (min_cluster_size)
- Review error messages

### Phase 3: Full Pipeline (After Testing)

```bash
# Run with validated components
python scripts/run_full_pipeline.py

# With custom parameters
python scripts/run_full_pipeline.py \
  --data-file data/aiscientist/pubmed_data.csv \
  --min-cluster-size 10 \
  --cluster-method hdbscan
```

### Phase 4: Interactive Analysis with Agents

```bash
# Start Claude Code
claude

# Use refined semantic classifier
@scientific-semantic-classifier
Analyze the validated clusters and explain the scientific rationale for cluster_2

@scientific-semantic-classifier
Which clusters have strong methodological coherence but weak temporal coherence?

@scientific-semantic-classifier
Justify why papers in cluster_5 belong together from a scientific perspective
```

### Phase 5: Hypothesis Generation

Only after classification is validated:

```
@hypothesis-generator-enhanced
Generate hypotheses from validated cluster_3 focusing on methodological gaps

@hypothesis-generator-enhanced
What cross-cluster opportunities exist between cluster_2 and cluster_7?
```

---

## Using PubMed MCP Server

### Query PubMed Directly

```
# In Claude Code with MCP server configured:

Search PubMed for papers on "pediatric immunotherapy" from 2023

# Returns: PMIDs, titles, abstracts, metadata

Fetch full details for PMID 12345678

# Returns: Complete paper metadata, MeSH terms, citations

Find papers citing PMID 12345678

# Returns: Citation network
```

### Workflow with MCP Server

```
1. Query PubMed for specific topic
2. Download paper metadata (title, abstract, MeSH)
3. Save as CSV
4. Run classification pipeline
5. Validate clusters
6. Generate hypotheses
```

---

## Quality Assurance Checklist

Before accepting a cluster classification:

### ✅ Methodological Coherence
- [ ] Dominant methodology identified
- [ ] ≥70% of papers use same methodology
- [ ] Methodology makes scientific sense

### ✅ Domain Specificity
- [ ] Clear scientific domain (not just "medicine")
- [ ] Domain-appropriate terminology
- [ ] Papers address related questions

### ✅ Temporal Validity
- [ ] Papers within 10-15 years (ideally)
- [ ] If >15 years, methods haven't drastically changed
- [ ] No ancient + modern papers together

### ✅ Scientific Justification
- [ ] Can explain WHY papers belong together
- [ ] Evidence from paper content (not just similarity)
- [ ] Would pass peer review

### ✅ Actionable Identity
- [ ] Cluster has specific name (not generic)
- [ ] Can identify research gaps in this cluster
- [ ] Can generate hypotheses specific to this cluster

---

## Common Issues and Solutions

### Issue: Low Methodological Coherence (<0.5)

**Problem**: Papers use different methods (RCT + case reports mixed)

**Solution**: 
```python
# Split cluster by methodology
# Adjust clustering parameters to be more strict
--min-cluster-size 15  # Larger minimum
```

### Issue: Wide Temporal Span (>15 years)

**Problem**: Methods from 2008 mixed with 2023 (e.g., pre/post ML era)

**Solution**:
```python
# Filter by date before clustering
df_recent = df[df['year'] >= 2015]

# Or split into temporal cohorts
```

### Issue: Generic Domain Classification

**Problem**: Cluster identified as "Medicine" instead of specific field

**Solution**:
- Examine MeSH terms for specificity
- Look at journal names
- Review paper titles for common specific terms
- Manually refine cluster name

### Issue: Mixed Conceptual Frameworks

**Problem**: Mechanistic + interventional mixed

**Solution**:
- This can be acceptable (e.g., mechanism + treatment testing)
- Check if papers genuinely related
- If not, consider splitting

---

## Best Practices

### 1. Always Test First
Never run full pipeline without testing components

### 2. Validate Classifications
Check validation scores before proceeding to gap analysis

### 3. Document Justifications
Keep track of WHY clusters make sense scientifically

### 4. Iterate
Classification is iterative - refine based on validation results

### 5. Use Domain Knowledge
Don't rely solely on algorithms - apply scientific expertise

### 6. Start Small
Test with 100-500 papers before scaling to 20k

### 7. Compare with Field Standards
How would experts in the field classify these papers?

---

## Success Metrics

### Classification Quality
- **Target**: ≥70% clusters score ≥0.7
- **Excellent**: ≥80% clusters score ≥0.8
- **Needs work**: <60% clusters score ≥0.6

### Scientific Validity
- Can explain each cluster to domain expert
- Cluster names are specific and accurate
- Justifications cite actual paper content

### Actionability
- Can identify specific gaps per cluster
- Can generate testable hypotheses
- Hypotheses are grounded in validated clusters

---

## Next Steps After Validation

1. **Review validation reports** in `output/test_validation.json`
2. **Manually inspect** borderline clusters (score 0.6-0.7)
3. **Refine parameters** if needed and re-run
4. **Proceed to gap analysis** only with validated clusters
5. **Use agents** for interactive exploration
6. **Generate hypotheses** from solid foundation

---

## Key Takeaway

**Classification quality determines hypothesis quality.**

Bad clusters → Generic gaps → Weak hypotheses
Good clusters → Specific gaps → Strong hypotheses

Therefore: **Validate classifications rigorously before generating hypotheses.**

---

## Files Reference

### Core Components
- `src/embeddings/paper_embedder.py` - Semantic embedding
- `src/clustering/semantic_clusterer.py` - Initial clustering
- `src/extraction/classification_validator.py` - **Scientific validation (NEW)**
- `src/extraction/gap_analyzer.py` - Gap identification
- `src/extraction/hypothesis_generator.py` - Hypothesis creation

### Agent Definitions
- `.claude/agents/scientific-semantic-classifier.md` - **Refined classifier agent**
- `.claude/agents/hypothesis-generator-enhanced.md` - Hypothesis agent

### Scripts
- `scripts/test_step_by_step.py` - **Component testing (USE THIS FIRST)**
- `scripts/run_full_pipeline.py` - Full pipeline (use after testing)

### Configuration
- `config/pipeline_config.yaml` - Pipeline parameters
- `config/mcp_pubmed_setup.md` - MCP server setup

### Documentation
- `QUICKSTART.md` - Quick start guide
- `ARCHITECTURE.md` - Technical architecture
- `REFINED_APPROACH.md` - **This document**

---

## Summary

The refined approach emphasizes:

1. **Scientific rigor** in classification
2. **Explicit justification** for cluster assignments
3. **Validation before proceeding**
4. **Domain-specific criteria**
5. **Quality assurance at every step**

This ensures that hypotheses generated are grounded in scientifically valid cluster analysis, not just algorithmic convenience.
