# Validation Criteria - Current Configuration

**Comprehensive list of all validation criteria used in the pipeline**

---

## 📊 Standard Validation Criteria (Always Applied)

### 1. Methodological Coherence (Weight: 35%)

**Purpose**: Assess if papers in a cluster use similar research methodologies

**Methodologies Detected**:

#### Clinical Trial
- Keywords: `randomized`, `controlled trial`, `phase i`, `phase ii`, `phase iii`, `rct`, `double-blind`, `placebo`
- Description: Clinical interventional studies (RCTs, trials)

#### Cohort Study
- Keywords: `cohort`, `prospective`, `retrospective`, `follow-up`, `longitudinal`, `observational`
- Description: Observational cohort studies with follow-up

#### Case-Control
- Keywords: `case-control`, `matched controls`, `odds ratio`, `case control`
- Description: Case-control studies examining risk factors

#### Cross-Sectional
- Keywords: `cross-sectional`, `survey`, `prevalence`, `questionnaire`
- Description: Cross-sectional surveys or prevalence studies

#### Systematic Review
- Keywords: `systematic review`, `meta-analysis`, `prisma`, `cochrane`
- Description: Systematic reviews and meta-analyses

#### Laboratory
- Keywords: `in vitro`, `in vivo`, `cell culture`, `animal model`, `mouse model`, `biochemical assay`
- Description: Laboratory-based experimental research

#### Computational
- Keywords: `bioinformatics`, `machine learning`, `deep learning`, `computational`, `algorithm`, `modeling`, `simulation`
- Description: Computational/bioinformatics analysis

#### Genomic
- Keywords: `genome-wide`, `gwas`, `sequencing`, `rna-seq`, `genomic`, `transcriptomic`, `whole exome`
- Description: Genomic or transcriptomic studies

**Scoring**:
- Dominance ≥70%: Score 0.8-1.0 (Strong coherence)
- Dominance ≥50%: Score 0.6-0.8 (Good coherence)
- Dominance <50%: Score <0.6 (Mixed methodologies)

---

### 2. Framework Coherence (Weight: 25%)

**Purpose**: Assess if papers share a conceptual framework

**Frameworks Detected**:

#### Mechanistic
- Keywords: `mechanism`, `pathway`, `molecular`, `signaling`
- Focus: Understanding biological mechanisms

#### Etiological
- Keywords: `etiology`, `risk factor`, `cause`, `pathogenesis`
- Focus: Identifying causes and risk factors

#### Descriptive
- Keywords: `prevalence`, `incidence`, `epidemiology`, `distribution`
- Focus: Describing disease patterns

#### Predictive
- Keywords: `prognosis`, `prediction`, `risk score`, `prognostic`
- Focus: Predicting outcomes

#### Interventional
- Keywords: `treatment`, `therapy`, `intervention`, `efficacy`
- Focus: Evaluating interventions

**Scoring**:
- Dominance ≥50%: Score 0.75-1.0 (Clear framework)
- Dominance <50%: Score <0.75 (Mixed frameworks)

---

### 3. Temporal Coherence (Weight: 15%)

**Purpose**: Assess if papers are from similar time periods

**Scoring**:
- Papers within 10 years: Score 0.8-1.0 (Excellent)
- Papers within 15 years: Score 0.6-0.8 (Good)
- Papers >15 years apart: Score <0.6 (Wide span)

**Note**: Methods and standards evolve over time, so temporal coherence matters

---

### 4. Internal Consistency (Weight: 15%)

**Purpose**: Assess title-abstract similarity within cluster

**Method**: 
- Calculates pairwise cosine similarity between paper embeddings
- Measures how similar papers are to each other

**Scoring**:
- Mean similarity >0.7: Score 0.8-1.0 (High consistency)
- Mean similarity 0.5-0.7: Score 0.6-0.8 (Moderate)
- Mean similarity <0.5: Score <0.6 (Low consistency)

---

### 5. MeSH Coherence (Weight: 10%)

**Purpose**: Assess if papers share Medical Subject Headings (MeSH terms)

**Method**:
- Analyzes overlap in MeSH terms across papers
- MeSH terms are standardized medical vocabulary

**Scoring**:
- High overlap: Score 0.8-1.0
- Moderate overlap: Score 0.5-0.8
- Low overlap: Score <0.5

---

## 🔍 Custom Validation Criteria (Optional)

### 6. Data Availability Criterion (Weight: 15%)

**Purpose**: Assess if papers mention data availability

**Keywords Detected**:
- `data available`, `publicly available`, `supplementary data`
- `github`, `figshare`, `zenodo`, `dryad`
- `data repository`, `open data`, `data sharing`
- `accession number`, `database`

**Scoring**:
- High data availability: Score 0.8-1.0
- Some data availability: Score 0.5-0.8
- No data availability: Score <0.5

**Importance**: Papers with available data are more reproducible

---

### 7. Clinical Trial Sponsor Criterion (Weight: 10%)

**Purpose**: Identify if clinical trials have industry sponsors

**Keywords Detected**:
- `funded by`, `sponsored by`, `supported by`
- `pharmaceutical`, `biotech`, `industry`
- `grant`, `funding`

**Scoring**:
- Clear sponsor information: Higher score
- No sponsor information: Lower score

**Importance**: Sponsor information affects study interpretation

---

### 8. Replication Status Criterion (Weight: 10%)

**Purpose**: Assess if findings have been replicated

**Keywords Detected**:
- `replicated`, `replication`, `validated`
- `confirmed`, `reproduced`, `independent study`
- `multi-center`, `multicenter`

**Scoring**:
- Evidence of replication: Score 0.8-1.0
- Some replication: Score 0.5-0.8
- No replication mentioned: Score <0.5

**Importance**: Replicated findings are more reliable

---

## 📈 Overall Scoring

### Calculation

**Standard Criteria** (Total: 100%):
```
Overall Score = 
  (Methodological × 0.35) +
  (Framework × 0.25) +
  (Temporal × 0.15) +
  (Consistency × 0.15) +
  (MeSH × 0.10)
```

**With Custom Criteria** (if enabled):
```
Base Score = Standard Overall Score × 0.65
Custom Score = 
  (Data Availability × 0.15) +
  (Sponsor × 0.10) +
  (Replication × 0.10)

Final Score = Base Score + Custom Score
```

### Thresholds

- **≥0.8**: ✅ EXCELLENT - High-quality, scientifically coherent cluster
- **0.6-0.8**: ✅ GOOD - Acceptable scientific coherence
- **<0.6**: ⚠️ REVIEW - Low coherence, needs review

---

## 🚩 Quality Flags

### Automatic Flags Generated

**Excellent Quality**:
- ✅ EXCELLENT: High-quality, scientifically coherent cluster (score ≥0.8)

**Good Quality**:
- ✅ GOOD: Acceptable scientific coherence (score 0.6-0.8)

**Needs Review**:
- ⚠️ REVIEW: Low coherence - consider reclassification (score <0.6)

**Specific Issues**:
- 🚩 METHODOLOGICAL: Mixed methodologies - may need splitting (method score <0.5)
- 🚩 CONCEPTUAL: Unclear conceptual framework (framework score <0.4)
- ⚠️ TEMPORAL: Wide time span - methods may have evolved (temporal score <0.5)

---

## 🎬 Recommended Actions

Based on validation scores, the system recommends:

### ACCEPT (Score ≥0.8)
- Cluster is scientifically valid
- Proceed with hypothesis generation

### ACCEPT WITH REVIEW (Score 0.6-0.8)
- Generally valid but needs manual review
- Validate key papers manually
- Check for edge cases

### SPLIT (Score <0.6 with specific issues)
- **By Methodology**: If methodological flag present
- **By Time Period**: If temporal flag present
- Cluster may contain multiple sub-topics

### RECLASSIFY (Score <0.6, multiple issues)
- Low coherence across multiple dimensions
- Consider different clustering parameters
- May need manual intervention

---

## 🔧 Configuration in Web App

### Current Setup (app.py)

```python
# Standard Validation (Always Applied)
validator = ClassificationValidator()
validation_results = validator.validate_all_clusters(
    valid_df, 
    labels, 
    text_column='abstract_text'  # Fixed for BCH dataset
)

# Custom Criteria (Optional - Enabled by default)
if config['use_custom_criteria']:
    custom_validator = CustomCriteriaValidator()
    
    # Add custom criteria with weights
    custom_validator.add_criterion(
        DataAvailabilityCriterion(), 
        weight=0.15
    )
    custom_validator.add_criterion(
        ClinicalTrialSponsorCriterion(), 
        weight=0.10
    )
    custom_validator.add_criterion(
        ReplicationStatusCriterion(), 
        weight=0.10
    )
    
    custom_results = custom_validator.evaluate_all_clusters(
        valid_df, 
        labels
    )
```

---

## 📊 Example Validation Output

### Cluster 4 (11 papers) - ML in Cardiology

**Scores**:
- Methodological Coherence: 0.86 (Computational)
- Framework Coherence: 0.72 (Predictive)
- Temporal Coherence: 0.85 (2020-2025)
- Internal Consistency: 0.78
- MeSH Coherence: 0.65

**Overall Score**: 0.78 ✅ GOOD

**Custom Criteria**:
- Data Availability: 0.70 (Some datasets mentioned)
- Sponsor: 0.60 (Mixed funding)
- Replication: 0.55 (Limited replication)

**Final Score**: 0.68

**Flags**:
- ✅ GOOD: Acceptable scientific coherence

**Recommendation**: ACCEPT WITH REVIEW

---

## 🎯 How to Modify Criteria

### Add New Custom Criterion

1. Create criterion class in `src/extraction/custom_criteria.py`:

```python
class MyCustomCriterion(ValidationCriterion):
    def __init__(self):
        super().__init__(
            name="my_criterion",
            description="My custom validation"
        )
    
    def evaluate(self, cluster_df: pd.DataFrame) -> float:
        # Your logic here
        score = calculate_score(cluster_df)
        return score
```

2. Add to pipeline in `app.py`:

```python
custom_validator.add_criterion(
    MyCustomCriterion(), 
    weight=0.10
)
```

### Adjust Weights

Modify weights in `app.py`:

```python
# Increase data availability importance
custom_validator.add_criterion(
    DataAvailabilityCriterion(), 
    weight=0.25  # Increased from 0.15
)
```

---

## 📚 References

- Standard criteria based on scientific research best practices
- MeSH terms from NLM Medical Subject Headings
- Custom criteria inspired by reproducibility initiatives
- Weights calibrated through empirical testing

---

**Document Version**: 1.0
**Last Updated**: November 9, 2025
**Code Location**: 
- `src/extraction/classification_validator.py` (Standard criteria)
- `src/extraction/custom_criteria.py` (Custom criteria)
- `app.py` (Web app configuration)
