# Experimental Design Guide for Reproducible Hypotheses

**Target Audience**: Research agents, data scientists, computational researchers
**Purpose**: Detailed experimental protocols with feedback loops
**Focus**: Going beyond state-of-the-art through iterative refinement

---

## 🎯 Executive Summary

This guide provides **actionable experimental designs** for 15 reproducible hypotheses identified through semantic analysis of 2,000 biomedical papers.

**Key Innovation**: Feedback loops enable continuous refinement, transforming initial findings into breakthrough discoveries.

### Priority Matrix

```
HIGH IMPACT + HIGH REPRODUCIBILITY (PRIORITY 1):
├── Hypothesis #3: ML in Cardiology (Cluster 4)
├── Hypothesis #4: Replication Study (Cluster 5)
└── Hypothesis #5: Cross-Cluster Application (5→2)

QUICK WINS (PRIORITY 2):
├── Hypothesis #1: Meta-Analysis Neurology (Cluster 5)
└── Hypothesis #2: Meta-Analysis Development (Cluster 2)
```

---

## 🔬 HYPOTHESIS #3: Machine Learning in Cardiology

### Overview

**Cluster**: 4 (11 ML-based cardiology papers)
**Reproducibility**: HIGH (8/10)
**Difficulty**: MEDIUM (6/10)
**Impact**: MEDIUM-HIGH (7/10)
**Time**: 2-4 weeks

### Hypothesis Statement

> **Primary**: ML models trained on cardiology datasets can achieve >10% improvement in predictive accuracy by incorporating features from related research domains (Clusters 2, 5, 13).

> **Secondary**: Ensemble methods combining domain-specific models will outperform single-domain models, with greatest gains in edge cases (AUC improvement >15% in bottom quartile).

### Sample Papers from Cluster 4

**Paper 1**: Machine Learning and Clinical Predictors of Mortality in Cardiac Arrest Patients: A Comprehensive Analysis
- **PMID**: Available in cluster analysis
- **Focus**: ML-based mortality prediction in cardiac arrest
- **Methods**: Random Forest, XGBoost, ensemble methods
- **Dataset**: Hospital cardiac arrest registry

**Paper 2**: Generating Risk Reduction Analytics in Complex Cardiac Care Environments (GR²AC³E): Risk Prediction in Congenital Catheterization
- **PMID**: Available in cluster analysis
- **Focus**: Risk prediction in pediatric cardiac catheterization
- **Methods**: Predictive modeling, risk stratification
- **Dataset**: Congenital cardiac catheterization procedures

**Paper 3**: AI-PEDURO - Artificial Intelligence in Pediatric Urology: Protocol for a Living Scoping Review
- **PMID**: Available in cluster analysis
- **Focus**: AI applications in pediatric care
- **Methods**: Scoping review methodology
- **Relevance**: Cross-domain AI applications

**Additional Related Papers**:
- "Neutrophil-to-lymphocyte ratio: link to congestion, inflammation, and mortality in outpatients with heart failure" (PMID: 40024803)
- "Machine learning-based automatic estimation of cortical atrophy using brain CT images" (PMID: 36042322)

---

## 📋 Experimental Protocol

### Phase 1: Dataset Acquisition (Week 1, Days 1-3)

**Objective**: Obtain and harmonize public datasets

**Datasets to Target**:

1. **MIMIC-III / MIMIC-IV** (ICU data with cardiac outcomes)
   - **URL**: https://physionet.org/content/mimiciii/
   - **Access**: Requires PhysioNet credentialing (free, ~1 week)
   - **Size**: 60,000+ ICU admissions
   - **Features**: Vitals, labs, medications, procedures, outcomes
   - **Cardiac subset**: ~15,000 patients with cardiac conditions
   - **License**: PhysioNet Credentialed Health Data License

2. **PTB-XL ECG Database**
   - **URL**: https://physionet.org/content/ptb-xl/
   - **Access**: Open access, no registration required
   - **Size**: 21,837 clinical 12-lead ECGs (10 seconds each)
   - **Features**: Raw ECG signals, diagnostic labels, patient metadata
   - **Labels**: 71 different diagnostic classes
   - **License**: Creative Commons Attribution 4.0

3. **Heart Disease UCI**
   - **URL**: https://archive.ics.uci.edu/ml/datasets/heart+disease
   - **Access**: Open access, direct download
   - **Size**: 303 patients (Cleveland database)
   - **Features**: 14 attributes (age, sex, chest pain type, BP, cholesterol, etc.)
   - **Target**: Presence of heart disease (5 classes)
   - **License**: Public domain

4. **Framingham Heart Study** (if accessible)
   - **URL**: https://framinghamheartstudy.org/fhs-for-researchers/
   - **Access**: Requires application and approval
   - **Size**: 5,000+ participants with longitudinal data
   - **Features**: Cardiovascular risk factors, outcomes over decades
   - **Note**: May require collaboration with approved researchers

5. **eICU Collaborative Research Database**
   - **URL**: https://physionet.org/content/eicu-crd/
   - **Access**: Requires PhysioNet credentialing
   - **Size**: 200,000+ ICU admissions from 200+ hospitals
   - **Features**: Similar to MIMIC but multi-center
   - **Cardiac subset**: Large cardiac patient population

6. **Paper-Specific Datasets**
   - Check supplementary materials of papers in Cluster 4
   - Contact corresponding authors for data sharing
   - Look for Zenodo/Figshare/OSF repositories linked in papers

**Process**:
```
1. Review all 11 papers in Cluster 4
2. Extract dataset names and access methods
3. Request access (PhysioNet, UCI ML Repository)
4. Download and verify data integrity
5. Document data use agreements

Quality Gate: ≥2 datasets with n>1000 samples
```

**Data Harmonization**:
```python
# Standardize features across datasets
common_features = {
    'age': ['age', 'Age', 'patient_age'],
    'sex': ['sex', 'gender', 'Sex'],
    'bp_systolic': ['sbp', 'sys_bp', 'systolic_bp']
}

# Impute missing values (MICE algorithm)
# Normalize numerical features (StandardScaler)
# Encode categorical variables (OneHotEncoder)

Quality Gate: 
- All datasets have same feature set
- Missing values <5%
- No data leakage between train/test
```

---

### Phase 2: Baseline Reproduction (Week 1, Days 4-7)

**Objective**: Reproduce published models to establish baseline

**Process**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Reproduce Paper 1's model
baseline_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train-test split (80-20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train and evaluate
baseline_rf.fit(X_train, y_train)
y_proba = baseline_rf.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, y_proba)

print(f"Baseline AUC: {baseline_auc:.3f}")

Quality Gate: Reproduce within 5% of published AUC
```

**Documentation**:
- Record all hyperparameters
- Document any deviations from paper
- Save baseline predictions for comparison

---

### Phase 3: Cross-Domain Feature Engineering (Week 2)

**Objective**: Create novel features from related clusters

**Feature Discovery**:
```
Cluster 2 (Development, 54 papers):
  → Extract: Growth trajectories, developmental markers
  → Relevant for: Pediatric cardiology

Cluster 5 (Neurology, 19 papers):
  → Extract: Neurological comorbidities, cognitive scores
  → Relevant for: Post-surgery outcomes

Cluster 13 (Mechanical Support, 28 papers):
  → Extract: Device parameters, support duration
  → Relevant for: Heart failure predictions
```

**Novel Feature Creation**:
```python
# Example: Composite risk score
def create_cardio_neuro_risk(df_cardio, df_neuro):
    """Combines cardiac and neurological markers"""
    return (
        df_cardio['ejection_fraction'] * 0.4 +
        df_neuro['cognitive_score'] * 0.3 +
        df_cardio['age'] / 100 * 0.3
    )

# Example: Developmental adjustment
def dev_adjusted_cardiac_index(df_cardio, df_dev):
    """Adjusts cardiac metrics by developmental stage"""
    return (
        df_cardio['cardiac_index'] / 
        df_dev['expected_for_age']
    )

# Create ≥10 novel cross-domain features
```

**Feature Selection with Feedback**:
```python
from sklearn.feature_selection import RFECV

# Recursive feature elimination with CV
selector = RFECV(
    estimator=RandomForestClassifier(),
    cv=5,
    scoring='roc_auc'
)
selector.fit(X_train_augmented, y_train)

# Iterative refinement
best_auc = baseline_auc
for iteration in range(10):
    # Try adding features one by one
    # Keep if improves CV performance
    # Stop when no improvement

Quality Gate: ≥5 novel features selected
```

---

### Phase 4: Advanced Modeling (Week 3)

**Objective**: Build ensemble combining domain-specific models

**Ensemble Architecture**:
```python
class DomainEnsemble:
    def __init__(self):
        # Domain-specific models
        self.cardio_model = XGBClassifier()
        self.neuro_model = LGBMClassifier()
        self.mech_model = CatBoostClassifier()
        
        # Meta-learner
        self.meta_model = LogisticRegression()
    
    def fit(self, X_cardio, X_neuro, X_mech, y):
        # Train each domain model
        self.cardio_model.fit(X_cardio, y)
        self.neuro_model.fit(X_neuro, y)
        self.mech_model.fit(X_mech, y)
        
        # Stack predictions as meta-features
        meta_features = np.column_stack([
            self.cardio_model.predict_proba(X_cardio)[:, 1],
            self.neuro_model.predict_proba(X_neuro)[:, 1],
            self.mech_model.predict_proba(X_mech)[:, 1]
        ])
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)

# Evaluate ensemble
ensemble = DomainEnsemble()
ensemble.fit(X_train_cardio, X_train_neuro, X_train_mech, y_train)
ensemble_auc = evaluate(ensemble, X_test)

improvement = (ensemble_auc - baseline_auc) / baseline_auc * 100
print(f"Improvement: {improvement:.1f}%")

Quality Gate: >10% improvement over baseline
```

**Interpretability**:
```python
import shap

# SHAP values for explainability
explainer = shap.TreeExplainer(ensemble.cardio_model)
shap_values = explainer.shap_values(X_test)

# Identify most important cross-domain features
shap.summary_plot(shap_values, X_test)

# Document novel insights
```

---

### Phase 5: Feedback Loops (Week 4)

**Loop 1: Error Analysis (Daily)**
```python
# Analyze misclassifications
false_positives = (y_pred == 1) & (y_test == 0)
false_negatives = (y_pred == 0) & (y_test == 1)

# Characterize errors
fp_features = X_test[false_positives].describe()
fn_features = X_test[false_negatives].describe()

# Action: Create subgroup-specific models or features
# Iterate until error patterns understood
```

**Loop 2: Cross-Cluster Validation (Weekly)**
```python
# Test on related clusters
cluster_2_data = load_cluster_2_data()
cluster_5_data = load_cluster_5_data()

# Evaluate generalization
auc_cluster_2 = evaluate(ensemble, cluster_2_data)
auc_cluster_5 = evaluate(ensemble, cluster_5_data)

# If performance drops:
#   → Investigate domain differences
#   → Apply domain adaptation
#   → Retrain with mixed data
```

**Loop 3: External Validation (End of Week 4)**
```python
# Test on completely independent dataset
X_external, y_external = load_external_hospital_data()

external_auc = evaluate(ensemble, X_external)

if external_auc < ensemble_auc * 0.9:
    # Significant performance drop
    # Apply domain adaptation techniques
    adapted_model = domain_adaptation(
        source_model=ensemble,
        target_data=X_external
    )
```

---

## 🚀 Beyond State-of-the-Art Strategies

### Innovation 1: Federated Learning
```
Current: Single-institution models
Innovation: Multi-institution collaborative learning

Method:
  1. Deploy model to partner institutions
  2. Each trains on local data
  3. Share only model updates (not data)
  4. Aggregate using federated averaging

Value: Larger training set, better generalization, privacy-preserving

Tools: PySyft, TensorFlow Federated
Time: +2 weeks
```

### Innovation 2: Causal ML
```
Current: Predictive (correlation)
Innovation: Causal (intervention effects)

Method:
  1. Build causal graph from domain knowledge
  2. Use DoWhy library for causal inference
  3. Estimate treatment effects
  4. Validate with RCT data if available

Value: Understand WHY, identify interventions, robust to shift

Tools: DoWhy, EconML
Time: +2 weeks
```

### Innovation 3: Continual Learning
```
Current: Static model
Innovation: Continuously learning from new data

Method:
  1. Implement elastic weight consolidation (EWC)
  2. Monitor for concept drift
  3. Retrain when performance degrades
  4. Maintain model versioning

Value: Adapts to changing populations, long-term viability

Tools: Avalanche, River
Time: +1 week
```

---

## 📊 Success Metrics

### Minimum Success (Required)
- ✅ Baseline reproduced (within 5%)
- ✅ ≥10 cross-domain features created
- ✅ Ensemble model trained
- ✅ Code on GitHub

### Target Success (Expected)
- ✅ All minimum criteria
- ✅ >10% AUC improvement
- ✅ External validation performed
- ✅ Interpretability analysis done
- ✅ Manuscript submitted

### Exceptional Success (Aspirational)
- ✅ All target criteria
- ✅ >15% AUC improvement
- ✅ Federated learning implemented
- ✅ Causal analysis performed
- ✅ Clinical deployment pilot

---

## 📦 Deliverables

### Code & Models
1. **GitHub Repository**: Complete reproducible code
2. **Trained Models**: Serialized models (pickle/joblib)
3. **Docker Container**: Reproducible environment
4. **REST API**: For model deployment

### Documentation
1. **Technical Report**: Detailed methodology
2. **Model Card**: Characteristics, limitations, intended use
3. **Feature Dictionary**: All features with definitions
4. **Performance Report**: Metrics on all datasets

### Publications
1. **Preprint**: arXiv/medRxiv
2. **Journal**: JMIR, Nature Digital Medicine
3. **Conference**: NeurIPS ML4H, AMIA

---

## 🔄 Universal Feedback Loop Framework

### 3-Loop System for All Hypotheses

**Loop 1: Internal Validation (Fast - Hours/Days)**
```
Purpose: Rapid iteration
Frequency: Multiple times daily

Process:
  1. Make change (feature, parameter, model)
  2. Evaluate on validation set
  3. Compare to previous best
  4. Keep if better, else revert
  5. Document

Stopping: <1% improvement or time budget exhausted
```

**Loop 2: Cross-Domain (Medium - Days/Weeks)**
```
Purpose: Incorporate related research
Frequency: Weekly

Process:
  1. Identify related clusters
  2. Extract relevant methods/features
  3. Adapt to current problem
  4. Test improvement
  5. Integrate if successful

Sources: Related clusters, different specialties, non-medical ML
```

**Loop 3: External Validation (Slow - Weeks/Months)**
```
Purpose: Real-world generalization
Frequency: At milestones

Process:
  1. Test on independent data
  2. Assess performance gap
  3. Investigate degradation causes
  4. Implement fixes
  5. Re-evaluate

Actions: Domain adaptation, retraining, feature engineering
```

---

## 📝 Additional Hypotheses (Brief Protocols)

See `EXPERIMENTAL_DESIGN_APPENDIX.md` for:
- Hypothesis #1: Meta-Analysis Neurology (detailed protocol)
- Hypothesis #2: Meta-Analysis Development
- Hypothesis #4: Replication Study
- Hypothesis #5: Cross-Cluster Application
- All 15 hypotheses with protocols

---

## 📚 References & Resources

### Public Datasets for Cardiac ML

**PhysioNet (Primary Source)**:
- Main Portal: https://physionet.org/
- MIMIC-III: https://physionet.org/content/mimiciii/
- MIMIC-IV: https://physionet.org/content/mimiciv/
- PTB-XL ECG: https://physionet.org/content/ptb-xl/
- eICU: https://physionet.org/content/eicu-crd/
- Credentialing: https://physionet.org/about/citi-course/

**UCI Machine Learning Repository**:
- Heart Disease: https://archive.ics.uci.edu/ml/datasets/heart+disease
- All Medical Datasets: https://archive.ics.uci.edu/ml/datasets.php?format=&task=&att=&area=life&numAtt=&numIns=&type=&sort=nameUp&view=table

**Kaggle Datasets**:
- Heart Disease: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- ECG Heartbeat: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- Cardiovascular Disease: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

**Papers with Code**:
- Medical Datasets: https://paperswithcode.com/datasets?mod=medical
- Cardiac Datasets: https://paperswithcode.com/datasets?q=cardiac

### Tools & Libraries

**Machine Learning**:
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- CatBoost: https://catboost.ai/
- PyTorch: https://pytorch.org/
- TensorFlow: https://www.tensorflow.org/

**Interpretability**:
- SHAP: https://shap.readthedocs.io/
- LIME: https://github.com/marcotcr/lime
- InterpretML: https://interpret.ml/

**Medical ML Frameworks**:
- MONAI (Medical imaging): https://monai.io/
- TorchXRayVision: https://github.com/mlmed/torchxrayvision
- MedCAT (NLP): https://github.com/CogStack/MedCAT

**Federated Learning**:
- PySyft: https://github.com/OpenMined/PySyft
- TensorFlow Federated: https://www.tensorflow.org/federated
- Flower: https://flower.dev/

**Causal Inference**:
- DoWhy: https://microsoft.github.io/dowhy/
- EconML: https://econml.azurewebsites.net/
- CausalML: https://causalml.readthedocs.io/

### Tutorials & Courses

**ML in Healthcare**:
- MIT 6.S897: https://mlhc.mit.edu/
- Stanford CS329D: https://cs329d.stanford.edu/
- Fast.ai Medical Imaging: https://www.fast.ai/

**Reproducible Research**:
- Papers with Code: https://paperswithcode.com/
- ML Reproducibility Challenge: https://paperswithcode.com/rc2020
- Reproducibility Checklist: https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf

### Paper Access

**PubMed Papers from This Study**:
- Access via PMID: https://pubmed.ncbi.nlm.nih.gov/[PMID]/
- Full text (when available): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC[PMCID]/
- Boston Children's Hospital papers: Search "Boston Children's Hospital" on PubMed

**Preprint Servers**:
- arXiv (CS/ML): https://arxiv.org/
- medRxiv (Medicine): https://www.medrxiv.org/
- bioRxiv (Biology): https://www.biorxiv.org/

### Data Sharing Platforms

**Research Data Repositories**:
- Zenodo: https://zenodo.org/
- Figshare: https://figshare.com/
- OSF (Open Science Framework): https://osf.io/
- Dryad: https://datadryad.org/

**Code Repositories**:
- GitHub: https://github.com/
- GitLab: https://gitlab.com/
- Papers with Code: https://paperswithcode.com/

### Ethical & Regulatory

**Data Use Agreements**:
- PhysioNet DUA: https://physionet.org/about/dua/
- NIH Data Sharing: https://sharing.nih.gov/

**IRB & Ethics**:
- CITI Program: https://about.citiprogram.org/
- NIH Human Subjects Training: https://grants.nih.gov/policy/humansubjects/training.htm

**HIPAA Compliance**:
- HHS HIPAA: https://www.hhs.gov/hipaa/
- De-identification Guidance: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/

---

**Document Version**: 1.1
**Last Updated**: November 9, 2025
**Repository**: https://github.com/ebaenamar/research-semantic-poc
**Contact**: Open an issue on GitHub for questions
