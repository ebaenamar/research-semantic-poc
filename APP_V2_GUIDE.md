# App V2 - Reproducible Hypothesis Generator

## 🆚 Diferencias: App V1 vs App V2

### App V1 (app.py) - ❌ Problemas
```
❌ Hierarchical Funnel demasiado estricto (0 clusters, 100% ruido)
❌ Hipótesis genéricas sin plan de acción
❌ No tiene verification_plan paso a paso
❌ No tiene requirements list
❌ Priority score incorrecto
```

### App V2 (app_v2.py) - ✅ Basado en Script Original
```
✅ Clustering simple HDBSCAN (funciona)
✅ 4 tipos de hipótesis CONCRETAS por cluster
✅ Verification plan detallado (8-9 pasos)
✅ Requirements list específica
✅ Priority score correcto (reproducibility + impact + ease)
✅ Weights configurables en UI
```

---

## 🎯 Características Principales

### 1. **Clustering Simple que Funciona**
```python
# Solo HDBSCAN estándar, sin complicaciones
clusterer = SemanticClusterer(method='hdbscan')
labels = clusterer.cluster_hdbscan(reduced, min_cluster_size=10)
```

### 2. **4 Tipos de Hipótesis Concretas**

#### A. ML Application
```
Title: "Improve ML Models with Cross-Cluster Features"
Hypothesis: "ML models can achieve >10% improvement by incorporating features from related domains"
Requirements:
  - Public datasets
  - scikit-learn/TensorFlow/PyTorch
  - GPU optional
  - Python 3.8+

Verification Plan (8 steps):
  1. Download datasets from papers
  2. Reproduce baseline models
  3. Implement improved models with new features
  4. Compare performance (AUC, accuracy, F1)
  5. Statistical significance testing
  6. K-fold cross-validation
  7. Test on held-out validation set
  8. Document with confidence intervals

Time: 2-4 weeks
Difficulty: MEDIUM
Impact: MEDIUM-HIGH
```

#### B. Meta-Analysis
```
Title: "Systematic Meta-Analysis of N Studies"
Hypothesis: "Meta-analysis will reveal consistent effect sizes and moderators"
Requirements:
  - Papers with effect sizes
  - R metafor package
  - Stats knowledge
  - Full-text access

Verification Plan (8 steps):
  1. Extract effect sizes (d, r, OR, RR)
  2. Convert to common metric
  3. Calculate pooled effect (random-effects)
  4. Assess heterogeneity (I², Q, τ²)
  5. Subgroup analyses
  6. Publication bias check (funnel plot, Egger)
  7. Sensitivity analyses
  8. Report with PRISMA guidelines

Time: 1-2 weeks
Difficulty: LOW-MEDIUM
Impact: MEDIUM
```

#### C. Replication Study
```
Title: "Replicate Key Findings with Public Data"
Hypothesis: "Findings can be replicated using public datasets"
Requirements:
  - Public datasets (GitHub, Figshare, Zenodo)
  - R/Python/SPSS/Stata
  - Original analysis code
  - Method documentation

Verification Plan (9 steps):
  1. Identify papers with public data
  2. Download datasets
  3. Verify data integrity
  4. Reproduce preprocessing
  5. Re-run original analyses
  6. Compare with published results
  7. Calculate effect size differences
  8. Document discrepancies
  9. Test on different subsets

Time: 1-3 weeks
Difficulty: LOW
Impact: HIGH (validates research)
```

#### D. Cross-Cluster Transfer
```
Title: "Transfer Methods Between Clusters X ↔ Y"
Hypothesis: "Methods from cluster X can be applied to cluster Y data"
Requirements:
  - Datasets from both clusters
  - Domain knowledge
  - Computational tools
  - Interpretation skills

Verification Plan (8 steps):
  1. Identify compatible datasets
  2. Adapt methods to new format
  3. Apply to new domain
  4. Compare with existing approaches
  5. Evaluate improvement
  6. Validate on held-out test set
  7. Analyze transferability factors
  8. Document what transfers vs. needs adaptation

Time: 3-6 weeks
Difficulty: MEDIUM-HIGH
Impact: HIGH (novel application)
```

---

## 🎛️ Pesos Configurables (Weights)

### En la UI - Sidebar > Reproducibility Weights

```python
Weights (deben sumar 1.0):

1. Computational/ML Methods: 0.40 (default)
   - Papers con ML, deep learning, algorithms
   - Mayor peso = prefiere métodos computacionales

2. Data Availability: 0.30 (default)
   - Papers que mencionan datasets públicos
   - Mayor peso = prefiere datos disponibles

3. No Clinical Trials: 0.15 (default)
   - Penaliza papers que requieren RCTs
   - Mayor peso = evita trials clínicos

4. No Lab Work: 0.15 (default)
   - Penaliza papers con in vitro/in vivo
   - Mayor peso = evita trabajo de laboratorio

Total = 1.0
```

### Score de Reproducibilidad
```python
reproducibility_score = (
    computational_score * 0.40 +
    data_availability * 0.30 +
    (1 - trials/10) * 0.15 +
    (1 - lab/10) * 0.15
)
```

### Min Threshold
```
Default: 0.3
Rango: 0.0 - 1.0

0.3 = Moderado (más clusters reproducibles)
0.5 = Estricto (solo muy reproducibles)
0.7 = Muy estricto (pocos clusters)
```

---

## 📊 Priority Score (Ranking de Hipótesis)

```python
priority_score = reproducibility_weight + impact_weight + ease_weight + data_bonus

Reproducibility:
- VERY HIGH: +3 puntos
- HIGH: +2 puntos
- MEDIUM: +1 punto

Impact:
- HIGH: +3 puntos
- MEDIUM-HIGH: +2 puntos
- MEDIUM: +2 puntos
- LOW: +1 punto

Difficulty (inverso):
- LOW: +2 puntos
- MEDIUM: +1 punto
- HIGH: +0 puntos

Data Available:
- Yes: +1 punto bonus
- No: +0 puntos

Max Score: ~9 puntos
```

---

## 🚀 Cómo Usar App V2

### 1. Lanzar la App
```bash
cd /Users/e.baena/CascadeProjects/research-semantic-poc
source venv/bin/activate
streamlit run app_v2.py
```

### 2. Configurar Weights
```
Sidebar > Reproducibility Weights > ⚙️ Configure Weights

Ejemplo 1 - Priorizar ML:
  Computational: 0.50 ↑
  Data Availability: 0.30
  No Trials: 0.10 ↓
  No Lab: 0.10 ↓
  
Ejemplo 2 - Priorizar Datos:
  Computational: 0.30 ↓
  Data Availability: 0.50 ↑
  No Trials: 0.10
  No Lab: 0.10

Ejemplo 3 - Evitar Experimentos:
  Computational: 0.35
  Data Availability: 0.25
  No Trials: 0.20 ↑
  No Lab: 0.20 ↑
```

### 3. Ajustar Threshold
```
Min Reproducibility Threshold: 0.3

Si obtienes 0 clusters:
  → Baja threshold a 0.2

Si obtienes muchos clusters no reproducibles:
  → Sube threshold a 0.4-0.5
```

### 4. Click "🚀 Generate Hypotheses"

### 5. Ver Resultados

**Tab: Hypotheses**
- Ranked por priority score
- Cada hypothesis expandible
- 4 tipos por cluster reproducible
- Verification plan paso a paso

**Tab: Clusters**
- Lista de clusters reproducibles
- Scores detallados
- Sample papers

**Tab: Visualization**
- UMAP scatter plot
- Coloreado por cluster

---

## 📈 Resultados Esperados

### Con 850 Papers (Boston Children's Hospital)

**Config Recomendada**:
```
Dataset Size: 850
Embedding: all-MiniLM-L6-v2
Min Cluster Size: 10
UMAP Components: 10

Weights:
  Computational: 0.40
  Data Available: 0.30
  No Trials: 0.15
  No Lab: 0.15
  
Threshold: 0.3
Max Hypotheses: 4
```

**Resultados Esperados**:
```
Computational Papers: ~600 (70%)
Clusters: 15-25
Reproducible Clusters: 8-12
Hypotheses: 24-48 (4 tipos × 6-12 clusters)

Top Hypothesis:
  Type: Replication
  Reproducibility: VERY HIGH
  Difficulty: LOW
  Impact: HIGH
  Priority Score: 8-9
  
  Verification plan: 9 pasos detallados
  Requirements: Public datasets + Python/R
  Time: 1-3 weeks
```

---

## ✅ Ventajas App V2 vs V1

| Feature | App V1 | App V2 |
|---------|--------|--------|
| Clustering | Hierarchical Funnel (0 clusters) | Simple HDBSCAN (funciona) |
| Hypotheses | Genéricas | 4 tipos concretos |
| Verification Plan | ❌ No | ✅ 8-9 pasos |
| Requirements | ❌ No | ✅ Lista detallada |
| Priority Score | Incorrecto | Correcto |
| Weights UI | Parcial | ✅ Completo |
| Actionable | ❌ No | ✅ Sí |

---

## 🎯 Ejemplo Real de Output

```
HYPOTHESIS #1 (Priority Score: 9.0)
================================

Type: Replication
Cluster: 4
Reproducibility: VERY HIGH
Difficulty: LOW
Impact: HIGH (validates existing research)
Time: 1-3 weeks

📋 HYPOTHESIS:
Key findings from cluster 4 can be independently replicated using 
publicly available datasets, validating original results and 
assessing generalizability.

📦 REQUIREMENTS:
- Public datasets (GitHub, Figshare, Zenodo, Dryad)
- Statistical software (R, Python, SPSS, Stata)
- Original analysis code if available
- Documentation of original methods

✅ VERIFICATION PLAN:
1. Identify papers with publicly available data
2. Download datasets from repositories
3. Verify data integrity and completeness
4. Reproduce original preprocessing steps
5. Re-run original analyses exactly
6. Compare results with published findings
7. Calculate effect size differences
8. Document any discrepancies and investigate causes
9. Test on different subsets/populations if available

Cluster 4 | 11 papers | Data Available: ✅
```

---

## 🔧 Troubleshooting

### Problema: 0 hypotheses generated
**Solución**:
1. Baja min_threshold a 0.2
2. Aumenta dataset_size a 2000
3. Ajusta weights para ser menos estricto

### Problema: Hypotheses muy similares
**Solución**:
1. Reduce max_hypotheses a 2-3
2. Filtra por tipo específico

### Problema: No encuentra clusters reproducibles
**Solución**:
1. Baja threshold a 0.2
2. Aumenta weight de computational a 0.5
3. Reduce min_cluster_size a 5

---

**Última actualización**: Nov 9, 2025  
**Comando**: `streamlit run app_v2.py`
