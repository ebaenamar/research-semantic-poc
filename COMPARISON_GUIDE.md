# 🆚 Comparación: App V1 vs App V2

## 🌐 URLs de Acceso

```
App V1 (Semantic Classification):  http://localhost:8501
App V2 (Reproducible Hypotheses):  http://localhost:8502
```

**Abre ambas en tabs diferentes para comparar side-by-side**

---

## 🎯 Configuración Recomendada para Comparación Justa

### App V1 (puerto 8501)
```
Dataset Size: 850
Embedding Model: all-MiniLM-L6-v2
Min Cluster Size: 10
Clustering Mode: ⚡ Standard (para comparar con V2)
Filter Incoherent: ❌ No
Use Custom Criteria: ✅ Yes
```

### App V2 (puerto 8502)
```
Dataset Size: 850
Embedding Model: all-MiniLM-L6-v2
Min Cluster Size: 10
UMAP Components: 10

Weights (defaults):
  Computational: 0.40
  Data Availability: 0.30
  No Trials: 0.15
  No Lab: 0.15
  
Min Threshold: 0.30
```

---

## 📊 Diferencias Esperadas

### 1. **Número de Clusters**

#### App V1 (Standard Mode)
```
Clusters: 20-30
Noise: 25-35%
Clustering Rate: 65-75%

Razón: HDBSCAN estándar con validación posterior
```

#### App V2
```
Clusters: 15-25
Noise: 20-30%
Reproducible Clusters: 8-12 (subset)

Razón: Filtra solo clusters reproducibles (score ≥0.3)
```

**Diferencia**: V2 muestra MENOS clusters porque filtra por reproducibilidad

---

### 2. **Tipo de Hipótesis**

#### App V1 - Hipótesis Genéricas
```
Hypothesis: Cluster 7 (9 papers)

Title: "ML/AI Application"

Description:
"Overview: This cluster contains 9 ML-based papers published 
between 2022-2025 (3 year span).

Reproducibility: MEDIUM (5.8/10) | Difficulty: LOW (1.9/10) | Impact: LOW (4.7/10)

Data Analysis (Real):
- 6/9 papers (67%) mention available datasets
- 9/9 papers (100%) are computational/ML-based
- 3/9 papers (33%) explicitly mention future work/gaps
- Average publication year: 2023.67

Methodology: Computational/bioinformatics (coherence: 0.80)
Framework: Descriptive approach (coherence: 1.00)
Common Themes: deep learning, prediction, classification

Research Opportunity: Computational focus with potential for 
ML/AI improvements. High validation score indicates coherence.

Recommended Approach: Develop or improve ML/AI models using 
insights from these papers. Focus on novel architectures, 
better features, or cross-domain transfer learning."
```

**Características**:
- ✅ Scores numéricos reales
- ✅ Data analysis detallado
- ✅ Coherence metrics
- ❌ NO tiene verification plan paso a paso
- ❌ NO tiene requirements list
- ❌ NO tiene PMIDs de papers
- ❌ Recomendación muy general

---

#### App V2 - Hipótesis Específicas
```
Hypothesis #1: Priority 9.0 ⭐⭐⭐

Title: "Replicate Random Forest, LSTM on MIMIC-III (Cluster 8)"

Type: Replication
Reproducibility: VERY HIGH
Difficulty: LOW
Impact: HIGH (validates existing research)
Time: 1-3 weeks

📋 HYPOTHESIS:
Replicate findings from 21 papers in cluster 8 using MIMIC-III, 
eICU. Focus on replicating Random Forest, LSTM, XGBoost for 
mortality prediction, AUC, sensitivity/specificity.
Key paper: 'Deep Learning for Prediction of Acute Kidney Injury...'

📦 REQUIREMENTS:
- Datasets: MIMIC-III, eICU
- Methods: Random Forest, LSTM, XGBoost
- Original analysis code if available
- Documentation of original methods

✅ VERIFICATION PLAN:
1. Download MIMIC-III, eICU datasets
2. Access papers: PMID 32158395, PMID 31234567, PMID 29876543
3. Verify data integrity and completeness
4. Reproduce Random Forest, LSTM, XGBoost implementation
5. Re-run original analyses with same parameters
6. Compare mortality prediction, AUC with published results
7. Calculate effect size differences (Cohen's d, correlation)
8. Document discrepancies (data version, preprocessing)
9. Test generalizability on different subsets

🔍 SPECIFIC DETAILS:
📊 Datasets: MIMIC-III, eICU, GitHub repository
🔬 Methods: Random Forest, LSTM, XGBoost, Logistic Regression
📈 Outcomes: mortality prediction, AUC, sensitivity, accuracy

📚 KEY PAPERS:
Paper 1:
Title: Deep Learning for Prediction of Acute Kidney Injury...
PMID: 32158395 [clickeable link]
Year: 2020
Abstract: We developed LSTM models for mortality prediction...

Paper 2:
Title: Comparative Analysis of ML Methods for ICU Mortality
PMID: 31234567 [link]
Year: 2019
...
```

**Características**:
- ✅ 4 tipos específicos por cluster (Replication, ML App, Meta, Cross-Cluster)
- ✅ Verification plan con 9 pasos ESPECÍFICOS
- ✅ Requirements detallados
- ✅ PMIDs con links a PubMed
- ✅ Datasets específicos mencionados (MIMIC-III)
- ✅ Métodos específicos (Random Forest, LSTM)
- ✅ Outcomes específicos (AUC, mortality)
- ✅ Tiempo estimado concreto (1-3 weeks)
- ✅ Accionable inmediatamente

---

### 3. **Validación de Clusters**

#### App V1 - Validación Robusta
```
Tab: ✅ Validation

Cluster 7:
  Overall Score: 5.81/10 ✅ Pass
  
  Breakdown:
  - Methodological Coherence: 0.80 (35%)
  - Framework Coherence: 1.00 (25%)
  - Temporal Coherence: 0.75 (15%)
  - Internal Consistency: 0.68 (15%)
  - MeSH Coherence: 0.45 (10%)
  
  Custom Criteria:
  - Data Availability: 0.67
  - Clinical Trial: 0.10
  - Replication: 0.20
```

**Ventaja**: Ve calidad científica de cada cluster

---

#### App V2 - Reproducibility Score
```
Tab: 🎯 Clusters

Cluster 8 (21 papers):
  Reproducibility: 0.77
  Computational: 0.71
  Data Available: 0.62
  Size: 21
```

**Ventaja**: Enfocado en reproducibilidad práctica

---

### 4. **Clustering Strategies**

#### App V1 - 5 Modos
```
⚡ Standard
🔬 Domain-Aware Only
🎯 Adaptive Only
🎯🔬 Domain-Aware + Adaptive
🎯 Hierarchical Funnel (Recommended)
```

**Ventaja**: Puedes experimentar con diferentes estrategias

---

#### App V2 - 1 Modo Simple
```
Solo HDBSCAN estándar
```

**Ventaja**: Simplicidad, probado que funciona

---

## 🧪 Experimento de Comparación

### Paso 1: Configurar Ambas Apps
```
Usa MISMA configuración en ambas:
- Dataset: 850 papers
- Embedding: all-MiniLM-L6-v2
- Min Cluster Size: 10
- V1: Standard mode (no funnel)
```

### Paso 2: Ejecutar Pipelines
```
App V1 (8501): Click "🚀 Run Pipeline"
App V2 (8502): Click "🚀 Generate Hypotheses"
```

### Paso 3: Comparar Métricas

#### Clusters:
```
V1: ¿Cuántos clusters? ¿% ruido?
V2: ¿Cuántos clusters? ¿Cuántos reproducibles?
```

#### Hypotheses:
```
V1: Selecciona Top 3 hipótesis
   → Lee descripción
   → ¿Tiene plan paso a paso? ❌
   → ¿Tiene PMIDs? ❌
   → ¿Accionable? 🤔

V2: Selecciona Top 3 hipótesis  
   → Lee descripción
   → ¿Tiene verification plan? ✅
   → ¿Tiene PMIDs? ✅
   → ¿Accionable? ✅
```

### Paso 4: Comparar MISMO Cluster

Encuentra un cluster que aparezca en ambas (e.g., cluster sobre ML):

```
V1 - Cluster 7 (9 papers ML):
  Validation: 5.81/10 ✅
  Hypothesis: "Develop ML models... (genérico)"
  No PMIDs
  No verification plan

V2 - Cluster 8 (21 papers ML):
  Reproducibility: 0.77
  Hypothesis: "Replicate Random Forest on MIMIC-III"
  4 hypotheses (Replication, ML App, Meta, Cross-Cluster)
  PMIDs: 32158395, 31234567, 29876543
  Verification: 9 pasos específicos
```

---

## 📈 Resultados Esperados

### App V1 Strengths:
```
✅ Validación científica robusta (8 criterios)
✅ Múltiples estrategias de clustering
✅ Scores de coherence detallados
✅ Thematic coherence filtering
✅ Hierarchical funnel para máxima pureza
✅ Mejor para: Publicaciones, rigor científico
```

### App V1 Weaknesses:
```
❌ Hipótesis genéricas sin detalles
❌ No verification plan
❌ No PMIDs de papers
❌ No datasets específicos
❌ Difícil ejecutar inmediatamente
```

---

### App V2 Strengths:
```
✅ 4 tipos de hipótesis CONCRETAS
✅ Verification plan paso a paso (9 pasos)
✅ PMIDs con links a PubMed
✅ Datasets específicos (MIMIC-III, eICU)
✅ Métodos específicos (Random Forest, LSTM)
✅ Requirements detallados
✅ Tiempo estimado realista
✅ Mejor para: Ejecución inmediata, reproducibilidad
```

### App V2 Weaknesses:
```
❌ No validación científica robusta
❌ Solo 1 modo de clustering
❌ No coherence metrics
❌ Menos features
```

---

## 🎯 Cuándo Usar Cada Una

### Usa App V1 (puerto 8501) si:
- Necesitas **rigor científico** para publicación
- Quieres **explorar diferentes clustering strategies**
- Te interesa **validación robusta** (8 criterios)
- Quieres **filtrar clusters incoherentes**
- Necesitas **hierarchical funnel** para pureza máxima
- El objetivo es **generar muchas ideas** para evaluar después

### Usa App V2 (puerto 8502) si:
- Quieres **ejecutar hipótesis AHORA**
- Necesitas **verification plans específicos**
- Quieres **PMIDs de papers clave**
- Te interesa **reproducibilidad práctica**
- Prefieres **simplicidad sobre features**
- El objetivo es **acción inmediata** con datos públicos

---

## 🔄 Best Practices

### Workflow Combinado (Recomendado):

```
1. Exploración (App V1):
   - Usa Hierarchical Funnel
   - Valida coherencia científica
   - Identifica clusters prometedores
   - Exporta resultados

2. Ejecución (App V2):
   - Focaliza en clusters reproducibles
   - Obtén verification plans
   - Extrae PMIDs y datasets
   - Ejecuta hipótesis

Resultado: Rigor científico + Accionabilidad
```

---

## 📊 Test Rápido

### ⏱️ 5 Minutos:

1. **App V1** (http://localhost:8501):
   - Run Pipeline (Standard mode)
   - Ve a Tab "Hypotheses"
   - Lee Hypothesis #1
   - ¿Puedes ejecutarla HOY? 🤔

2. **App V2** (http://localhost:8502):
   - Generate Hypotheses
   - Ve a Tab "Hypotheses"
   - Expande Hypothesis #1
   - Scroll a "Verification Plan"
   - Scroll a "Key Papers"
   - ¿Puedes ejecutarla HOY? ✅

3. **Veredicto**:
   - Si necesitas papers AHORA → V2
   - Si necesitas validación científica → V1
   - Si quieres ambos → Usa V1 primero, luego V2

---

## 🛑 Cerrar Apps

```bash
# Cerrar App V1
kill $(lsof -t -i:8501)

# Cerrar App V2
kill $(lsof -t -i:8502)

# Cerrar ambas
pkill -f "streamlit run"
```

---

**Apps corriendo ahora**:
- ✅ App V1: http://localhost:8501
- ✅ App V2: http://localhost:8502

**Abre ambas y compara los resultados!**
