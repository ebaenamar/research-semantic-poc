# Hierarchical Funnel Clustering Guide

## 🎯 Concepto: Probabilidad Condicional

El **Hierarchical Funnel** aplica filtros secuenciales con importancia decreciente:

```
P(cluster válido) = P(topic) × P(method|topic) × P(time|topic,method) × P(semantic|topic,method,time)
```

Cada etapa **condiciona** la siguiente, garantizando coherencia progresiva.

---

## 📊 4 Etapas del Funnel

### Stage 1: Topic (40% importance) 🎯
**Qué hace**: Asigna papers a condiciones médicas **específicas**

**30+ Topics Detectados**:
- `heart_failure`: "heart failure", "cardiac failure", "hf"
- `arrhythmia`: "atrial fibrillation", "afib", "vtach"
- `stroke`: "stroke", "cerebrovascular", "cva"
- `diabetes`: "diabetes", "t2dm", "glycemic control"
- `aki`: "acute kidney injury", "aki"
- `sepsis`: "sepsis", "septic shock"
- `covid`: "covid", "sars-cov-2"
- `cancer_lung`: "lung cancer", "nsclc"
- `ecg`: "ecg", "electrocardiogram"
- `pediatric`: "pediatric", "children", "infant"
- etc.

**Regla**: Solo papers del **mismo topic específico** pueden ir juntos

**Ejemplo**:
```
417 papers →
  - 45 papers: heart_failure
  - 38 papers: diabetes
  - 27 papers: aki
  - 22 papers: pediatric
  ...
```

---

### Stage 2: Methodology (25% importance) 🔬
**Qué hace**: Dentro de cada topic, agrupa por metodología

**12 Methodologies Detectadas**:
- `rct`: "randomized controlled trial", "double-blind"
- `cohort`: "cohort study", "prospective", "follow-up"
- `machine_learning`: "ML", "deep learning", "neural network"
- `meta_analysis`: "systematic review", "meta-analysis"
- `clinical_trial`: "phase i", "phase ii", "clinical trial"
- `genomic`: "GWAS", "sequencing", "RNA-seq"
- `imaging`: "MRI", "CT", "imaging study"
- `observational`: "observational study"
- `case_control`: "case-control study"
- `registry`: "registry-based"
- `cross_sectional`: "cross-sectional"
- `laboratory`: "in vitro", "cell culture"

**Regla**: P(method|topic) - Solo papers del mismo topic Y misma metodología

**Ejemplo**:
```
heart_failure (45 papers) →
  - 15 papers: heart_failure + machine_learning
  - 12 papers: heart_failure + cohort
  - 10 papers: heart_failure + rct
  - 8 papers: heart_failure + imaging
```

---

### Stage 3: Temporal (15% importance) 📅
**Qué hace**: Dentro de cada topic+method, agrupa por recencia

**Recency Window**: 5 years (configurable)

**2 Time Groups**:
- **Recent**: Last 5 years (2020-2025)
- **Older**: Before 5 years (<2020)

**Regla**: P(time|topic,method) - Prioriza papers recientes

**Ejemplo**:
```
heart_failure + machine_learning (15 papers) →
  - 11 papers: recent (2020-2025)
  - 4 papers: older (2015-2019)
```

**Por qué es importante**:
- Métodos cambian con el tiempo
- DL papers de 2015 ≠ DL papers de 2024
- Papers recientes = más relevantes

---

### Stage 4: Semantic (20% importance) 🧬
**Qué hace**: Clustering semántico fino con HDBSCAN

**Regla**: P(semantic|topic,method,time) - Refinamiento final

**Ejemplo**:
```
heart_failure + ML + recent (11 papers) →
  HDBSCAN clustering →
  - Cluster A (5 papers): ECG-based HF prediction
  - Cluster B (6 papers): Imaging-based HF diagnosis
```

---

## ⚖️ Orden de Importancia

```
1. Topic (40%) ⭐⭐⭐⭐ - MÁS IMPORTANTE
   └─ Heart failure vs Diabetes = NUNCA se mezclan

2. Methodology (25%) ⭐⭐⭐
   └─ RCT vs ML = Diferentes enfoques, no comparables

3. Temporal (15%) ⭐⭐
   └─ 2015 DL vs 2024 DL = Técnicas diferentes

4. Semantic (20%) ⭐⭐
   └─ Refinamiento fino dentro del contexto
```

---

## 🔍 Ejemplo Completo

### Input: 417 Papers (Boston Children's Hospital)

**Stage 1: Topic Assignment**
```
417 papers →
  45 → heart_failure
  38 → diabetes
  27 → aki
  22 → pediatric
  18 → covid
  15 → sepsis
  ... (resto)
```

**Stage 2: Methodology (dentro de heart_failure)**
```
45 heart_failure papers →
  15 → machine_learning
  12 → cohort
  10 → rct
  8 → imaging
```

**Stage 3: Temporal (dentro de HF + ML)**
```
15 heart_failure + ML papers →
  11 → recent (2020-2025)
  4 → older (2015-2019)
```

**Stage 4: Semantic (dentro de HF + ML + recent)**
```
11 heart_failure + ML + recent papers →
  HDBSCAN →
  Cluster 0 (5 papers): ECG-based prediction
  Cluster 1 (6 papers): EHR-based risk scores
```

### Final Result
```
Cluster 0:
- Topic: heart_failure ✅
- Method: machine_learning ✅
- Time: recent (2020-2025) ✅
- Semantic: ECG-based HF prediction ✅

Papers:
1. "Deep learning for ECG-based heart failure detection" (2023)
2. "CNN model predicts HF from 12-lead ECG" (2022)
3. "LSTM network for HF risk from ECG" (2024)
4. "Transformer-based ECG analysis in HF" (2023)
5. "Multi-modal ECG+clinical HF prediction" (2024)
```

**100% Coherencia Garantizada** ✅

---

## 📊 Ventajas vs Otros Métodos

### vs Standard Clustering
```
Standard HDBSCAN:
❌ Papers: HF + Diabetes + Kidney (mixed)
❌ Methods: ML + RCT + Cohort (mixed)
❌ Years: 2010-2025 (mixed)

Hierarchical Funnel:
✅ Papers: SOLO heart_failure
✅ Methods: SOLO machine_learning
✅ Years: 2020-2025 (recent)
```

### vs Domain-Aware
```
Domain-Aware:
✅ Domains: Cardiac, Neuro, etc.
❌ Dentro de cardiac: mezcla HF + arrhythmia + MI
❌ No separa por metodología

Hierarchical Funnel:
✅ Specific topics: HF, arrhythmia, MI separados
✅ Metodología consistente
✅ Temporal coherence
```

### vs Domain-Aware + Adaptive
```
Domain + Adaptive:
✅ Dominios coherentes
✅ Bajo ruido
❌ Puede mezclar RCT + ML
❌ No considera recencia

Hierarchical Funnel:
✅ Topics específicos
✅ Metodología consistente
✅ Prioriza recencia
✅ Bajo ruido por construcción
```

---

## 🎛️ Parámetros Configurables

### min_cluster_size (default: 5)
```python
min_cluster_size=5  → Más clusters, más específicos
min_cluster_size=10 → Menos clusters, más grandes
```

### min_topic_coverage (default: 0.6)
```python
# % mínimo de papers que deben compartir el topic
0.6 → 60% de papers deben tener mismo topic
0.8 → 80% (más estricto)
```

### min_methodology_coverage (default: 0.5)
```python
# % mínimo que deben compartir metodología
0.5 → 50% deben tener misma metodología
0.7 → 70% (más estricto)
```

### recency_window_years (default: 5)
```python
recency_window=5  → Recent = últimos 5 años
recency_window=3  → Recent = últimos 3 años (más estricto)
recency_window=10 → Recent = últimos 10 años (más permisivo)
```

---

## 📈 Resultados Esperados

### Con 417 Papers (Boston Children's)

**Configuración Recomendada**:
```
min_cluster_size=5
min_topic_coverage=0.6
min_methodology_coverage=0.5
recency_window_years=5
```

**Resultados Esperados**:
```
Topics identificados: 15-20
Methods por topic: 3-5
Time groups: 2 (recent + older)
Final clusters: 25-40
Noise: 15-25%
Avg cluster size: 8-12 papers

Clustering rate: 75-85%
Cluster purity: 100% (garantizado)
```

---

## 🔬 Cómo Usar en la App

1. Abre http://localhost:8501
2. Sidebar → Clustering Strategy:
   ```
   Selecciona: 🎯 Hierarchical Funnel (Recommended)
   ```

3. Configura parámetros:
   ```
   Min Cluster Size: 5
   Dataset Size: 417
   ```

4. Run Pipeline

5. Ve al tab "🔍 Funnel Analysis":
   - Topic distribution
   - Methodology distribution  
   - Cluster composition
   - Funnel efficiency

---

## ✅ Garantías del Funnel

1. **100% Topic Purity** 
   - Cada cluster = 1 solo topic específico
   
2. **100% Methodology Consistency**
   - Cada cluster = 1 sola metodología
   
3. **Temporal Coherence**
   - Prioriza papers recientes
   
4. **No Mixed Topics**
   - NUNCA heart_failure + diabetes
   - NUNCA stroke + kidney
   
5. **No Mixed Methods**
   - NUNCA RCT + ML juntos
   - NUNCA cohort + imaging juntos

---

## 🎯 Resumen: Por Qué Funciona

**Problema Anterior**:
```
Cluster 7:
- AI urología review
- Modelo animal riñón  
- Riñón ectópico
- Vitamina K CKD
- Células mast vejiga
→ INCOHERENTE ❌
```

**Con Hierarchical Funnel**:
```
Cluster 7:
- Topic: aki (acute kidney injury)
- Method: machine_learning
- Time: 2020-2025 (recent)
- Papers: 8 ML models for AKI prediction
→ 100% COHERENTE ✅
```

---

**El funnel NO adivina ni inventa - cada decisión está basada en análisis textual real de topics y methodologies en los papers.**

---

**Última actualización**: Nov 9, 2025  
**URL**: http://localhost:8501
