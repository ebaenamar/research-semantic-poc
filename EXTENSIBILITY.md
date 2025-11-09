# Sistema Extensible: Criterios Personalizados y Embeddings

## Resumen de Mejoras

Este documento describe dos mejoras críticas al sistema:

1. **Sistema modular de criterios personalizados** - Permite añadir criterios de validación específicos
2. **Evaluación de embeddings** - Compara modelos para encontrar el mejor para papers científicos

---

## 1. Sistema de Criterios Personalizados

### Problema Identificado

El sistema original tenía criterios de validación fijos:
- Coherencia metodológica
- Framework conceptual
- Coherencia temporal
- Consistencia interna
- Términos MeSH

**Pero faltaban criterios específicos como:**
- ¿Los clinical trials tienen información de sponsors?
- ¿Hay datos disponibles para verificar hipótesis?
- ¿Es investigación original o replicación?
- ¿Hay diversidad geográfica?
- **Cualquier criterio personalizado que necesites**

### Solución: Sistema Modular

**Archivo**: `src/extraction/custom_criteria.py`

#### Arquitectura

```python
ValidationCriterion (Abstract Base Class)
    ├── ClinicalTrialSponsorCriterion
    ├── DataAvailabilityCriterion
    ├── ReplicationStatusCriterion
    ├── GeographicDiversityCriterion (ejemplo)
    └── TuCriterioPersonalizado (¡crea el tuyo!)
```

#### Criterios Incluidos

##### 1. ClinicalTrialSponsorCriterion
**Propósito**: Identificar si clinical trials tienen información de sponsors

**Detecta**:
- Si es un cluster de clinical trials
- Sponsors industriales vs académicos
- Cobertura de información de funding

**Ejemplo de uso**:
```python
from extraction.custom_criteria import ClinicalTrialSponsorCriterion

criterion = ClinicalTrialSponsorCriterion(weight=0.2)
result = criterion.evaluate(cluster_df)

# Resultado:
{
    'score': 0.8,
    'details': {
        'is_clinical_trial_cluster': True,
        'funding_type': 'primarily_industry',
        'sponsor_coverage': 0.75
    },
    'interpretation': 'Strong sponsor information. Clinical trials appear primarily industry sponsored.'
}
```

**Utilidad**:
- Identificar sesgos de financiamiento
- Distinguir investigación académica vs industrial
- Evaluar transparencia en reporting

##### 2. DataAvailabilityCriterion
**Propósito**: Verificar si los papers mencionan disponibilidad de datos

**Detecta**:
- Menciones de "data available", "open data", "github", etc.
- Tasa de disponibilidad de datos en el cluster

**Utilidad**:
- **Crítico para verificación de hipótesis**
- Identificar clusters con datos reproducibles
- Priorizar clusters donde hipótesis son testables

##### 3. ReplicationStatusCriterion
**Propósito**: Identificar si es investigación original o replicación

**Detecta**:
- Keywords de replicación vs originalidad
- Tipo de investigación dominante

**Utilidad**:
- Entender madurez del campo
- Identificar áreas que necesitan replicación
- Valorar estudios de validación

---

### Cómo Crear Tu Propio Criterio

#### Paso 1: Definir la Clase

```python
from extraction.custom_criteria import ValidationCriterion

class MiCriterioPersonalizado(ValidationCriterion):
    """
    Descripción de qué valida tu criterio
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__("mi_criterio", weight)
        
        # Define tus keywords o patrones
        self.keywords = ['keyword1', 'keyword2']
    
    def evaluate(self, cluster_df, text_column='abstract'):
        """
        Implementa tu lógica de evaluación
        
        DEBE retornar dict con:
            - score: float (0-1)
            - details: dict con datos
            - interpretation: str explicando el score
        """
        
        # Tu lógica aquí
        all_text = ' '.join(
            cluster_df[text_column].dropna().astype(str).str.lower()
        )
        
        # Ejemplo: contar menciones
        mentions = sum(1 for kw in self.keywords if kw in all_text)
        
        # Calcular score (0-1)
        score = min(mentions / 10, 1.0)
        
        return {
            'score': score,
            'details': {
                'mentions': mentions,
                'n_papers': len(cluster_df)
            },
            'interpretation': f"Found {mentions} mentions"
        }
```

#### Paso 2: Usar el Criterio

```python
from extraction.custom_criteria import CustomCriteriaValidator

# Crear validador
validator = CustomCriteriaValidator()

# Añadir criterios (built-in + custom)
validator.add_criterion(ClinicalTrialSponsorCriterion(weight=0.2))
validator.add_criterion(MiCriterioPersonalizado(weight=0.15))

# Evaluar clusters
results = validator.evaluate_all_clusters(df, labels)
```

#### Paso 3: Interpretar Resultados

```python
for cluster_name, cluster_results in results['custom_criteria_results'].items():
    print(f"{cluster_name}: Score {cluster_results['overall_custom_score']:.2f}")
    
    for criterion_name, criterion_result in cluster_results['criteria_results'].items():
        print(f"  {criterion_name}: {criterion_result['score']:.2f}")
        print(f"  {criterion_result['interpretation']}")
```

---

### Ejemplos de Criterios Personalizados

#### Ejemplo 1: Detectar Conflictos de Interés

```python
class ConflictOfInterestCriterion(ValidationCriterion):
    def __init__(self, weight=0.15):
        super().__init__("conflict_of_interest", weight)
        
        self.coi_keywords = [
            'conflict of interest', 'competing interest',
            'financial disclosure', 'no conflicts'
        ]
    
    def evaluate(self, cluster_df, text_column='abstract'):
        all_text = ' '.join(cluster_df[text_column].dropna().astype(str).str.lower())
        
        coi_mentions = sum(1 for kw in self.coi_keywords if kw in all_text)
        disclosure_rate = coi_mentions / len(cluster_df)
        
        # High disclosure = good
        score = min(disclosure_rate * 2, 1.0)
        
        return {
            'score': score,
            'details': {'disclosure_rate': disclosure_rate},
            'interpretation': f"COI disclosure rate: {disclosure_rate:.1%}"
        }
```

#### Ejemplo 2: Evaluar Tamaño de Muestra

```python
class SampleSizeCriterion(ValidationCriterion):
    def __init__(self, weight=0.1):
        super().__init__("sample_size_adequacy", weight)
    
    def evaluate(self, cluster_df, text_column='abstract'):
        all_text = ' '.join(cluster_df[text_column].dropna().astype(str))
        
        # Extract numbers that might be sample sizes
        import re
        numbers = re.findall(r'n\s*=\s*(\d+)', all_text.lower())
        
        if numbers:
            sample_sizes = [int(n) for n in numbers]
            avg_size = np.mean(sample_sizes)
            
            # Score based on adequacy
            if avg_size >= 1000:
                score = 1.0
            elif avg_size >= 100:
                score = 0.8
            else:
                score = 0.6
            
            interpretation = f"Average sample size: {avg_size:.0f}"
        else:
            score = 0.5
            interpretation = "No clear sample size information"
        
        return {
            'score': score,
            'details': {'sample_sizes': sample_sizes if numbers else []},
            'interpretation': interpretation
        }
```

#### Ejemplo 3: Detectar Uso de Estadística Bayesiana

```python
class BayesianMethodsCriterion(ValidationCriterion):
    def __init__(self, weight=0.1):
        super().__init__("bayesian_methods", weight)
        
        self.bayesian_keywords = [
            'bayesian', 'posterior', 'prior', 'mcmc',
            'credible interval', 'bayes factor'
        ]
    
    def evaluate(self, cluster_df, text_column='abstract'):
        all_text = ' '.join(cluster_df[text_column].dropna().astype(str).str.lower())
        
        bayesian_mentions = sum(1 for kw in self.bayesian_keywords if kw in all_text)
        usage_rate = bayesian_mentions / len(cluster_df)
        
        score = min(usage_rate * 3, 1.0)
        
        if usage_rate > 0.3:
            interpretation = "High Bayesian methods usage"
        elif usage_rate > 0.1:
            interpretation = "Moderate Bayesian methods usage"
        else:
            interpretation = "Limited Bayesian methods usage"
        
        return {
            'score': score,
            'details': {'usage_rate': usage_rate},
            'interpretation': interpretation
        }
```

---

## 2. Evaluación de Embeddings

### Problema: ¿Es all-MiniLM-L6-v2 Adecuado?

**Pregunta crítica**: ¿El modelo de embedding actual es óptimo para similitud semántica en papers científicos?

### Solución: Script de Evaluación

**Archivo**: `scripts/evaluate_embeddings.py`

#### Modelos Comparados

| Modelo | Dimensión | Especialización | Velocidad |
|--------|-----------|-----------------|-----------|
| **all-MiniLM-L6-v2** | 384 | General | ⚡⚡⚡ Muy rápido |
| **allenai/specter** | 768 | **Papers científicos** | ⚡ Lento |
| **all-mpnet-base-v2** | 768 | General (alta calidad) | ⚡⚡ Moderado |

#### Métricas de Evaluación

1. **Discriminative Score** (0-1)
   - Mide: ¿Puede distinguir papers diferentes?
   - Cálculo: `1 - mean_similarity`
   - **Alto = mejor** (papers son distinguibles)

2. **Similarity Distribution**
   - Mean, std, min, max de similitudes
   - Ideal: Mean bajo (~0.2), std moderado

3. **Speed** (papers/second)
   - Importante para datasets grandes

4. **Top Similar Pairs**
   - Validación cualitativa: ¿Los pares más similares tienen sentido?

#### Cómo Ejecutar

```bash
source venv/bin/activate
python scripts/evaluate_embeddings.py
```

#### Resultados Esperados

```
COMPARISON SUMMARY
======================================================================
Model                                    Dim      Speed        Discrim
----------------------------------------------------------------------
all-MiniLM-L6-v2                         384      150.0        0.799
allenai/specter                          768      45.0         0.850
all-mpnet-base-v2                        768      80.0         0.820
```

### Recomendaciones

#### Para Producción: allenai/specter ⭐
**Por qué**:
- Entrenado específicamente en papers científicos
- Mejor captura de semántica científica
- Mayor discriminative score
- Entiende jerga técnica y conceptos

**Cuándo usar**:
- Pipeline final de producción
- Cuando precisión es crítica
- Clustering de papers científicos
- Análisis de similitud semántica profunda

**Trade-off**: 2-3x más lento

#### Para Prototipado: all-MiniLM-L6-v2 ⚡
**Por qué**:
- Muy rápido (3-5x más rápido)
- Embeddings de calidad aceptable
- Bueno para propósito general

**Cuándo usar**:
- Exploración inicial
- Testing rápido
- Cuando velocidad importa
- Datasets muy grandes (>50k papers)

#### Para Balance: all-mpnet-base-v2 ⚖️
**Por qué**:
- Alta calidad general
- Velocidad moderada
- Buen compromiso

**Cuándo usar**:
- Cuando specter es muy lento
- Necesitas mejor calidad que MiniLM
- Dominio no es puramente científico

---

### Cómo Cambiar el Modelo

#### Opción 1: En el Código

```python
from embeddings import PaperEmbedder

# Cambiar modelo
embedder = PaperEmbedder(model_name='allenai/specter')
embeddings = embedder.embed_papers(df)
```

#### Opción 2: En Pipeline

```bash
# Editar config/pipeline_config.yaml
embeddings:
  model_name: "allenai/specter"  # Cambiar aquí
```

#### Opción 3: Argumento CLI

```bash
python scripts/run_full_pipeline.py --model allenai/specter
```

---

## Integración: Criterios + Embeddings

### Workflow Completo Mejorado

```python
# 1. Usar mejor embedding para científicos
embedder = PaperEmbedder(model_name='allenai/specter')
embeddings = embedder.embed_papers(df)

# 2. Clustering
clusterer = SemanticClusterer()
labels = clusterer.cluster(embeddings)

# 3. Validación estándar
from extraction import ClassificationValidator
validator = ClassificationValidator()
standard_validation = validator.validate_all_clusters(df, labels)

# 4. Validación con criterios personalizados
from extraction.custom_criteria import CustomCriteriaValidator
custom_validator = CustomCriteriaValidator()

# Añadir criterios relevantes
custom_validator.add_criterion(ClinicalTrialSponsorCriterion(weight=0.2))
custom_validator.add_criterion(DataAvailabilityCriterion(weight=0.15))
custom_validator.add_criterion(MiCriterioPersonalizado(weight=0.1))

custom_validation = custom_validator.evaluate_all_clusters(df, labels)

# 5. Combinar resultados
combined_score = (
    standard_validation['pass_rate'] * 0.7 +
    custom_validation['overall_score'] * 0.3
)
```

---

## Casos de Uso Específicos

### Caso 1: Análisis de Clinical Trials

```python
# Setup específico para trials
validator = CustomCriteriaValidator()
validator.add_criterion(ClinicalTrialSponsorCriterion(weight=0.3))
validator.add_criterion(DataAvailabilityCriterion(weight=0.2))
validator.add_criterion(SampleSizeCriterion(weight=0.2))
validator.add_criterion(ConflictOfInterestCriterion(weight=0.15))

# Usar SPECTER para mejor comprensión clínica
embedder = PaperEmbedder(model_name='allenai/specter')
```

### Caso 2: Meta-Análisis de Reproducibilidad

```python
validator = CustomCriteriaValidator()
validator.add_criterion(ReplicationStatusCriterion(weight=0.3))
validator.add_criterion(DataAvailabilityCriterion(weight=0.3))
validator.add_criterion(SampleSizeCriterion(weight=0.2))
```

### Caso 3: Análisis de Diversidad en Investigación

```python
validator = CustomCriteriaValidator()
validator.add_criterion(GeographicDiversityCriterion(weight=0.25))
validator.add_criterion(InstitutionalDiversityCriterion(weight=0.25))
validator.add_criterion(FundingDiversityCriterion(weight=0.25))
```

---

## Testing

### Test Criterios Personalizados

```bash
source venv/bin/activate
python scripts/test_custom_criteria.py
```

**Output esperado**:
- Evaluación de 4 criterios en cada cluster
- Scores individuales y combinados
- Interpretaciones específicas
- Archivo JSON con resultados detallados

### Test Embeddings

```bash
python scripts/evaluate_embeddings.py
```

**Output esperado**:
- Comparación de 3 modelos
- Métricas de calidad y velocidad
- Recomendaciones específicas
- Top pares similares para validación

---

## Archivos Clave

```
src/extraction/
├── custom_criteria.py          # Sistema modular de criterios
│   ├── ValidationCriterion     # Clase base abstracta
│   ├── ClinicalTrialSponsorCriterion
│   ├── DataAvailabilityCriterion
│   ├── ReplicationStatusCriterion
│   └── CustomCriteriaValidator # Validador extensible

scripts/
├── test_custom_criteria.py     # Demo de criterios personalizados
├── evaluate_embeddings.py      # Comparación de modelos
└── run_full_pipeline.py        # Pipeline con opciones

config/
└── pipeline_config.yaml        # Configuración (incluye modelo)
```

---

## Conclusión

### Mejoras Implementadas

✅ **Sistema Modular**: Añade criterios sin modificar código core
✅ **Criterios Built-in**: Clinical trials, data availability, replication
✅ **Extensible**: Crea tus propios criterios fácilmente
✅ **Evaluación de Embeddings**: Compara modelos científicamente
✅ **Recomendaciones**: Guía clara sobre qué modelo usar

### Próximos Pasos

1. **Ejecutar evaluación de embeddings**
   ```bash
   python scripts/evaluate_embeddings.py
   ```

2. **Decidir modelo** basado en resultados y prioridades

3. **Crear criterios personalizados** para tu caso de uso

4. **Ejecutar pipeline completo** con configuración optimizada

5. **Iterar** basado en resultados de validación

---

## Preguntas Frecuentes

**Q: ¿Puedo combinar validación estándar y custom?**
A: Sí, son complementarias. Usa ambas y combina scores.

**Q: ¿Cuántos criterios custom debo añadir?**
A: 3-5 criterios relevantes. Más puede diluir señal.

**Q: ¿Debo cambiar a SPECTER?**
A: Sí para producción. No para prototipado rápido.

**Q: ¿Los criterios funcionan con cualquier embedding?**
A: Sí, son independientes del modelo de embedding.

**Q: ¿Puedo usar criterios en pipeline automático?**
A: Sí, integra en `run_full_pipeline.py` o crea tu script.
