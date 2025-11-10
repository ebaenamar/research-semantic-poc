# PubMed Enrichment System - Guía Completa

## ✅ Implementado

**Archivo**: `src/external/pubmed_client.py`

Sistema completo de enriquecimiento de papers con PubMed API + caché local.

---

## 🎯 Características

### 1. **Caché Local Inteligente**
```python
Cache location: output/cache/pubmed/{pmid}.json
Expiry: 30 días
Behavior: 
  - Primera vez: llama a PubMed API
  - Siguientes veces: usa caché (instantáneo)
  - Después de 30 días: refresca desde API
```

### 2. **Rate Limiting Automático**
```python
Sin API key: 3 requests/segundo
Con API key: 10 requests/segundo

Auto-delay entre requests para cumplir límites
```

### 3. **Metadatos Enriquecidos**
```python
Campos extraídos:
- title: Título oficial
- journal: Nombre completo del journal
- year: Año de publicación
- doi: DOI del paper
- authors: Lista de autores
- publication_types: ['Journal Article', 'Clinical Trial', etc.]
- mesh_terms: MeSH headings oficiales
- abstract: Abstract completo
- url: Link directo a PubMed
```

### 4. **Validación para Meta-Análisis**
```python
Checks automáticos:
✓ No más de 30% reviews
✓ No más de 20% methods/protocols
✓ MeSH overlap ≥20% (homogeneidad)
✓ Mínimo 5 papers con metadata

Return: {'valid': True/False, 'reason': '...', 'recommendation': '...'}
```

---

## 🚀 Cómo Usar

### **Test Básico**

```bash
cd /Users/e.baena/CascadeProjects/research-semantic-poc
source venv/bin/activate
python -m src.external.pubmed_client
```

Esto probará con PMID 36042322 y mostrará:
```
Title: Machine learning-based automatic estimation...
Journal: ...
Year: 2022
MeSH Terms: Machine Learning, Tomography X-Ray Computed, ...
Publication Types: Journal Article
From cache: False  # Primera vez

# Segunda ejecución → From cache: True (instantáneo)
```

---

### **Uso en Código**

```python
from src.external import PubMedClient

# Inicializar cliente
client = PubMedClient()

# Enriquecer un solo paper
metadata = client.fetch_details('36042322')

print(metadata['title'])
print(metadata['mesh_terms'])
print(metadata['publication_types'])

# Enriquecer lista de papers
papers = [
    {'pmid': '36042322', 'title': '...'},
    {'pmid': '39792693', 'title': '...'},
    {'pmid': '33894656', 'title': '...'}
]

enriched = client.enrich_papers(papers, use_cache=True)

# Cada paper ahora tiene campos adicionales:
# journal, mesh_terms, publication_types, etc.

# Validar para meta-análisis
validation = client.validate_for_meta_analysis(enriched)

if validation['valid']:
    print("✅ Papers suitable for meta-analysis")
    print(f"MeSH coverage: {validation['mesh_coverage']:.1%}")
else:
    print(f"❌ Not suitable: {validation['reason']}")
    print(f"Recommendation: {validation['recommendation']}")
```

---

## 📊 Integración en App V2

### **Opción 1: Toggle en Sidebar** (Recomendado)

Añadir en `app_v2.py` sidebar:

```python
st.subheader("🔬 PubMed Enrichment")

enrich_pubmed = st.checkbox(
    "Enrich with PubMed metadata",
    value=False,
    help="Fetches MeSH terms, journal, pub types (cached, slow first time)"
)

if enrich_pubmed:
    st.caption("⚠️ First run may take 1-2 min. Subsequent runs use cache.")
```

### **Opción 2: En Pipeline** 

Modificar `generate_data_driven_hypotheses`:

```python
def generate_data_driven_hypotheses(...):
    # ... código existente ...
    
    # Enrich papers if enabled
    if config.get('enrich_pubmed', False):
        from src.external import PubMedClient
        client = PubMedClient()
        
        # Enrich sample papers
        details['sample_papers'] = client.enrich_papers(
            details['sample_papers'][:5],
            use_cache=True
        )
    
    # ... resto del código ...
```

### **Opción 3: Validación Meta-Análisis**

Mejorar Type 2 (Meta-Analysis):

```python
# Type 2: Meta-Analysis
if len(cluster_df) >= 10:
    # Validate suitability for meta-analysis
    if config.get('validate_meta_analysis', True):
        from src.external import PubMedClient
        client = PubMedClient()
        
        validation = client.validate_for_meta_analysis(
            details['sample_papers'][:10]
        )
        
        if not validation['valid']:
            # Skip meta-analysis hypothesis
            print(f"Skipping meta-analysis for cluster {cluster_id}: {validation['reason']}")
            continue  # Don't generate this hypothesis
    
    # ... generate meta-analysis hypothesis ...
```

---

## 🎨 UI Mejoras

### **Mostrar Metadata Enriquecida**

En App V2, actualizar sección "Key Papers":

```python
# Key Papers with PMIDs
if 'key_papers' in hyp and hyp['key_papers']:
    st.markdown("---")
    st.markdown("### 📚 Key Papers in Cluster")
    
    for i, paper in enumerate(hyp['key_papers'], 1):
        with st.container():
            st.markdown(f"**Paper {i}**")
            st.markdown(f"**Title:** {paper['title']}")
            
            if paper.get('pmid'):
                st.markdown(f"**PMID:** [{paper['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/)")
            
            # NUEVO: Metadata enriquecida
            if paper.get('journal'):
                st.caption(f"📖 Journal: {paper['journal']}")
            
            if paper.get('year'):
                st.caption(f"📅 Year: {paper['year']}")
            
            if paper.get('mesh_terms'):
                mesh_display = ', '.join(paper['mesh_terms'][:5])
                st.caption(f"🏷️ MeSH: {mesh_display}")
            
            if paper.get('publication_types'):
                pub_types = ', '.join(paper['publication_types'])
                st.caption(f"📄 Type: {pub_types}")
            
            if paper.get('abstract'):
                with st.expander("View Abstract"):
                    st.write(paper['abstract'])
            
            st.markdown("---")
```

---

## 📈 Resultados Esperados

### **Antes (sin enrichment)**:
```
Paper 1:
Title: Machine learning-based automatic estimation...
PMID: 36042322
Year: 2022.0
Abstract: Cortical atrophy is measured clinically...
```

### **Después (con enrichment)**:
```
Paper 1:
Title: Machine learning-based automatic estimation of cortical 
       atrophy using brain computed tomography images
PMID: 36042322
📖 Journal: Scientific Reports
📅 Year: 2022
🏷️ MeSH: Machine Learning, Tomography X-Ray Computed, Cerebral Cortex, 
         Brain, Atrophy
📄 Type: Journal Article

[View Abstract ▼]
  Cortical atrophy is measured clinically according to established 
  visual rating scales based on magnetic resonance imaging (MRI)...
```

---

## ⚡ Performance

### **Primera Ejecución** (sin caché):
```
10 papers: ~10 segundos (rate limited)
20 papers: ~20 segundos
50 papers: ~50 segundos
```

### **Ejecuciones Siguientes** (con caché):
```
10 papers: <0.1 segundos ✨
20 papers: <0.1 segundos ✨
50 papers: <0.1 segundos ✨
```

---

## 🔐 API Key (Opcional)

Para rate limits más altos (10 req/s):

```python
# Get API key: https://www.ncbi.nlm.nih.gov/account/settings/

client = PubMedClient(api_key='your_api_key_here')
```

En App V2:

```python
# Sidebar
api_key = st.text_input(
    "NCBI API Key (optional)",
    type="password",
    help="For faster enrichment (10 req/s vs 3 req/s)"
)

# In pipeline
if config.get('enrich_pubmed'):
    client = PubMedClient(api_key=api_key if api_key else None)
```

---

## 🛡️ Robustez

### **Manejo de Errores**:
- ✅ Network timeout (10-15s)
- ✅ Invalid PMID → skip silently
- ✅ API error → retry once, then skip
- ✅ Malformed XML → parse what's available
- ✅ Cache corruption → refetch from API

### **Fallbacks**:
- Si falla API → usa datos del CSV original
- Si falta PMID → intenta con DOI (futuro)
- Si caché corrupto → borra y refetch

---

## 🎯 Next Steps

### **Inmediato** (5 min):
```bash
# Test el cliente
python -m src.external.pubmed_client

# Ver caché creado
ls output/cache/pubmed/
cat output/cache/pubmed/36042322.json
```

### **Integración** (15 min):
1. Añadir checkbox en App V2 sidebar
2. Enriquecer `sample_papers` en pipeline
3. Actualizar UI para mostrar metadata

### **Validación Meta-Análisis** (30 min):
1. Añadir `validate_meta_analysis` check
2. Skip meta-analysis si `validation['valid'] == False`
3. Mostrar razón en UI si se skippea

---

## 📝 Ejemplo Completo

```python
from src.external import PubMedClient

# Init
client = PubMedClient()

# Test PMIDs del cluster problemático
pmids = ['36042322', '39792693', '33894656', '33343224', '35209064']

# Enrich
papers = [{'pmid': pmid} for pmid in pmids]
enriched = client.enrich_papers(papers)

# Mostrar MeSH de cada uno
for paper in enriched:
    print(f"\nPMID {paper['pmid']}:")
    print(f"  Title: {paper['title'][:60]}...")
    print(f"  Journal: {paper.get('journal', 'N/A')}")
    print(f"  MeSH: {', '.join(paper.get('mesh_terms', [])[:3])}")
    print(f"  Types: {', '.join(paper.get('publication_types', []))}")

# Validate for meta-analysis
validation = client.validate_for_meta_analysis(enriched)
print(f"\n✅ Valid for meta-analysis: {validation['valid']}")
print(f"Reason: {validation.get('reason', validation.get('recommendation'))}")
```

Expected output:
```
PMID 36042322:
  Title: Machine learning-based automatic estimation of cortical...
  Journal: Scientific Reports
  MeSH: Machine Learning, Tomography X-Ray Computed, Cerebral Cortex
  Types: Journal Article

PMID 39792693:
  Title: Artificial Intelligence for Predicting HER2 Status...
  Journal: ...
  MeSH: Artificial Intelligence, Stomach Neoplasms, Receptor ErbB-2
  Types: Journal Article

...

✅ Valid for meta-analysis: False
Reason: Low MeSH overlap (avg coverage: 15.2%)
```

---

## 🎉 Beneficios

1. **Referencias Completas** → PMIDs con journal, MeSH, year
2. **Validación Automática** → No más meta-análisis inválidos
3. **Caché Persistente** → Solo llama API una vez
4. **Filtrado Inteligente** → Excluye reviews/methods
5. **UX Mejorada** → Metadata rica en UI

---

**Estado**: ✅ Cliente implementado y listo para integrar
**Next**: Añadir toggle en App V2 y probar con cluster real
