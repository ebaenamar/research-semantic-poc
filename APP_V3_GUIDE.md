# App V3 - Complete Guide
## PubMed Enrichment + Meta-Analysis Validation

---

## 🚀 Quick Start

```bash
# App V1 (Original, Hierarchical Funnel)
http://localhost:8501

# App V2 (Simple Clustering, Detailed Hypotheses)
http://localhost:8502

# App V3 (V2 + PubMed Enrichment + Validation)
http://localhost:8503  ← NEW!
```

---

## 🆕 What's New in V3

### **1. PubMed Metadata Enrichment** (Optional)
```
Toggle: "Enrich papers with PubMed metadata"

What it does:
- Fetches journal, year, MeSH terms, publication types
- First run: ~10 seconds (API calls)
- Subsequent runs: <0.1 seconds (cached)
- Cache: output/cache/pubmed/*.json

What you see:
- 📖 Journal: Scientific Reports
- 📅 Year: 2022
- 🏷️ MeSH: Machine Learning, Brain, Atrophy...
- 📄 Type: Journal Article, Research Support
```

### **2. Automatic Meta-Analysis Validation** (Default ON)
```
Toggle: "Validate Meta-Analysis suitability"

Checks:
✓ Not too many reviews (<30%)
✓ Not too many methods papers (<20%)
✓ Homogeneous topics (MeSH overlap ≥20%)
✓ Minimum 5 papers with metadata

Result:
- Valid → Generate meta-analysis hypothesis
- Invalid → Skip + show rejection reason
```

### **3. Smart Rejections Display**
```
When meta-analysis is rejected, you see:

⚠️ "1 Meta-Analysis hypotheses rejected"

Expandable details:
- Cluster ID
- Rejection reason: "Too many reviews (2/5)"
- Recommendation: "Papers too heterogeneous..."
```

---

## 📊 Comparison: V1 vs V2 vs V3

| Feature | V1 | V2 | V3 |
|---------|----|----|-----|
| **Clustering** | Hierarchical Funnel (4 stages) | Simple HDBSCAN | Simple HDBSCAN |
| **Hypothesis Detail** | Basic | Detailed with PMIDs | Detailed + Enriched |
| **PubMed Enrichment** | ❌ | ❌ | ✅ (optional) |
| **Meta-Analysis Validation** | ❌ | ❌ | ✅ (default ON) |
| **MeSH Terms** | ❌ | ❌ | ✅ |
| **Journal Info** | ❌ | ❌ | ✅ |
| **Publication Types** | ❌ | ❌ | ✅ |
| **Rejection Feedback** | ❌ | ❌ | ✅ |
| **Caching** | ❌ | ❌ | ✅ (30 days) |
| **Port** | 8501 | 8502 | 8503 |

---

## 🎯 Use Cases

### **Use V1 when:**
- You want strict, multi-stage filtering
- You need domain-aware clustering
- You want funnel analysis visualizations

### **Use V2 when:**
- You want quick, simple clustering
- You need detailed hypothesis info
- You don't need external API calls

### **Use V3 when:**
- You want complete paper metadata
- You need to validate meta-analysis suitability
- You want to see MeSH terms and journal info
- You want to avoid heterogeneous meta-analyses

---

## 🔬 Testing Workflow

### **Test 1: Without Enrichment**
1. Open http://localhost:8503
2. Uncheck "Enrich papers with PubMed metadata"
3. Keep "Validate Meta-Analysis" OFF
4. Generate hypotheses
5. **Result**: Same as V2 (fast, no API calls)

### **Test 2: With Validation Only**
1. Uncheck "Enrich papers"
2. Check "Validate Meta-Analysis" ON
3. Generate hypotheses
4. **Result**: 
   - Fetches metadata for validation (~10s first time)
   - Rejects invalid meta-analyses
   - Shows rejection reasons

### **Test 3: Full Enrichment**
1. Check "Enrich papers with PubMed metadata" ON
2. Check "Validate Meta-Analysis" ON
3. Generate hypotheses
4. **Result**:
   - First run: ~10-15 seconds
   - Papers show: journal, MeSH, pub types
   - Invalid meta-analyses rejected
   - Cache created for next time

### **Test 4: Cache Performance**
1. Run Test 3 again (same settings)
2. **Result**: 
   - Same results in <1 second
   - All data from cache
   - Papers show "✅ Metadata from cache"

---

## 📋 Example Output Comparison

### **V2 Output (Paper Display)**:
```
Paper 1:
Title: Machine learning-based automatic estimation...
PMID: 36042322
Year: 2022.0
Abstract: Cortical atrophy is measured...
```

### **V3 Output (With Enrichment)**:
```
Paper 1:
Title: Machine learning-based automatic estimation of cortical 
       atrophy using brain computed tomography images.
PMID: 36042322
📖 Journal: Scientific Reports
📅 Year: 2022
📄 Type: Journal Article, Research Support, Non-U.S. Gov't
🏷️ MeSH: Alzheimer Disease, Atrophy, Brain, Humans, Machine Learning (+3 more)

[View Abstract ▼]
✅ Metadata from cache
```

---

## ⚠️ Meta-Analysis Rejection Examples

### **Example 1: Too Many Reviews**
```
Cluster 5: Meta-Analysis rejected
Reason: Too many reviews (2/5 = 40%)
Recommendation: Papers too heterogeneous for meta-analysis

What happened:
- Validation detected 40% reviews (threshold: <30%)
- System skipped meta-analysis generation
- Generated other hypothesis types instead
```

### **Example 2: Low MeSH Overlap**
```
Cluster 8: Meta-Analysis rejected
Reason: Low MeSH overlap (avg coverage: 15.2%)
Recommendation: Papers too heterogeneous for meta-analysis

What happened:
- Only 15.2% MeSH overlap (threshold: ≥20%)
- Topics too mixed: neuro + onco + biophysics
- System skipped meta-analysis generation
```

### **Example 3: Valid Meta-Analysis**
```
Cluster 3: Meta-Analysis generated ✅
- 0 reviews (0%)
- MeSH overlap: 68%
- Homogeneous topic: diagnostic imaging
- All primary research articles
```

---

## 🎛️ Configuration Guide

### **Sidebar: PubMed Enrichment Section**

```python
🔬 PubMed Enrichment
*NEW in V3*

☐ Enrich papers with PubMed metadata
  Help: Fetches journal, MeSH terms, publication types. 
        First run slow (~10s), then cached.

⚡ First enrichment: ~10s. Subsequent runs: <0.1s (cached)

NCBI API Key (optional): [password field]
  Help: For faster enrichment (10 req/s vs 3 req/s)
        Get at: https://www.ncbi.nlm.nih.gov/account/settings/

☑ Validate Meta-Analysis suitability
  Help: Checks: not too many reviews, homogeneous topics 
        (MeSH overlap ≥20%)

✓ Will reject heterogeneous clusters for meta-analysis
```

### **Recommended Settings**

**For Quick Testing:**
```
Enrich: OFF
Validate: OFF
→ Same as V2, instant results
```

**For First-Time Enrichment:**
```
Enrich: ON
Validate: ON
Dataset Size: 850
→ ~10-15s first run, builds cache
```

**For Production Use:**
```
Enrich: ON
Validate: ON
Dataset Size: 2000
NCBI API Key: [your key]
→ Fast with cache, comprehensive validation
```

---

## 🗂️ Cache Management

### **Cache Location**
```bash
output/cache/pubmed/
├── 36042322.json  (2.0K)
├── 39792693.json  (2.5K)
├── 33894656.json  (1.8K)
└── ...
```

### **Cache Info**
```
Expiry: 30 days
Format: JSON
Fields: pmid, title, journal, year, mesh_terms, 
        publication_types, abstract, url, _fetched_at
```

### **Clear Cache**
```bash
# Delete all cached papers
rm -rf output/cache/pubmed/*.json

# Delete single paper cache
rm output/cache/pubmed/36042322.json
```

### **Force Refresh**
```bash
# Option 1: Delete cache
rm -rf output/cache/pubmed/*.json

# Option 2: Cache auto-expires after 30 days
# Option 3: Modify paper → refetch automatically
```

---

## 📊 Performance Benchmarks

### **Without Enrichment**
```
Dataset Size: 850 papers
Time: ~45 seconds
- Loading: 2s
- Embeddings: 30s
- Clustering: 8s
- Hypotheses: 5s
```

### **With Enrichment (First Run)**
```
Dataset Size: 850 papers
Time: ~55 seconds (+10s)
- Loading: 2s
- Embeddings: 30s
- Clustering: 8s
- Hypotheses: 5s
- PubMed Enrichment: 10s ← NEW
```

### **With Enrichment (Cached)**
```
Dataset Size: 850 papers
Time: ~45 seconds (no overhead!)
- Loading: 2s
- Embeddings: 30s
- Clustering: 8s
- Hypotheses: 5s
- PubMed Enrichment: <0.1s ← FROM CACHE
```

---

## 🔍 Validation Logic

### **Meta-Analysis Validation Steps**

```python
1. Check: Minimum papers
   → Requires ≥5 papers with metadata
   → If < 5: REJECT

2. Check: Review ratio
   → Count papers with "Review" in publication types
   → If > 30%: REJECT ("Too many reviews")

3. Check: Methods papers ratio
   → Count papers with "Methods"/"Protocol" in types
   → If > 20%: REJECT ("Too many methods papers")

4. Check: Topic homogeneity
   → Calculate MeSH term overlap across papers
   → Average coverage = (paper_mesh ∩ all_mesh) / all_mesh
   → If < 20%: REJECT ("Low MeSH overlap")

5. All checks passed
   → VALID: Generate meta-analysis hypothesis
```

### **Example Validation Results**

```python
# Valid Cluster
{
    'valid': True,
    'n_papers': 12,
    'review_count': 1,
    'method_count': 0,
    'mesh_coverage': 0.68,
    'recommendation': 'Papers appear suitable for meta-analysis'
}

# Invalid Cluster (Too Many Reviews)
{
    'valid': False,
    'reason': 'Too many reviews (5/12 = 42%)',
    'n_papers': 12,
    'review_count': 5,
    'recommendation': 'Papers too heterogeneous for meta-analysis'
}

# Invalid Cluster (Heterogeneous)
{
    'valid': False,
    'reason': 'Low MeSH overlap (avg coverage: 15.2%)',
    'mesh_coverage': 0.152,
    'recommendation': 'Papers too heterogeneous for meta-analysis'
}
```

---

## 💡 Tips & Tricks

### **Tip 1: Start Without Enrichment**
```
First run: Enrich OFF
→ See basic results fast
→ Identify interesting clusters

Second run: Enrich ON (same clusters)
→ See full metadata
→ Validation kicks in
```

### **Tip 2: Use API Key for Large Datasets**
```
Without key: 3 requests/second
With key: 10 requests/second

Dataset 2000 papers:
- Without key: ~3-4 minutes first run
- With key: ~1 minute first run
- Cached: <1 second (both)
```

### **Tip 3: Monitor Rejections**
```
If many meta-analyses rejected:
→ Clusters are heterogeneous
→ Try:
  • Increase min_cluster_size (more focused clusters)
  • Adjust UMAP components
  • Use different embedding model
```

### **Tip 4: Inspect Cache**
```bash
# View cached paper
cat output/cache/pubmed/36042322.json | jq .

# Count cached papers
ls output/cache/pubmed/*.json | wc -l

# Find papers with specific MeSH
grep -l "Machine Learning" output/cache/pubmed/*.json
```

---

## 🚨 Troubleshooting

### **Problem: "No module named 'external'"**
**Solution:**
```bash
# Ensure src/external/ exists
ls src/external/pubmed_client.py

# Check __init__.py
cat src/external/__init__.py
```

### **Problem: "PubMed API timeout"**
**Solution:**
```
1. Check internet connection
2. Try with API key (faster)
3. Reduce dataset size initially
4. Cached papers work offline
```

### **Problem: "All meta-analyses rejected"**
**Solution:**
```
This is expected if:
- Clusters are heterogeneous (mixed topics)
- Many reviews in dataset
- Low MeSH overlap

Try:
- Uncheck "Validate Meta-Analysis" to see hypotheses anyway
- Increase min_cluster_size for more focused clusters
- Check rejection reasons in expandable section
```

### **Problem: "Slow first enrichment"**
**Solution:**
```
Normal! PubMed API has rate limits:
- Without key: 3 req/s
- With key: 10 req/s

For 50 papers:
- Without key: ~17 seconds
- With key: ~5 seconds

Subsequent runs: <0.1s (cached)
```

---

## 📈 Expected Results

### **Typical Run (850 papers, enrichment ON, first time)**

```
Pipeline Summary:
- Computational Papers: 650
- Clusters Found: 8
- Reproducible Clusters: 5
- Hypotheses Generated: 15

⚠️ 2 Meta-Analysis hypotheses rejected

Hypothesis Types:
- ML Application: 5
- Meta-Analysis: 3 (2 rejected, 1 valid)
- Replication: 5
- Cross-Cluster: 2

Top Hypothesis:
#1: Improve CNN models on TCGA dataset (Priority: 9.2)
  Type: ML Application
  Reproducibility: HIGH
  Time: 2-4 weeks
  
  Papers include:
  - PMID 36042322
    📖 Journal: Scientific Reports
    🏷️ MeSH: Machine Learning, Brain, Atrophy...
    ✅ Metadata from cache
```

---

## 🎉 Summary

**V3 = V2 + Smart Enhancements**

✅ All V2 features (detailed hypotheses, PMIDs, verification plans)
✅ PubMed metadata enrichment (journal, MeSH, pub types)
✅ Automatic meta-analysis validation
✅ Intelligent caching (30-day expiry)
✅ Rejection feedback with reasons
✅ Optional API key for speed
✅ Offline-capable (after initial fetch)

**Perfect for:**
- Researchers wanting complete paper metadata
- Avoiding invalid meta-analyses
- Understanding why clusters are rejected
- Building a local metadata cache

---

## 🔗 Related Files

- **App V3 Code**: `/app_v3.py`
- **PubMed Client**: `/src/external/pubmed_client.py`
- **Client Guide**: `/PUBMED_ENRICHMENT_GUIDE.md`
- **Test Script**: `/test_pubmed_enrichment.py`
- **Cache Directory**: `/output/cache/pubmed/`

---

**Ready to test? Open http://localhost:8503 and explore!** 🚀
