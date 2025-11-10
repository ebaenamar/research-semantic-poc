# Noise Reduction Guide

**Strategies to reduce clustering noise and find tight, coherent clusters**

---

## 🎯 Problem: High Noise Ratio (72.3%)

When you see high noise (>50%), it means:
- Clustering parameters too strict
- Papers don't form clear groups
- Min cluster size too large
- Need different clustering strategy

---

## ✅ Solution: Adaptive Clustering

New **🎯 Adaptive (Reduce Noise)** mode uses 3 strategies:

### Strategy 1: Hierarchical HDBSCAN
- Very lenient parameters (min_samples=2)
- Captures more papers in clusters
- Smaller minimum cluster size (3-5 papers)
- Uses "excess of mass" method

### Strategy 2: Adaptive DBSCAN
- Automatically determines epsilon (distance threshold)
- Uses 20th percentile of k-nearest distances
- Distance-based rather than density-based
- Good for finding tight groups

### Strategy 3: Future Work Focused
- Prioritizes papers mentioning:
  - "future work", "future research"
  - "limitation", "gap", "need for"
  - "further research", "warrant"
  - "remains to be", "unexplored"
- These papers more likely to form research opportunities

### Strategy 4: Rescue Noise Points
- Assigns noise points to nearest cluster
- Only if within 10th percentile of cluster distances
- Reduces noise without compromising quality

---

## 📊 Expected Results

### Before (Standard Clustering)
```
Papers: 848
Clusters: 8
Noise: 613 (72.3%)
```

### After (Adaptive Clustering)
```
Papers: 848
Clusters: 15-25 (more, smaller clusters)
Noise: <250 (<30%)
```

---

## 🎛️ How to Use

### In Web App

1. **Select Clustering Mode**: 🎯 Adaptive (Reduce Noise)
2. **Set Min Cluster Size**: 3-5 (smaller = more clusters)
3. **Run Pipeline**

### Parameters

**Min Cluster Size**:
- 3: Very small, tight clusters (more clusters, less noise)
- 5: Small clusters (balanced)
- 10: Medium clusters (fewer clusters)
- 15+: Large clusters (may increase noise)

**Recommendation**: Start with 5

---

## 🔍 What Makes a Good Cluster?

### Quality Metrics

1. **Cohesion** (0-1)
   - How tight is the cluster?
   - Higher = papers more similar to each other

2. **Future Work Score** (0-1)
   - How many papers mention future work?
   - Higher = more research opportunities

3. **Recency Score** (0-1)
   - How recent are the papers?
   - Higher = more relevant

4. **Overall Quality**
   - Weighted combination: 40% cohesion + 40% future + 20% recency

---

## 💡 Finding Research Opportunities

### What to Look For

**High-Quality Clusters** have:
- ✅ Tight cohesion (similar papers)
- ✅ Multiple future work mentions
- ✅ Recent publications (2020+)
- ✅ Clear common theme
- ✅ Identified gaps or limitations

**Example Good Cluster**:
```
Size: 7 papers
Cohesion: 0.85
Future Work Score: 0.71 (5 papers mention future work)
Recency: 0.88 (avg year: 2022)
Theme: "Machine learning for cardiac risk prediction"
Gap: "Limited external validation"
```

---

## 🎯 Comparison of Modes

### 🎯 Adaptive (Reduce Noise)
**Best for**: Finding research opportunities
**Pros**:
- Lowest noise (<30%)
- More clusters (15-25)
- Smaller, tighter groups
- Focuses on future work

**Cons**:
- More clusters to review
- Some very small clusters

**Use when**: You want maximum coverage and many hypotheses

---

### 🔬 Domain-Aware
**Best for**: Preventing topic mixing
**Pros**:
- No mixing of cardiac + brain
- Scientifically coherent
- Domain-specific clusters

**Cons**:
- May have moderate noise (40-50%)
- Fewer clusters per domain
- Some papers unclassified

**Use when**: Domain purity is critical

---

### ⚡ Standard
**Best for**: Quick exploration
**Pros**:
- Fast
- Simple
- Good for large datasets

**Cons**:
- Can have high noise (50-70%)
- May mix incompatible topics
- Fewer clusters

**Use when**: First pass or very large dataset

---

## 📈 Optimization Tips

### If Still Too Much Noise

1. **Lower min_cluster_size** to 3
2. **Try different embedding model** (allenai/specter for scientific)
3. **Filter dataset** (computational papers only)
4. **Increase dataset size** (more papers = better clusters)

### If Too Many Small Clusters

1. **Increase min_cluster_size** to 7-10
2. **Use Domain-Aware** instead
3. **Merge similar clusters** manually

### If Clusters Not Coherent

1. **Enable Domain-Aware** mode
2. **Check common themes** in hypothesis descriptions
3. **Review papers** in cluster manually
4. **Adjust validation criteria**

---

## 🔬 Technical Details

### Hierarchical HDBSCAN Parameters
```python
min_cluster_size=5        # Smaller than standard (15)
min_samples=2             # Very lenient
cluster_selection_method='eom'  # Excess of mass
allow_single_cluster=False
```

### Adaptive DBSCAN
```python
# Epsilon calculated as:
k = min_samples
kth_distances = [distance to kth neighbor for each point]
epsilon = 20th percentile of kth_distances
```

### Rescue Strategy
```python
for noise_point in noise_points:
    nearest_cluster = find_nearest_cluster(noise_point)
    distance = distance_to_cluster_centroid(noise_point, nearest_cluster)
    threshold = 10th_percentile_of_within_cluster_distances(nearest_cluster)
    
    if distance <= threshold:
        assign_to_cluster(noise_point, nearest_cluster)
```

---

## 📊 Example Output

### Adaptive Clustering Log
```
==================================================================
ADAPTIVE CLUSTERING - NOISE REDUCTION
==================================================================
Target noise ratio: 30.0%
Min cluster size: 5

🔍 Strategy 1: Hierarchical HDBSCAN
   Found 18 clusters
   Noise ratio: 42.3%

🔍 Strategy 2: Adaptive DBSCAN
   Adaptive epsilon: 0.3421
   Found 22 clusters
   Noise ratio: 35.1%

🔍 Strategy 3: Future Work Focused
   Papers with future work mentions: 234
   Found 20 clusters
   Noise ratio: 38.7%

🚨 Noise still high (35.1%), applying rescue strategy...
   Attempting to rescue 297 noise points...
   Rescued 89 points (30.0%)
   Final noise ratio: 24.6%

==================================================================
FINAL RESULTS:
Clusters: 22
Noise: 208 papers (24.6%)
==================================================================
```

---

## ✅ Success Criteria

**Good Result**:
- Noise < 30%
- 15-30 clusters
- Average cluster size: 10-30 papers
- Most clusters have future work mentions
- Clear themes in hypotheses

**Excellent Result**:
- Noise < 20%
- 20-40 clusters
- High cohesion scores (>0.7)
- 70%+ clusters mention future work
- Actionable research opportunities

---

## 🎯 Recommended Workflow

1. **First Run**: Adaptive mode, min_size=5, 200 papers
2. **Review**: Check noise ratio and cluster quality
3. **Adjust**: If needed, tweak min_cluster_size
4. **Scale Up**: Once satisfied, try 500-1000 papers
5. **Refine**: Use Domain-Aware for domain-specific analysis

---

**Document Version**: 1.0
**Last Updated**: November 9, 2025
**Code**: `src/clustering/adaptive_clusterer.py`
