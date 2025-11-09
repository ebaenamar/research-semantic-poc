# Web Interface - Quick Start Guide

**Interactive web application for hypothesis generation from scientific literature**

---

## 🚀 Quick Start (2 minutes)

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install Streamlit (if not already installed)
pip install streamlit>=1.28.0
```

### 2. Launch Web App

```bash
# From project root
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 🎯 Features

### ✅ Real Dataset
- Uses actual Boston Children's Hospital PubMed dataset
- No fake or synthetic data
- 2,000 real scientific papers

### ✅ Interactive Configuration
- **Dataset Size**: Choose 200, 500, 1000, or 2000 papers
- **Embedding Models**: 
  - `all-MiniLM-L6-v2` (fast, 384-dim)
  - `allenai/specter` (best for scientific papers, 768-dim)
  - `all-mpnet-base-v2` (balanced, 768-dim)
- **Clustering Parameters**: Adjust min_cluster_size, min_samples, UMAP components
- **Filters**: Computational papers only option

### ✅ Custom Validation Criteria
- Data Availability Criterion
- Clinical Trial Sponsor Criterion
- Replication Status Criterion
- All from existing `src/extraction/custom_criteria.py`

### ✅ Real-Time Pipeline
- Live progress tracking
- Stage-by-stage status updates
- Error handling

### ✅ Interactive Results
- **Clusters Tab**: 2D visualization, cluster sizes
- **Validation Tab**: Scores, pass/fail status
- **Hypotheses Tab**: Ranked hypotheses with details
- **Export Tab**: Download JSON and CSV results

---

## 📊 Example Workflow

### Step 1: Configure Pipeline

In the sidebar:
1. Select dataset size (start with 200 for speed)
2. Choose embedding model (default: all-MiniLM-L6-v2)
3. Adjust clustering parameters if needed
4. Enable custom criteria (recommended)

### Step 2: Run Pipeline

Click **🚀 Run Pipeline** button

The pipeline will:
1. Load dataset (real data from `data/aiscientist/data/pubmed_data_2000.csv`)
2. Generate embeddings (using sentence-transformers)
3. Cluster papers (HDBSCAN)
4. Validate clusters (existing validation logic)
5. Generate hypotheses (ranked by priority)

### Step 3: Explore Results

**Clusters Tab**:
- Interactive scatter plot of paper clusters
- Hover to see paper titles
- Bar chart of cluster sizes

**Validation Tab**:
- Overall validation statistics
- Scores by cluster
- Pass/fail visualization

**Hypotheses Tab**:
- Top 10 hypotheses ranked by priority score
- Each hypothesis shows:
  - Type (Meta-Analysis, ML Application, etc.)
  - Cluster information
  - Sample papers
  - Reproducibility score

**Export Tab**:
- Download complete results as JSON
- Download hypotheses as CSV
- Includes configuration and metadata

---

## 🔧 Configuration Options

### Dataset Size
```
200 papers   → ~1-2 minutes
500 papers   → ~3-5 minutes
1000 papers  → ~5-10 minutes
2000 papers  → ~10-20 minutes
```

### Embedding Models

**all-MiniLM-L6-v2** (Default)
- ✅ Fast (384 dimensions)
- ✅ Good for general text
- ⚠️ Not specialized for scientific papers

**allenai/specter** (Recommended for scientific papers)
- ✅ Best for scientific literature
- ✅ 768 dimensions
- ⚠️ Slower than MiniLM

**all-mpnet-base-v2**
- ✅ Balanced performance
- ✅ 768 dimensions
- ✅ Good general-purpose model

### Clustering Parameters

**Min Cluster Size** (default: 15)
- Lower → More clusters, smaller sizes
- Higher → Fewer clusters, larger sizes
- Recommended: 10-20 for 200-500 papers, 15-30 for 1000+ papers

**Min Samples** (default: 5)
- HDBSCAN parameter for core point density
- Lower → More lenient clustering
- Higher → Stricter clustering

**UMAP Components** (default: 10)
- Dimensionality reduction target
- Lower → Faster, less detail
- Higher → Slower, more detail
- Recommended: 5-15

---

## 📁 Data Flow

```
Real Dataset (CSV)
    ↓
Filter (optional: computational papers only)
    ↓
Generate Embeddings (sentence-transformers)
    ↓
Filter NaN embeddings
    ↓
UMAP Dimensionality Reduction
    ↓
HDBSCAN Clustering
    ↓
Validation (standard + custom criteria)
    ↓
Hypothesis Generation
    ↓
Interactive Results Display
```

---

## 🎨 User Interface

### Sidebar (Configuration)
- Dataset settings
- Model selection
- Clustering parameters
- Validation options
- Run button

### Main Area (Results)
- Summary metrics (4 cards)
- 4 tabs:
  1. 🎯 Clusters - Visualizations
  2. ✅ Validation - Scores and status
  3. 💡 Hypotheses - Ranked list
  4. 📥 Export - Download options

---

## 💾 Export Formats

### JSON Export
```json
{
  "config": {
    "dataset_size": 200,
    "embedding_model": "all-MiniLM-L6-v2",
    "min_cluster_size": 15,
    ...
  },
  "summary": {
    "n_papers": 200,
    "n_clusters": 12,
    "n_hypotheses": 10,
    "timestamp": "2025-11-09T16:30:00"
  },
  "hypotheses": [
    {
      "id": 1,
      "cluster_id": 4,
      "type": "ML Application",
      "title": "...",
      "priority_score": 7.5,
      ...
    }
  ],
  "validation": {
    "pass_rate": 0.75,
    "passed": 9,
    "total": 12
  }
}
```

### CSV Export (Hypotheses)
```csv
id,cluster_id,type,title,description,reproducibility,size,priority_score
1,4,ML Application,...,...,0.68,11,7.5
2,5,Meta-Analysis,...,...,0.72,19,7.2
...
```

---

## 🔍 Verification

### Dataset is Real
```python
# Check dataset location
data_path = Path('data/aiscientist/data/pubmed_data_2000.csv')
df = pd.read_csv(data_path)
print(f"Real papers: {len(df)}")
# Output: Real papers: 2000
```

### Embeddings are Real
```python
# Uses sentence-transformers library
embedder = PaperEmbedder(model_name='all-MiniLM-L6-v2')
embeddings = embedder.embed_papers(df)
# Real embeddings from actual paper titles and abstracts
```

### Validation Uses Existing Code
```python
# From src/extraction/classification_validator.py
validator = ClassificationValidator()
validation_results = validator.validate_all_clusters(df, labels)

# From src/extraction/custom_criteria.py
custom_validator = CustomCriteriaValidator()
custom_validator.add_criterion(DataAvailabilityCriterion(), weight=0.15)
```

---

## 🐛 Troubleshooting

### Dataset Not Found
```bash
# Run setup script to download dataset
bash setup.sh
```

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Streamlit Not Found
```bash
pip install streamlit>=1.28.0
```

### Port Already in Use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## 📈 Performance Tips

### For Speed
- Use smaller dataset (200 papers)
- Use `all-MiniLM-L6-v2` model
- Lower UMAP components (5-7)
- Enable "Computational Papers Only" filter

### For Quality
- Use larger dataset (1000-2000 papers)
- Use `allenai/specter` model
- Higher UMAP components (10-15)
- Enable custom validation criteria

---

## 🔒 Data Privacy

- All processing is local
- No data sent to external servers
- Dataset is from public PubMed records
- No API keys required

---

## 📚 Next Steps

1. **Run with default settings** to see how it works
2. **Experiment with parameters** to understand their effects
3. **Export results** for further analysis
4. **Compare different models** to see quality differences
5. **Use hypotheses** as starting points for research

---

## 🤝 Integration with Existing Code

The web app uses:
- ✅ `src/embeddings/paper_embedder.py` - Real embeddings
- ✅ `src/clustering/semantic_clusterer.py` - Real clustering
- ✅ `src/extraction/classification_validator.py` - Real validation
- ✅ `src/extraction/custom_criteria.py` - Real custom criteria
- ✅ `data/aiscientist/data/pubmed_data_2000.csv` - Real dataset

**No mock data, no fake results, 100% real pipeline**

---

## 📞 Support

- Check `README.md` for project overview
- Check `QUICKSTART.md` for command-line usage
- Check `EXPERIMENTAL_DESIGN_GUIDE.md` for hypothesis details
- Open an issue on GitHub for bugs

---

**Version**: 1.0
**Last Updated**: November 9, 2025
**Repository**: https://github.com/ebaenamar/research-semantic-poc
