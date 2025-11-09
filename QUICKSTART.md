# Quick Start Guide

Get started with semantic classification and hypothesis generation in 5 minutes.

## Prerequisites

- Python 3.11+
- Git

## Installation

### 1. Clone and Setup

```bash
cd /Users/e.baena/CascadeProjects/research-semantic-poc

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for advanced NLP)
python -m spacy download en_core_web_sm
```

### 2. Get the Data

```bash
# Clone the aiscientist repository with PubMed data
mkdir -p data
cd data
git clone https://github.com/sergeicu/aiscientist
cd ..
```

This gives you:
- `data/aiscientist/pubmed_data.csv` - 20,415 Boston Children's Hospital papers
- `data/aiscientist/pubmed_data_2000.csv` - 2,000 paper subset (for testing)

## Running the Pipeline

### Option 1: Full Pipeline (Recommended)

Run everything in one command:

```bash
python scripts/run_full_pipeline.py
```

This will:
1. Load the dataset (2,000 papers by default)
2. Generate semantic embeddings
3. Cluster papers by similarity
4. Analyze research gaps
5. Generate testable hypotheses

**Time**: ~5-10 minutes on modern laptop

**Output files** (in `output/`):
- `embeddings.pkl` - Semantic embeddings
- `clusters.json` - Cluster analysis
- `gap_analysis.json` - Research gaps
- `hypotheses.json` - Generated hypotheses

### Option 2: Use Full Dataset

For the complete 20k papers:

```bash
python scripts/run_full_pipeline.py --data-file data/aiscientist/pubmed_data.csv
```

**Time**: ~30-60 minutes

### Option 3: Custom Parameters

```bash
python scripts/run_full_pipeline.py \
  --data-file data/aiscientist/pubmed_data_2000.csv \
  --model all-MiniLM-L6-v2 \
  --cluster-method hdbscan \
  --min-cluster-size 10 \
  --batch-size 32
```

**Available options**:
- `--model`: Embedding model
  - `all-MiniLM-L6-v2` (fast, default)
  - `allenai/specter` (specialized for papers, slower)
  
- `--cluster-method`: Clustering algorithm
  - `hdbscan` (density-based, default)
  - `dbscan` (density-based, simpler)
  - `hierarchical` (tree-based)
  
- `--min-cluster-size`: Minimum papers per cluster (default: 10)
- `--batch-size`: Embedding batch size (default: 32)
- `--force`: Regenerate embeddings even if they exist

## Using Claude Agents

After running the pipeline, use Claude Code agents for interactive analysis:

### 1. Start Claude Code

```bash
cd /Users/e.baena/CascadeProjects/research-semantic-poc
claude
```

### 2. Interact with Agents

**Semantic Classifier Agent**:
```
@semantic-classifier
What are the main research themes in the dataset?
```

```
@semantic-classifier
Which clusters have the most methodological gaps?
```

```
@semantic-classifier
Find cross-cluster opportunities with high novelty potential
```

**Hypothesis Generator Agent**:
```
@hypothesis-generator-enhanced
Generate 5 novel hypotheses from the gap analysis
```

```
@hypothesis-generator-enhanced
Evaluate the top hypothesis for feasibility and impact
```

```
@hypothesis-generator-enhanced
What data would I need to verify hypothesis #3?
```

## Expected Results

### Clusters
You should see 8-12 semantic clusters representing different research areas, for example:
- Clinical trials and treatment outcomes
- Genetic and genomic studies
- Epidemiology and population health
- Imaging and diagnostics
- Immunology and infectious diseases

### Gaps Identified
- **Temporal**: Areas with outdated research (>5 years)
- **Methodological**: Underutilized techniques (e.g., ML, computational)
- **Contradictions**: Papers with conflicting findings
- **Opportunities**: Cross-cluster combinations

### Hypotheses Generated
Typically 15-25 hypotheses across categories:
- Temporal updates (applying modern methods)
- Methodological transfers (new technique applications)
- Contradiction resolutions (reconciling conflicts)
- Cross-domain synthesis (novel combinations)
- Data-driven discoveries (pattern-based)

## Exploring Results

### View Summary Statistics

```bash
# Quick summary
python -c "
import json
with open('output/hypotheses.json') as f:
    data = json.load(f)
print(f\"Total hypotheses: {data['metadata']['total_hypotheses']}\")
print(f\"By type: {data['summary']['by_type']}\")
print(f\"\\nTop hypothesis: {data['summary']['top_5'][0]['hypothesis']}\")
"
```

### Interactive Analysis

Use Jupyter notebooks for deeper analysis:

```bash
jupyter notebook notebooks/
```

## Troubleshooting

### Issue: Out of memory
**Solution**: 
- Use smaller dataset: `pubmed_data_2000.csv`
- Reduce batch size: `--batch-size 16`
- Use smaller model: keep default `all-MiniLM-L6-v2`

### Issue: Slow embedding generation
**Solution**:
- Embeddings are cached after first run
- Use `--force` only when needed
- GPU acceleration: Install `torch` with CUDA support

### Issue: Too few/many clusters
**Solution**:
- Adjust `--min-cluster-size` (lower = more clusters)
- Try different `--cluster-method`
- Use full dataset for better clustering

### Issue: Data file not found
**Solution**:
```bash
cd data
git clone https://github.com/sergeicu/aiscientist
```

## Next Steps

1. **Review hypotheses**: Check `output/hypotheses.json` for novel ideas
2. **Explore gaps**: Examine `output/gap_analysis.json` for opportunities
3. **Analyze clusters**: Study `output/clusters.json` for themes
4. **Interactive exploration**: Use Claude agents for Q&A
5. **Custom analysis**: Create Jupyter notebooks in `notebooks/`

## Example Workflow

```bash
# 1. Run pipeline
python scripts/run_full_pipeline.py

# 2. Quick check results
cat output/hypotheses.json | head -50

# 3. Start interactive session
claude

# 4. In Claude Code:
@semantic-classifier Summarize the research landscape

@hypothesis-generator-enhanced Which hypothesis has the highest impact potential?

# 5. Deep dive
jupyter notebook
```

## Performance Expectations

| Dataset Size | Embedding Time | Clustering Time | Total Time |
|--------------|----------------|-----------------|------------|
| 2,000 papers | ~3 min         | ~2 min          | ~5 min     |
| 20,000 papers| ~25 min        | ~10 min         | ~35 min    |

*Times on M1 Mac with 16GB RAM*

## Tips

- **Start small**: Use the 2k subset first to understand the output
- **Iterate**: Re-run with different parameters to see impact
- **Cache**: Embeddings are cached - only regenerate if changing models
- **Explore**: Use agents to ask specific questions about results
- **Document**: Keep notes on interesting findings in notebooks

## Getting Help

- Check `README.md` for architecture details
- Review agent files in `.claude/agents/`
- Examine source code in `src/`
- Look at output JSON files for structure

Ready to discover novel research directions! 🚀
