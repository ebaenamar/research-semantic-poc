# Research Semantic POC

**Intelligent semantic classification and hypothesis generation system for scientific literature with extensible validation criteria.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project implements a **classification-first approach** to scientific research discovery:

1. **Semantic Classification** - Cluster papers by methodology, domain, and conceptual framework
2. **Scientific Validation** - Rigorous multi-dimensional quality assessment
3. **Gap Analysis** - Identify research gaps, contradictions, and opportunities
4. **Hypothesis Generation** - Generate novel, testable hypotheses beyond state-of-the-art

### Key Innovation

**Extensible validation system** - Add custom criteria (clinical trial sponsors, data availability, etc.) without modifying core code.

## 📖 Documentation

### 🌟 Essential Reading

**START HERE**: [`TAKEAWAYS.md`](TAKEAWAYS.md) - Complete project summary with architecture, features, and lessons learned

### 📚 Core Documentation

- **Setup**: [`QUICKSTART.md`](QUICKSTART.md) - Get running in 5 minutes
- **Latest App**: [`APP_V3_GUIDE.md`](APP_V3_GUIDE.md) - Web interface with PubMed enrichment
- **Architecture**: [`ARCHITECTURE.md`](ARCHITECTURE.md) - System design and components
- **Validation**: [`VALIDATION_CRITERIA.md`](VALIDATION_CRITERIA.md) - Quality assessment criteria

### 🔬 Advanced Topics

- [`EXPERIMENTAL_DESIGN_GUIDE.md`](EXPERIMENTAL_DESIGN_GUIDE.md) - Research protocols
- [`HYPOTHESIS_EXECUTION_SUMMARY.md`](HYPOTHESIS_EXECUTION_SUMMARY.md) - Execution checklist
- [`TESTING_CHECKLIST.md`](TESTING_CHECKLIST.md) - Testing procedures
- [`NOISE_REDUCTION_GUIDE.md`](NOISE_REDUCTION_GUIDE.md) - Clustering optimization

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/ebaenamar/research-semantic-poc.git
cd research-semantic-poc

# Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run web interface (latest version with PubMed enrichment)
streamlit run app_v3.py --server.port 8503

# Or run batch script
python scripts/generate_reproducible_hypotheses.py
```

## 📊 Architecture

```
Dataset (20k PubMed papers)
    ↓
Semantic Embedding (transformer models)
    ↓
Clustering (HDBSCAN)
    ↓
Scientific Validation (5 dimensions + custom criteria)
    ↓
Gap Analysis (temporal, methodological, contradictions)
    ↓
Hypothesis Generation (ranked by novelty + feasibility + impact)
```

## 🔬 Key Features

### 1. Scientific Semantic Classifier
- **Transformer-based embeddings** (all-MiniLM-L6-v2 or allenai/specter)
- **Multi-dimensional clustering** by methodology, domain, framework
- **Rigorous validation** with scientific criteria
- **Explicit justification** for all classifications

### 2. Extensible Validation System ⭐
- **Built-in criteria**: Methodological coherence, temporal consistency, framework alignment
- **Custom criteria**: Clinical trial sponsors, data availability, replication status
- **Modular architecture**: Add your own criteria without touching core code
- **Weighted scoring**: Configure importance of each criterion

### 3. Gap Analysis
- **Temporal gaps**: Outdated research areas
- **Methodological gaps**: Underused techniques
- **Contradictions**: Conflicting findings requiring resolution
- **Cross-cluster opportunities**: Novel combinations

### 4. Hypothesis Generation
- **Evidence-based**: Grounded in validated cluster analysis
- **Verifiable**: Includes specific verification plans
- **Ranked**: By novelty, feasibility, and impact
- **Actionable**: Clear next steps for testing

## 📦 Data Source

Uses the Boston Children's Hospital PubMed dataset from [sergeicu/aiscientist](https://github.com/sergeicu/aiscientist):
- 20,415 PubMed records
- Includes titles, abstracts, metadata, citations
- Automatically downloaded during setup

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Git
- Git LFS (for dataset)

### Setup

```bash
# Clone repository
git clone https://github.com/ebaenamar/research-semantic-poc.git
cd research-semantic-poc

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

## 🧪 Testing

### Step-by-Step Component Testing

```bash
# Test all components independently
python scripts/test_automated.py
```

**Tests**:
1. ✅ Data loading (2,000 papers)
2. ✅ Embedding generation (384-dim vectors)
3. ✅ Semantic clustering (HDBSCAN)
4. ✅ Scientific validation (5 dimensions)

### Custom Criteria Testing

```bash
# Test extensible validation system
python scripts/test_custom_criteria.py
```

### Embedding Evaluation

```bash
# Compare embedding models
python scripts/evaluate_embeddings.py
```

## 🎯 Usage

### Basic Pipeline

```bash
# Run with default settings (2,000 papers)
python scripts/run_full_pipeline.py

# Run with full dataset (20,000 papers)
python scripts/run_full_pipeline.py --data-file data/aiscientist/data/pubmed_data.csv

# Use SPECTER for better scientific similarity
python scripts/run_full_pipeline.py --model allenai/specter
```

### Custom Criteria

```python
from extraction.custom_criteria import (
    CustomCriteriaValidator,
    ClinicalTrialSponsorCriterion,
    DataAvailabilityCriterion
)

# Create validator
validator = CustomCriteriaValidator()

# Add built-in criteria
validator.add_criterion(ClinicalTrialSponsorCriterion(weight=0.2))
validator.add_criterion(DataAvailabilityCriterion(weight=0.15))

# Evaluate clusters
results = validator.evaluate_all_clusters(df, labels)
```

### Create Your Own Criterion

```python
from extraction.custom_criteria import ValidationCriterion

class MyCriterion(ValidationCriterion):
    def __init__(self, weight=0.1):
        super().__init__("my_criterion", weight)
    
    def evaluate(self, cluster_df, text_column='abstract'):
        # Your validation logic
        score = 0.8  # Calculate score (0-1)
        
        return {
            'score': score,
            'details': {'your': 'data'},
            'interpretation': 'Your explanation'
        }

# Use it
validator.add_criterion(MyCriterion(weight=0.15))
```

## 📂 Project Structure

```
research-semantic-poc/
├── src/
│   ├── embeddings/           # Semantic embedding generation
│   ├── clustering/           # HDBSCAN clustering
│   └── extraction/
│       ├── classification_validator.py    # Scientific validation
│       ├── custom_criteria.py            # ⭐ Extensible criteria system
│       ├── gap_analyzer.py               # Gap identification
│       └── hypothesis_generator.py       # Hypothesis creation
├── scripts/
│   ├── test_automated.py                 # Component testing
│   ├── test_custom_criteria.py          # Custom criteria demo
│   ├── evaluate_embeddings.py           # Model comparison
│   └── run_full_pipeline.py             # Full pipeline
├── .claude/
│   └── agents/                           # Claude Code agents
├── config/
│   ├── pipeline_config.yaml             # Configuration
│   └── mcp_pubmed_setup.md              # MCP server setup
├── notebooks/                            # Jupyter analysis
├── docs/
│   ├── ARCHITECTURE.md                  # Technical architecture
│   ├── EXTENSIBILITY.md                 # Custom criteria guide
│   ├── REFINED_APPROACH.md              # Methodology
│   ├── IMPROVEMENTS_SUMMARY.md          # Recent improvements
│   └── TEST_RESULTS.md                  # Test results
└── README.md                            # This file
```

## 🔧 Configuration

Edit `config/pipeline_config.yaml`:

```yaml
embeddings:
  model_name: "all-MiniLM-L6-v2"  # or "allenai/specter"
  batch_size: 32

clustering:
  method: "hdbscan"
  min_cluster_size: 10

gap_analysis:
  temporal:
    outdated_threshold_years: 5
```

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical deep dive
- **[EXTENSIBILITY.md](EXTENSIBILITY.md)** - Custom criteria guide
- **[REFINED_APPROACH.md](REFINED_APPROACH.md)** - Scientific methodology
- **[TEST_RESULTS.md](TEST_RESULTS.md)** - Validation results

## 🎓 Key Concepts

### Scientific Validation Dimensions

1. **Methodological Coherence** (35%) - Papers use same research methods
2. **Framework Coherence** (25%) - Consistent conceptual approach
3. **Temporal Coherence** (15%) - Papers within reasonable time span
4. **Internal Consistency** (15%) - Shared vocabulary and terminology
5. **MeSH Coherence** (10%) - Medical subject heading consistency

### Custom Criteria Examples

- **ClinicalTrialSponsorCriterion** - Identifies funding sources (academic vs industry)
- **DataAvailabilityCriterion** - Checks if data is available for verification
- **ReplicationStatusCriterion** - Distinguishes original vs replication studies

## 🚀 Advanced Usage

### With Claude Code Agents

```bash
# Start Claude Code
claude

# Use semantic classifier agent
@scientific-semantic-classifier
Analyze the validated clusters and explain scientific rationale

# Use hypothesis generator agent
@hypothesis-generator-enhanced
Generate novel hypotheses from cluster analysis
```

### With MCP Server (Optional)

For real-time PubMed access:

```bash
# See config/mcp_pubmed_setup.md for instructions
# Enables direct querying of PubMed database
```

## 📊 Performance

| Dataset Size | Embedding Time | Clustering Time | Total Time |
|--------------|----------------|-----------------|------------|
| 2,000 papers | ~3 min         | ~2 min          | ~5 min     |
| 20,000 papers| ~25 min        | ~10 min         | ~35 min    |

*M1 Mac, 16GB RAM*

## 🤝 Contributing

Contributions welcome! Areas of interest:

1. **New validation criteria** - Domain-specific validators
2. **Embedding models** - Test new transformer models
3. **Gap analysis** - Additional gap types
4. **Hypothesis strategies** - Novel generation approaches

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Dataset: [sergeicu/aiscientist](https://github.com/sergeicu/aiscientist)
- Inspired by: [sergeicu/claude-agentic-researchers](https://github.com/sergeicu/claude-agentic-researchers)
- Embeddings: [sentence-transformers](https://www.sbert.net/)
- Clustering: [HDBSCAN](https://hdbscan.readthedocs.io/)

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ for the research community**
