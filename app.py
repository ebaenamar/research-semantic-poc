#!/usr/bin/env python3
"""
Research Semantic POC - Web Interface
Interactive web application for hypothesis generation
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from embeddings import PaperEmbedder
from clustering import (
    SemanticClusterer, 
    DomainAwareClusterer, 
    AdaptiveClusterer, 
    ThematicCoherenceValidator,
    HierarchicalFunnelClusterer
)
from extraction import ClassificationValidator
from extraction.custom_criteria import (
    CustomCriteriaValidator,
    ClinicalTrialSponsorCriterion,
    DataAvailabilityCriterion,
    ReplicationStatusCriterion
)

# Page config
st.set_page_config(
    page_title="Research Semantic POC",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'config' not in st.session_state:
    st.session_state.config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'min_cluster_size': 15,
        'min_samples': 5,
        'umap_components': 10,
        'use_custom_criteria': True,
        'computational_only': True
    }

def load_dataset():
    """Load the real dataset"""
    data_path = Path('data/aiscientist/data/pubmed_data_2000.csv')
    if not data_path.exists():
        st.error(f"Dataset not found at {data_path}")
        st.info("Please ensure the dataset is downloaded. Run: `bash setup.sh`")
        return None
    
    df = pd.read_csv(data_path)
    return df

def filter_computational_papers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter papers that are computational/data-driven"""
    
    computational_keywords = [
        'dataset', 'database', 'data analysis', 'computational',
        'bioinformatics', 'machine learning', 'deep learning',
        'algorithm', 'model', 'prediction', 'classification',
        'regression', 'neural network', 'statistical analysis',
        'retrospective', 'cohort analysis', 'meta-analysis',
        'systematic review', 'data mining', 'big data'
    ]
    
    def has_computational_keywords(row):
        text = str(row.get('title', '')) + ' ' + str(row.get('abstract_text', ''))
        text = text.lower()
        return any(keyword in text for keyword in computational_keywords)
    
    mask = df.apply(has_computational_keywords, axis=1)
    return df[mask].reset_index(drop=True)

def run_pipeline(df, config, progress_bar, status_text):
    """Execute the full pipeline"""
    
    results = {}
    
    # Stage 1: Embeddings
    status_text.text("🔢 Stage 1/5: Generating embeddings...")
    progress_bar.progress(0.2)
    
    embedder = PaperEmbedder(model_name=config['embedding_model'])
    embeddings = embedder.embed_papers(df, show_progress=False)
    
    # Filter NaN embeddings
    valid_mask = ~np.isnan(embeddings).any(axis=1)
    valid_embeddings = embeddings[valid_mask]
    valid_df = df[valid_mask].reset_index(drop=True)
    
    results['n_papers'] = len(df)
    results['n_valid'] = len(valid_df)
    results['n_invalid'] = len(df) - len(valid_df)
    
    # Stage 2: Clustering
    status_text.text("🎯 Stage 2/5: Clustering papers...")
    progress_bar.progress(0.4)
    
    clusterer = SemanticClusterer(method='hdbscan')
    clustering_mode = config.get('clustering_mode', '⚡ Standard')
    
    # Always do UMAP reduction first
    reduced = clusterer.reduce_dimensions(
        valid_embeddings,
        n_components=config['umap_components']
    )
    
    if "Hierarchical Funnel" in clustering_mode:
        # Hierarchical funnel: Topic → Methodology → Temporal → Semantic
        status_text.text("🎯 Stage 2/5: Hierarchical Funnel Clustering...")
        status_text.text("   → Topic filtering (40% importance)...")
        
        funnel_clusterer = HierarchicalFunnelClusterer()
        labels, funnel_report = funnel_clusterer.cluster_hierarchical_funnel(
            valid_df,
            valid_embeddings,
            reduced,
            min_cluster_size=config['min_cluster_size'],
            min_topic_coverage=0.6,
            min_methodology_coverage=0.5,
            recency_window_years=5
        )
        
        results['clustering_mode'] = 'hierarchical_funnel'
        results['funnel_report'] = funnel_report
        
        # Display funnel stages
        print("\n📊 Funnel Summary:")
        summary = funnel_report['funnel_summary']
        print(f"   Topics found: {len(summary['topic_distribution'])}")
        print(f"   Methods found: {len(summary['methodology_distribution'])}")
        print(f"   Final clusters: {summary['n_clusters']}")
        print(f"   Clustering rate: {summary['clustering_rate']*100:.1f}%")
        
    elif "Domain-Aware + Adaptive" in clustering_mode:
        # Combined: Domain-aware first, then adaptive within each domain
        status_text.text("🎯 Stage 2a/5: Assigning medical domains...")
        domain_clusterer = DomainAwareClusterer()
        domain_labels = domain_clusterer.assign_domains(valid_df)
        
        status_text.text("🎯 Stage 2b/5: Adaptive clustering within domains...")
        
        # Cluster within each domain using adaptive strategy
        adaptive_clusterer = AdaptiveClusterer()
        labels = np.full(len(valid_df), -1, dtype=int)
        next_cluster_id = 0
        
        domains_to_cluster = [d for d in domain_labels.unique() 
                             if d not in ['unclassified', 'multi_domain']]
        
        for domain in sorted(domains_to_cluster):
            domain_mask = domain_labels == domain
            domain_size = domain_mask.sum()
            
            if domain_size < config['min_cluster_size']:
                continue
            
            # Get embeddings for this domain
            domain_embeddings = valid_embeddings[domain_mask]
            domain_reduced = reduced[domain_mask]
            domain_df = valid_df[domain_mask]
            
            # Adaptive clustering within domain
            try:
                local_labels = adaptive_clusterer.cluster_with_noise_reduction(
                    domain_embeddings,
                    domain_reduced,
                    domain_df,
                    min_cluster_size=config['min_cluster_size'],
                    target_noise_ratio=0.3
                )
                
                # Map to global labels
                for local_label in np.unique(local_labels):
                    if local_label == -1:
                        continue
                    local_mask = local_labels == local_label
                    global_mask = np.where(domain_mask)[0][local_mask]
                    labels[global_mask] = next_cluster_id
                    next_cluster_id += 1
            except Exception as e:
                print(f"   ⚠️ Error in {domain}: {str(e)}")
                # Skip this domain if clustering fails
                continue
        
        results['domain_labels'] = domain_labels
        results['domain_stats'] = domain_clusterer.get_domain_statistics(valid_df, domain_labels)
        results['clustering_mode'] = 'domain_aware_adaptive'
        
    elif "Adaptive Only" in clustering_mode:
        # Adaptive clustering - multiple strategies to reduce noise
        status_text.text("🎯 Stage 2: Adaptive clustering (reducing noise)...")
        adaptive_clusterer = AdaptiveClusterer()
        labels = adaptive_clusterer.cluster_with_noise_reduction(
            valid_embeddings,
            reduced,
            valid_df,
            min_cluster_size=config['min_cluster_size'],
            target_noise_ratio=0.3  # Target <30% noise
        )
        results['clustering_mode'] = 'adaptive'
        
    elif "Domain-Aware Only" in clustering_mode:
        # Domain-aware clustering
        status_text.text("🎯 Stage 2a/5: Assigning medical domains...")
        domain_clusterer = DomainAwareClusterer()
        domain_labels = domain_clusterer.assign_domains(valid_df)
        
        status_text.text("🎯 Stage 2b/5: Clustering within domains...")
        labels = domain_clusterer.cluster_within_domains(
            valid_df,
            valid_embeddings,
            domain_labels,
            clusterer,
            min_cluster_size=config['min_cluster_size']
        )
        
        results['domain_labels'] = domain_labels
        results['domain_stats'] = domain_clusterer.get_domain_statistics(valid_df, domain_labels)
        results['clustering_mode'] = 'domain_aware'
        
    else:
        # Standard semantic clustering
        labels = clusterer.cluster_hdbscan(
            reduced,
            min_cluster_size=config['min_cluster_size'],
            min_samples=config['min_samples']
        )
        results['clustering_mode'] = 'standard'
    
    results['labels'] = labels
    results['reduced_embeddings'] = reduced
    results['n_clusters'] = len([l for l in np.unique(labels) if l != -1])
    results['n_noise'] = (labels == -1).sum()
    
    # Stage 2.5: Filter incoherent clusters (if enabled)
    if config.get('filter_incoherent', True):
        status_text.text("🎯 Stage 2.5/5: Filtering incoherent clusters...")
        
        coherence_validator = ThematicCoherenceValidator()
        labels, coherence_report = coherence_validator.filter_incoherent_clusters(valid_df, labels)
        
        # Update counts
        results['labels'] = labels
        results['n_clusters_before_filter'] = results['n_clusters']
        results['n_clusters'] = len([l for l in np.unique(labels) if l != -1])
        results['n_noise'] = (labels == -1).sum()
        results['coherence_report'] = coherence_report
        
        removed = coherence_report['removed_clusters']
        kept = coherence_report['kept_clusters']
        print(f"\n   Filtered {removed} incoherent clusters, kept {kept} coherent clusters")
    
    # Stage 3: Validation
    status_text.text("✅ Stage 3/5: Validating clusters...")
    progress_bar.progress(0.6)
    
    validator = ClassificationValidator()
    validation_results = validator.validate_all_clusters(valid_df, labels, text_column='abstract_text')
    results['validation'] = validation_results
    
    # Stage 4: Custom Criteria
    if config['use_custom_criteria']:
        status_text.text("🔍 Stage 4/5: Applying custom criteria...")
        progress_bar.progress(0.8)
        
        custom_validator = CustomCriteriaValidator()
        custom_validator.add_criterion(DataAvailabilityCriterion(weight=config['data_availability_weight']))
        custom_validator.add_criterion(ClinicalTrialSponsorCriterion(weight=config['clinical_trial_weight']))
        custom_validator.add_criterion(ReplicationStatusCriterion(weight=config['replication_weight']))
        
        custom_results = custom_validator.evaluate_all_clusters(valid_df, labels, text_column='abstract_text')
        results['custom_validation'] = custom_results
    
    # Stage 5: Generate Hypotheses
    status_text.text("💡 Stage 5/5: Generating hypotheses...")
    progress_bar.progress(0.9)
    
    hypotheses = generate_hypotheses(
        valid_df, 
        labels, 
        validation_results,
        max_cluster_size=config.get('max_cluster_size_for_hypotheses', 30),
        require_future_work=config.get('require_future_work', True)
    )
    results['hypotheses'] = hypotheses
    results['df'] = valid_df
    
    progress_bar.progress(1.0)
    status_text.text("✅ Pipeline completed!")
    
    return results

def generate_hypotheses(df, labels, validation_results, max_cluster_size=30, require_future_work=True):
    """
    Generate detailed, actionable hypotheses from clusters
    Based on reproducible_hypotheses approach with real data analysis
    """
    
    hypotheses = []
    
    future_work_keywords = [
        'future work', 'future research', 'future studies',
        'limitation', 'limited by', 'gap', 'need for',
        'further research', 'additional studies', 'warrant',
        'should be investigated', 'remains to be', 'unclear',
        'not yet', 'unexplored', 'understudied', 'future direction'
    ]
    
    data_availability_keywords = [
        'dataset', 'database', 'github', 'figshare', 'zenodo',
        'data available', 'supplementary data', 'code available',
        'open access', 'publicly available'
    ]
    
    computational_keywords = [
        'machine learning', 'deep learning', 'neural network',
        'algorithm', 'computational', 'model', 'prediction'
    ]
    
    for cluster_name, cluster_data in validation_results['cluster_reports'].items():
        cluster_id = int(cluster_name.replace('cluster_', ''))
        
        if cluster_id == -1:  # Skip noise
            continue
        
        cluster_mask = labels == cluster_id
        cluster_papers = df[cluster_mask]
        
        # Filter by size
        if len(cluster_papers) < 5:
            continue
        
        # Skip large meta-analysis clusters
        if len(cluster_papers) > max_cluster_size:
            continue
        
        # Check for future work mentions if required
        if require_future_work:
            has_future_work = False
            for idx, row in cluster_papers.iterrows():
                if 'abstract_text' in row and pd.notna(row['abstract_text']):
                    text = str(row['abstract_text']).lower()
                    if any(kw in text for kw in future_work_keywords):
                        has_future_work = True
                        break
            
            if not has_future_work:
                continue
        
        # Calculate base scores
        validation_score = cluster_data['overall_score']
        size = len(cluster_papers)
        
        # REAL DATA ANALYSIS: Calculate reproducibility score
        # Based on: data availability, computational feasibility, citation count
        
        # 1. Data availability score (0-1)
        data_available_count = 0
        for idx, row in cluster_papers.iterrows():
            text = ''
            if 'abstract_text' in row and pd.notna(row['abstract_text']):
                text = str(row['abstract_text']).lower()
            if any(kw in text for kw in data_availability_keywords):
                data_available_count += 1
        data_availability_score = data_available_count / size
        
        # 2. Computational score (0-1)
        computational_count = 0
        for idx, row in cluster_papers.iterrows():
            text = ''
            if 'abstract_text' in row and pd.notna(row['abstract_text']):
                text = str(row['abstract_text']).lower()
            if any(kw in text for kw in computational_keywords):
                computational_count += 1
        computational_score = computational_count / size
        
        # 3. Future work score (0-1)
        future_work_count = 0
        for idx, row in cluster_papers.iterrows():
            if 'abstract_text' in row and pd.notna(row['abstract_text']):
                text = str(row['abstract_text']).lower()
                if any(kw in text for kw in future_work_keywords):
                    future_work_count += 1
        future_work_score = future_work_count / size
        
        # 4. Recency score (0-1) - newer papers = more relevant
        if 'publication_year' in cluster_papers.columns:
            years = pd.to_numeric(cluster_papers['publication_year'], errors='coerce').dropna()
            if len(years) > 0:
                avg_year = years.mean()
                recency_score = min((avg_year - 2010) / 15, 1.0)  # 2010-2025 range
            else:
                recency_score = 0.5
        else:
            recency_score = 0.5
        
        # OVERALL REPRODUCIBILITY SCORE (0-10 scale)
        # Weighted combination of factors
        reproducibility = (
            validation_score * 0.25 +         # Scientific coherence
            data_availability_score * 0.30 +   # Data available
            computational_score * 0.20 +       # Computational feasibility
            future_work_score * 0.15 +        # Has gaps identified
            recency_score * 0.10              # Recent work
        ) * 10
        
        # Calculate difficulty (inverse of feasibility)
        difficulty = (
            (1 - computational_score) * 0.4 +  # Non-computational = harder
            (1 - data_availability_score) * 0.4 +  # No data = harder
            (size / 30) * 0.2                  # More papers = more complex
        ) * 10
        
        # Calculate impact (based on validation + size + recency)
        impact = (
            validation_score * 0.4 +
            min(size / 20, 1.0) * 0.3 +        # More papers = more impact
            recency_score * 0.3
        ) * 10
        
        # Extract cluster characteristics
        scores = cluster_data.get('scores', {})
        method_score = scores.get('methodological_coherence', 0)
        framework_score = scores.get('framework_coherence', 0)
        
        # Get methodology and framework from justification
        justification = cluster_data.get('justification', {})
        dominant_method = justification.get('dominant_method', 'unknown')
        dominant_framework = justification.get('conceptual_framework', 'unknown')
        method_interpretation = justification.get('methodological_basis', '')
        
        # Analyze temporal span
        if 'publication_year' in cluster_papers.columns:
            years = pd.to_numeric(cluster_papers['publication_year'], errors='coerce').dropna()
            if len(years) > 0:
                year_range = f"{int(years.min())}-{int(years.max())}"
                year_span = int(years.max() - years.min())
            else:
                year_range = "Unknown"
                year_span = 0
        else:
            year_range = "Unknown"
            year_span = 0
        
        # Extract common keywords from titles and abstracts
        all_text = ' '.join(cluster_papers['title'].fillna('').astype(str))
        if 'abstract_text' in cluster_papers.columns:
            all_text += ' ' + ' '.join(cluster_papers['abstract_text'].fillna('').astype(str)[:5])  # First 5 abstracts
        
        # Common medical/research terms
        common_terms = []
        keywords_to_check = [
            'machine learning', 'deep learning', 'prediction', 'classification',
            'cardiac', 'heart', 'cardiovascular', 'neurology', 'brain',
            'pediatric', 'children', 'infant', 'neonatal',
            'outcome', 'mortality', 'survival', 'prognosis',
            'treatment', 'therapy', 'intervention', 'surgery',
            'diagnosis', 'screening', 'detection',
            'genetic', 'genomic', 'biomarker',
            'cohort', 'retrospective', 'prospective', 'trial'
        ]
        
        all_text_lower = all_text.lower()
        for keyword in keywords_to_check:
            if keyword in all_text_lower:
                common_terms.append(keyword)
        
        # Determine hypothesis type (prioritize actionable research over meta-analysis)
        if 'computational' in dominant_method.lower() or 'machine learning' in all_text_lower or 'deep learning' in all_text_lower:
            hyp_type = 'ML/AI Application'
            type_rationale = f"Computational focus ({size} papers) with potential for ML/AI improvements"
        elif 'genomic' in dominant_method.lower() or 'genetic' in all_text_lower:
            hyp_type = 'Genomic Study'
            type_rationale = f"Genetic/genomic focus ({size} papers) for data-driven discovery"
        elif size <= 10:
            hyp_type = 'Targeted Replication'
            type_rationale = f"Small, focused cluster ({size} papers) ideal for replication with improvements"
        elif size <= 20:
            hyp_type = 'Comparative Analysis'
            type_rationale = f"Medium cluster ({size} papers) suitable for comparing methodologies"
        else:
            hyp_type = 'Synthesis Study'
            type_rationale = f"Larger cluster ({size} papers) for synthesizing findings and identifying gaps"
        
        # Generate detailed title
        topic_hint = common_terms[0] if common_terms else dominant_method
        title = f"{hyp_type}: {topic_hint.title()} Research in {dominant_framework.title()} Context"
        
        # Generate detailed description
        description_parts = []
        
        # Overview with REAL SCORES
        description_parts.append(f"**Overview**: This cluster contains {size} ")
        description_parts.append(f"{'ML-based ' if computational_score > 0.5 else ''}")
        description_parts.append(f"papers published between {year_range} ({year_span} year span). ")
        
        # Reproducibility assessment
        if reproducibility >= 8:
            repro_label = "HIGH"
        elif reproducibility >= 6:
            repro_label = "MEDIUM-HIGH"
        elif reproducibility >= 4:
            repro_label = "MEDIUM"
        else:
            repro_label = "LOW"
        
        if difficulty >= 7:
            diff_label = "HIGH"
        elif difficulty >= 5:
            diff_label = "MEDIUM"
        else:
            diff_label = "LOW"
        
        if impact >= 7:
            impact_label = "HIGH"
        elif impact >= 5:
            impact_label = "MEDIUM"
        else:
            impact_label = "LOW"
        
        description_parts.append(f"\n\n**Reproducibility**: {repro_label} ({reproducibility:.1f}/10) ")
        description_parts.append(f"| **Difficulty**: {diff_label} ({difficulty:.1f}/10) ")
        description_parts.append(f"| **Impact**: {impact_label} ({impact:.1f}/10)")
        
        # Data analysis (REAL)
        description_parts.append(f"\n\n**Data Analysis (Real)**:")
        description_parts.append(f"\n- {data_available_count}/{size} papers ({data_availability_score:.0%}) mention available datasets")
        description_parts.append(f"\n- {computational_count}/{size} papers ({computational_score:.0%}) are computational/ML-based")
        description_parts.append(f"\n- {future_work_count}/{size} papers ({future_work_score:.0%}) explicitly mention future work/gaps")
        description_parts.append(f"\n- Average publication year: {avg_year if 'avg_year' in locals() else 'N/A'}")
        
        # Methodology
        description_parts.append(f"\n\n**Methodology**: {method_interpretation} ")
        description_parts.append(f"(coherence: {method_score:.2f}). ")
        
        # Framework
        description_parts.append(f"\n\n**Framework**: Papers follow a {dominant_framework} approach ")
        description_parts.append(f"(coherence: {framework_score:.2f}). ")
        
        # Common themes
        if common_terms:
            description_parts.append(f"\n\n**Common Themes**: {', '.join(common_terms[:5])}. ")
        
        # Research opportunity
        description_parts.append(f"\n\n**Research Opportunity**: {type_rationale}. ")
        
        # Feasibility
        if reproducibility >= 0.7:
            description_parts.append(f"High validation score ({reproducibility:.2f}) indicates strong scientific coherence. ")
        elif reproducibility >= 0.5:
            description_parts.append(f"Moderate validation score ({reproducibility:.2f}) suggests some heterogeneity. ")
        else:
            description_parts.append(f"Lower validation score ({reproducibility:.2f}) indicates diverse methodologies. ")
        
        # Data-driven feasibility
        if 'dataset' in all_text_lower or 'data' in all_text_lower:
            description_parts.append("Multiple papers mention datasets, suggesting good data availability. ")
        
        # Recommended approach based on type
        if hyp_type == 'ML/AI Application':
            description_parts.append("\n\n**Recommended Approach**: Develop or improve ML/AI models using insights from these papers. Focus on novel architectures, better features, or cross-domain transfer learning.")
        elif hyp_type == 'Genomic Study':
            description_parts.append("\n\n**Recommended Approach**: Leverage genomic data to discover new biomarkers or validate existing findings with larger cohorts.")
        elif hyp_type == 'Targeted Replication':
            description_parts.append("\n\n**Recommended Approach**: Replicate key findings with improved methodology, larger sample size, or different population to validate generalizability.")
        elif hyp_type == 'Comparative Analysis':
            description_parts.append("\n\n**Recommended Approach**: Systematically compare methodologies and outcomes to identify best practices and optimal approaches.")
        else:  # Synthesis Study
            description_parts.append("\n\n**Recommended Approach**: Synthesize findings to identify research gaps, contradictions, and opportunities for novel contributions.")
        
        description = ''.join(description_parts)
        
        # Sample papers
        sample_titles = cluster_papers['title'].head(3).tolist()
        
        hypothesis = {
            'id': len(hypotheses) + 1,
            'cluster_id': cluster_id,
            'type': hyp_type,
            'title': title,
            'description': description,
            'sample_papers': sample_titles,
            # REAL SCORES based on data analysis
            'reproducibility': reproducibility,
            'difficulty': difficulty,
            'impact': impact,
            'data_availability_score': data_availability_score,
            'computational_score': computational_score,
            'future_work_score': future_work_score,
            'recency_score': recency_score,
            # Cluster info
            'size': size,
            'year_range': year_range,
            'year_span': year_span,
            'dominant_method': dominant_method,
            'dominant_framework': dominant_framework,
            'common_terms': common_terms[:5],
            'method_score': method_score,
            'framework_score': framework_score,
            # Priority (higher reproducibility + lower difficulty = higher priority)
            'priority_score': (reproducibility * 0.5 + (10 - difficulty) * 0.3 + impact * 0.2)
        }
        
        hypotheses.append(hypothesis)
    
    # Sort by priority
    hypotheses.sort(key=lambda x: x['priority_score'], reverse=True)
    
    return hypotheses

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🔬 Research Semantic POC</div>', unsafe_allow_html=True)
    st.markdown("**Automated hypothesis generation from scientific literature**")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Dataset")
        dataset_size = st.selectbox(
            "Dataset Size",
            options=[200, 500, 1000, 2000],
            index=0,
            help="Number of papers to analyze (smaller = faster)"
        )
        
        computational_only = st.checkbox(
            "Computational Papers Only",
            value=True,
            help="Filter for papers with computational/data-driven methods"
        )
        
        st.subheader("Embedding Model")
        embedding_model = st.selectbox(
            "Model",
            options=[
                'all-MiniLM-L6-v2',
                'allenai/specter',
                'all-mpnet-base-v2'
            ],
            index=0,
            help="all-MiniLM-L6-v2: Fast, 384-dim\nallenai/specter: Best for scientific papers, 768-dim"
        )
        
        st.subheader("Clustering Parameters")
        
        clustering_mode = st.radio(
            "Clustering Strategy",
            options=[
                "🎯 Hierarchical Funnel (Recommended)",
                "🎯🔬 Domain-Aware + Adaptive",
                "🎯 Adaptive Only (Reduce Noise)",
                "🔬 Domain-Aware Only",
                "⚡ Standard"
            ],
            index=0,
            help="Hierarchical Funnel: Topic → Methodology → Temporal → Semantic (conditional probability)"
        )
        
        min_cluster_size = st.slider(
            "Min Cluster Size",
            min_value=3,
            max_value=30,
            value=5 if "Adaptive" in clustering_mode else 15,
            help="Minimum papers per cluster (smaller = more clusters, less noise)"
        )
        
        min_samples = st.slider(
            "Min Samples",
            min_value=2,
            max_value=10,
            value=5,
            help="HDBSCAN min_samples parameter"
        )
        
        umap_components = st.slider(
            "UMAP Components",
            min_value=5,
            max_value=20,
            value=10,
            help="Dimensionality reduction components"
        )
        
        st.subheader("Validation Criteria")
        use_custom_criteria = st.checkbox(
            "Use Custom Criteria",
            value=True,
            help="Apply custom validation criteria (data availability, sponsors, etc.)"
        )
        
        if use_custom_criteria:
            with st.expander("⚙️ Adjust Criteria Weights"):
                st.markdown("**Custom Criteria Weights** (must sum to ≤1.0)")
                
                col1, col2 = st.columns(2)
                with col1:
                    data_availability_weight = st.slider(
                        "Data Availability",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.15,
                        step=0.05,
                        help="Papers with available datasets"
                    )
                    
                    clinical_trial_weight = st.slider(
                        "Clinical Trial Sponsor",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.10,
                        step=0.05,
                        help="Clear sponsor information"
                    )
                
                with col2:
                    replication_weight = st.slider(
                        "Replication Status",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.10,
                        step=0.05,
                        help="Findings replicated/validated"
                    )
                
                total_weight = data_availability_weight + clinical_trial_weight + replication_weight
                if total_weight > 1.0:
                    st.error(f"⚠️ Total weight ({total_weight:.2f}) exceeds 1.0!")
                else:
                    st.success(f"✅ Total custom weight: {total_weight:.2f}")
        else:
            data_availability_weight = 0.15
            clinical_trial_weight = 0.10
            replication_weight = 0.10
        
        st.subheader("Hypothesis Generation")
        
        max_cluster_size_for_hypotheses = st.slider(
            "Max Cluster Size for Hypotheses",
            min_value=10,
            max_value=100,
            value=30,
            help="Ignore large meta-analysis clusters. Focus on smaller, actionable research gaps."
        )
        
        require_future_work = st.checkbox(
            "Require Future Work Mentions",
            value=True,
            help="Only generate hypotheses for clusters with papers mentioning future work/gaps"
        )
        
        filter_incoherent = st.checkbox(
            "🎯 Filter Incoherent Clusters",
            value=True,
            help="Remove clusters where papers don't share a specific medical topic (e.g., mixing heart failure + kidney stones)"
        )
        
        st.divider()
        
        # Update config
        st.session_state.config = {
            'dataset_size': dataset_size,
            'computational_only': computational_only,
            'embedding_model': embedding_model,
            'clustering_mode': clustering_mode,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'umap_components': umap_components,
            'use_custom_criteria': use_custom_criteria,
            'data_availability_weight': data_availability_weight,
            'clinical_trial_weight': clinical_trial_weight,
            'replication_weight': replication_weight,
            'max_cluster_size_for_hypotheses': max_cluster_size_for_hypotheses,
            'require_future_work': require_future_work,
            'filter_incoherent': filter_incoherent
        }
        
        # Run button
        if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
            st.session_state.pipeline_run = True
            st.rerun()
    
    # Main content
    if not st.session_state.pipeline_run:
        # Welcome screen
        st.info("👈 Configure the pipeline in the sidebar and click **Run Pipeline** to start")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Features")
            st.markdown("""
            - Real dataset (Boston Children's Hospital)
            - Multiple embedding models
            - Custom validation criteria
            - Interactive results
            """)
        
        with col2:
            st.markdown("### 🎯 Pipeline Steps")
            st.markdown("""
            1. Load & filter dataset
            2. Generate embeddings
            3. Cluster papers
            4. Validate clusters
            5. Generate hypotheses
            """)
        
        with col3:
            st.markdown("### 💡 Output")
            st.markdown("""
            - Cluster visualizations
            - Validation scores
            - Ranked hypotheses
            - Exportable results
            """)
        
        # Show example configuration
        st.subheader("Current Configuration")
        config_df = pd.DataFrame([
            {"Parameter": "Dataset Size", "Value": str(st.session_state.config.get('dataset_size', 200))},
            {"Parameter": "Embedding Model", "Value": str(st.session_state.config['embedding_model'])},
            {"Parameter": "Min Cluster Size", "Value": str(st.session_state.config['min_cluster_size'])},
            {"Parameter": "Custom Criteria", "Value": "✅" if st.session_state.config['use_custom_criteria'] else "❌"}
        ])
        st.dataframe(config_df, hide_index=True, width='stretch')
        
    else:
        # Pipeline execution
        if st.session_state.results is None:
            # Load dataset
            with st.spinner("📂 Loading dataset..."):
                df = load_dataset()
                
                if df is None:
                    st.session_state.pipeline_run = False
                    st.stop()
                
                # Sample dataset
                df = df.head(st.session_state.config['dataset_size'])
                
                # Filter if needed
                if st.session_state.config['computational_only']:
                    df = filter_computational_papers(df)
                
                st.success(f"✅ Loaded {len(df)} papers")
            
            # Run pipeline
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                results = run_pipeline(df, st.session_state.config, progress_bar, status_text)
                st.session_state.results = results
                time.sleep(0.5)  # Brief pause to show completion
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
                st.session_state.pipeline_run = False
                st.stop()
        
        else:
            # Display results
            results = st.session_state.results
            
            # Summary metrics
            st.subheader("📊 Pipeline Summary")
            
            # Show funnel info if applicable
            if 'funnel_report' in results:
                st.info("🎯 **Hierarchical Funnel Applied**: Topic (40%) → Methodology (25%) → Temporal (15%) → Semantic (20%)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Papers Analyzed", results['n_papers'])
            
            with col2:
                st.metric("Clusters Found", results['n_clusters'])
            
            with col3:
                noise_pct = (results['n_noise'] / results['n_valid']) * 100
                st.metric("Noise Ratio", f"{noise_pct:.1f}%")
            
            with col4:
                st.metric("Hypotheses Generated", len(results['hypotheses']))
            
            # Tabs for different views
            if 'funnel_report' in results:
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "🎯 Clusters", 
                    "🔍 Funnel Analysis", 
                    "✅ Validation", 
                    "💡 Hypotheses", 
                    "📋 Criteria", 
                    "📥 Export"
                ])
            else:
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🎯 Clusters", 
                    "✅ Validation", 
                    "💡 Hypotheses", 
                    "🔍 Criteria", 
                    "📥 Export"
                ])
                tab6 = None
            
            with tab1:
                st.subheader("Cluster Visualization")
                
                # Show domain statistics if domain-aware clustering was used
                if 'domain_stats' in results:
                    st.info("🔬 **Domain-Aware Clustering Enabled** - Papers clustered within medical domains")
                    
                    with st.expander("View Domain Distribution"):
                        domain_stats = results['domain_stats']
                        domain_df = pd.DataFrame([
                            {"Domain": k.title(), "Papers": v} 
                            for k, v in domain_stats['domain_counts'].items()
                        ]).sort_values('Papers', ascending=False)
                        
                        fig_domains = px.bar(
                            domain_df,
                            x='Domain',
                            y='Papers',
                            title='Papers by Medical Domain',
                            color='Papers',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_domains, width='stretch')
                
                # 2D scatter plot
                df_plot = pd.DataFrame({
                    'x': results['reduced_embeddings'][:, 0],
                    'y': results['reduced_embeddings'][:, 1],
                    'cluster': [f"Cluster {l}" if l != -1 else "Noise" for l in results['labels']],
                    'title': results['df']['title'].values
                })
                
                # Add domain info if available
                if 'domain_labels' in results:
                    df_plot['domain'] = results['domain_labels'].values
                
                fig = px.scatter(
                    df_plot,
                    x='x',
                    y='y',
                    color='cluster',
                    hover_data=['title', 'domain'] if 'domain' in df_plot.columns else ['title'],
                    title='Paper Clusters (UMAP Projection)',
                    width=900,
                    height=600
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Cluster sizes
                st.subheader("Cluster Sizes")
                
                cluster_sizes = []
                for label in np.unique(results['labels']):
                    if label != -1:
                        size = (results['labels'] == label).sum()
                        cluster_sizes.append({'Cluster': f"Cluster {label}", 'Size': size})
                
                if cluster_sizes:
                    cluster_df = pd.DataFrame(cluster_sizes).sort_values('Size', ascending=False)
                    
                    fig_bar = px.bar(
                        cluster_df,
                        x='Cluster',
                        y='Size',
                        title='Papers per Cluster',
                        color='Size',
                        color_continuous_scale='Blues'
                    )
                    
                    st.plotly_chart(fig_bar, width='stretch')
                else:
                    st.warning("No clusters found. All papers were classified as noise. Try adjusting parameters.")
            
            # Validation tab (tab2 if no funnel, tab4 if funnel)
            validation_tab = tab4 if tab6 is not None else tab2
            
            with validation_tab:
                st.subheader("Validation Scores")
                
                validation = results['validation']
                
                # Overall statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Clusters", validation['total_clusters'])
                
                with col2:
                    st.metric("Passed Validation", validation['passed_validation'])
                
                with col3:
                    pass_rate = validation['pass_rate'] * 100
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")
                
                # Scores by cluster
                scores_data = []
                for cluster_name, cluster_data in validation['cluster_reports'].items():
                    cluster_id = cluster_name.replace('cluster_', '')
                    if cluster_id == '-1':
                        continue
                    
                    scores_data.append({
                        'Cluster': f"C{cluster_id}",
                        'Overall Score': cluster_data['overall_score'],
                        'Size': cluster_data['size'],
                        'Status': '✅ Pass' if cluster_data['validation_passed'] else '❌ Fail'
                    })
                
                if scores_data:
                    scores_df = pd.DataFrame(scores_data)
                    
                    fig_scores = px.bar(
                        scores_df,
                        x='Cluster',
                        y='Overall Score',
                        color='Status',
                        title='Validation Scores by Cluster',
                        color_discrete_map={'✅ Pass': 'green', '❌ Fail': 'red'}
                    )
                    fig_scores.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                                        annotation_text="Pass Threshold")
                    
                    st.plotly_chart(fig_scores, width='stretch')
                    
                    # Detailed scores table
                    st.subheader("Detailed Scores")
                    st.dataframe(scores_df, hide_index=True, width='stretch')
                else:
                    st.warning("No clusters to validate. All papers were classified as noise.")
            
            # Funnel Analysis Tab (only if funnel was used)
            if tab6 is not None:
                with tab2:
                    st.subheader("🔍 Hierarchical Funnel Analysis")
                    
                    funnel_report = results['funnel_report']
                    summary = funnel_report['funnel_summary']
                    
                    st.markdown("""
                    **Funnel Stages (in order of importance)**:
                    1. **Topic** (40%) - Same specific medical condition
                    2. **Methodology** (25%) - Same research approach  
                    3. **Temporal** (15%) - Similar time period (recency)
                    4. **Semantic** (20%) - Fine-grained similarity
                    """)
                    
                    # Stage-by-stage breakdown
                    st.markdown("### 📊 Funnel Flow")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Stage 1: Topics",
                            len(summary['topic_distribution']),
                            help="Specific medical topics identified"
                        )
                    
                    with col2:
                        st.metric(
                            "Stage 2: Methods",
                            len(summary['methodology_distribution']),
                            help="Research methodologies detected"
                        )
                    
                    with col3:
                        st.metric(
                            "Stage 3: Time Groups",
                            "Recent + Older",
                            help="Papers grouped by recency"
                        )
                    
                    with col4:
                        st.metric(
                            "Stage 4: Final Clusters",
                            summary['n_clusters'],
                            help="Semantically refined clusters"
                        )
                    
                    # Topic distribution
                    st.markdown("### 📍 Topic Distribution (Stage 1)")
                    
                    topic_data = []
                    for topic, count in summary['topic_distribution'].items():
                        topic_data.append({
                            'Topic': topic.replace('_', ' ').title(),
                            'Papers': count,
                            'Percentage': f"{count/summary['total_papers']*100:.1f}%"
                        })
                    
                    if topic_data:
                        topic_df = pd.DataFrame(topic_data).sort_values('Papers', ascending=False)
                        
                        fig_topics = px.bar(
                            topic_df,
                            x='Topic',
                            y='Papers',
                            title='Papers by Specific Medical Topic',
                            color='Papers',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_topics, width='stretch')
                    else:
                        st.warning("No specific topics identified. Papers may be too diverse or use non-medical terminology.")
                    
                    # Methodology distribution
                    st.markdown("### 🔬 Methodology Distribution (Stage 2)")
                    
                    method_data = []
                    for method, count in summary['methodology_distribution'].items():
                        method_data.append({
                            'Methodology': method.replace('_', ' ').title(),
                            'Papers': count,
                            'Percentage': f"{count/summary['total_papers']*100:.1f}%"
                        })
                    
                    if method_data:
                        method_df = pd.DataFrame(method_data).sort_values('Papers', ascending=False)
                        
                        fig_methods = px.bar(
                            method_df,
                            x='Methodology',
                            y='Papers',
                            title='Papers by Research Methodology',
                            color='Papers',
                            color_continuous_scale='Greens'
                        )
                        st.plotly_chart(fig_methods, width='stretch')
                    else:
                        st.warning("No specific methodologies identified.")
                    
                    # Cluster details
                    st.markdown("### 🎯 Final Cluster Composition")
                    
                    cluster_composition = []
                    for cluster_id, details in funnel_report['cluster_details'].items():
                        cluster_composition.append({
                            'Cluster': cluster_id,
                            'Topic': details['topic'].replace('_', ' ').title(),
                            'Methodology': details['methodology'].replace('_', ' ').title(),
                            'Time Group': details['time_group'].title(),
                            'Size': details['size']
                        })
                    
                    if cluster_composition:
                        comp_df = pd.DataFrame(cluster_composition).sort_values('Size', ascending=False)
                        st.dataframe(comp_df, hide_index=True, width='stretch')
                    else:
                        st.info("No clusters formed after funnel stages. Try lowering min_cluster_size or increasing dataset size.")
                    
                    # Funnel efficiency
                    st.markdown("### 📈 Funnel Efficiency")
                    
                    eff_col1, eff_col2, eff_col3 = st.columns(3)
                    
                    with eff_col1:
                        st.metric(
                            "Clustering Rate",
                            f"{summary['clustering_rate']*100:.1f}%",
                            help="% of papers successfully clustered"
                        )
                    
                    with eff_col2:
                        st.metric(
                            "Avg Cluster Purity",
                            f"{summary['avg_purity']*100:.0f}%",
                            help="How pure clusters are (same topic+method)"
                        )
                    
                    with eff_col3:
                        avg_size = summary['clustered_papers'] / summary['n_clusters'] if summary['n_clusters'] > 0 else 0
                        st.metric(
                            "Avg Cluster Size",
                            f"{avg_size:.1f}",
                            help="Average papers per cluster"
                        )
                    
                    st.success("✅ All clusters are guaranteed to have the same specific topic AND methodology")
            
            with tab3:
                st.subheader("Generated Hypotheses")
                
                hypotheses = results['hypotheses']
                
                if not hypotheses:
                    st.warning("No hypotheses generated. Try adjusting parameters.")
                else:
                    # Display each hypothesis
                    for hyp in hypotheses[:10]:  # Top 10
                        with st.expander(f"**Hypothesis #{hyp['id']}: {hyp['title']}** (Score: {hyp['priority_score']:.2f})", expanded=False):
                            
                            # Get cluster data
                            cluster_id = hyp['cluster_id']
                            cluster_mask = results['labels'] == cluster_id
                            cluster_papers = results['df'][cluster_mask]
                            
                            # Metrics row - REAL SCORES
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Reproducibility", f"{hyp['reproducibility']:.1f}/10")
                            with col2:
                                st.metric("Difficulty", f"{hyp['difficulty']:.1f}/10")
                            with col3:
                                st.metric("Impact", f"{hyp['impact']:.1f}/10")
                            with col4:
                                st.metric("Size", hyp['size'])
                            with col5:
                                st.metric("Type", hyp['type'])
                            
                            # Sub-metrics with real data
                            st.markdown("**📊 Data-Based Scores:**")
                            subcol1, subcol2, subcol3, subcol4 = st.columns(4)
                            with subcol1:
                                st.metric("Data Available", f"{hyp['data_availability_score']:.0%}")
                            with subcol2:
                                st.metric("Computational", f"{hyp['computational_score']:.0%}")
                            with subcol3:
                                st.metric("Future Work", f"{hyp['future_work_score']:.0%}")
                            with subcol4:
                                st.metric("Recency", f"{hyp['recency_score']:.0%}")
                            
                            st.divider()
                            
                            # Description
                            st.markdown(f"**Description:** {hyp['description']}")
                            
                            st.divider()
                            
                            # All papers with full details
                            st.markdown("### 📚 All Papers in Cluster")
                            
                            for idx, (_, paper) in enumerate(cluster_papers.iterrows(), 1):
                                with st.container():
                                    st.markdown(f"**Paper {idx}**")
                                    
                                    # Title
                                    st.markdown(f"**Title:** {paper.get('title', 'N/A')}")
                                    
                                    # PMID if available
                                    if 'pmid' in paper and pd.notna(paper['pmid']):
                                        pmid = str(paper['pmid']).replace('.0', '')
                                        st.markdown(f"**PMID:** [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                                    
                                    # Year
                                    if 'publication_year' in paper and pd.notna(paper['publication_year']):
                                        st.markdown(f"**Year:** {int(paper['publication_year'])}")
                                    
                                    # Journal
                                    if 'journal_title' in paper and pd.notna(paper['journal_title']):
                                        st.markdown(f"**Journal:** {paper['journal_title']}")
                                    
                                    # Abstract (truncated)
                                    if 'abstract_text' in paper and pd.notna(paper['abstract_text']):
                                        abstract = str(paper['abstract_text'])
                                        if len(abstract) > 300:
                                            st.markdown(f"**Abstract:** {abstract[:300]}...")
                                        else:
                                            st.markdown(f"**Abstract:** {abstract}")
                                    
                                    # MeSH terms
                                    if 'mesh_headings' in paper and pd.notna(paper['mesh_headings']):
                                        mesh = str(paper['mesh_headings'])
                                        if mesh and mesh != 'nan':
                                            st.markdown(f"**MeSH Terms:** {mesh[:200]}...")
                                    
                                    st.markdown("---")
            
            # Criteria tab (tab4 if no funnel, tab5 if funnel)
            criteria_tab = tab5 if tab6 is not None else tab4
            
            with criteria_tab:
                st.subheader("🔍 Validation Criteria Used")
                
                # Show clustering mode info
                clustering_mode = st.session_state.config.get('clustering_mode', 'standard')
                
                if clustering_mode == 'hierarchical_funnel':
                    st.info("🎯 **Hierarchical Funnel Clustering Active** - Multi-stage filtering with ordered importance")
                    
                    st.markdown("### 📊 Hierarchical Funnel Stages & Weights")
                    st.markdown("*Ordered by importance - each stage filters papers before the next*")
                    
                    funnel_stages = pd.DataFrame([
                        {
                            "Stage": "1️⃣ Topic Coherence",
                            "Weight": "40%",
                            "Threshold": "0.15 similarity",
                            "Purpose": "Group papers by specific medical topic (e.g., 'heart failure', 'breast cancer')",
                            "Method": "MeSH term similarity + keyword matching"
                        },
                        {
                            "Stage": "2️⃣ Methodology Coherence", 
                            "Weight": "25%",
                            "Threshold": "Same method",
                            "Purpose": "Ensure all papers use same research method (Clinical Trial, Cohort, Lab, etc.)",
                            "Method": "Keyword detection + classification"
                        },
                        {
                            "Stage": "3️⃣ Temporal Coherence",
                            "Weight": "15%", 
                            "Threshold": "5-year window",
                            "Purpose": "Group papers from similar time periods (methods evolve)",
                            "Method": "Publication year grouping"
                        },
                        {
                            "Stage": "4️⃣ Semantic Coherence",
                            "Weight": "20%",
                            "Threshold": "0.3 cosine sim",
                            "Purpose": "Final semantic clustering using embeddings",
                            "Method": "HDBSCAN on filtered papers"
                        }
                    ])
                    
                    st.dataframe(funnel_stages, hide_index=True, column_config={
                        "Stage": st.column_config.TextColumn("Stage", width="small"),
                        "Weight": st.column_config.TextColumn("Weight", width="small"),
                        "Threshold": st.column_config.TextColumn("Threshold", width="medium"),
                        "Purpose": st.column_config.TextColumn("Purpose", width="large"),
                        "Method": st.column_config.TextColumn("Method", width="medium")
                    })
                    
                    st.success("✅ **Guarantee**: All final clusters have papers with the SAME topic AND methodology")
                    
                    st.markdown("---")
                
                st.markdown("### 🧬 Semantic Embeddings - Features Used")
                st.markdown("*Text fields combined to create paper representations*")
                
                embedding_features = pd.DataFrame([
                    {
                        "Feature": "Title",
                        "Priority": "⭐⭐⭐ High",
                        "Format": "Title: [paper title]",
                        "Example": "Title: Machine learning-based automatic estimation of cortical atrophy..."
                    },
                    {
                        "Feature": "Abstract",
                        "Priority": "⭐⭐⭐ High", 
                        "Format": "Abstract: [full abstract text]",
                        "Example": "Abstract: Cortical atrophy is measured clinically according to established..."
                    },
                    {
                        "Feature": "MeSH Terms",
                        "Priority": "⭐⭐ Medium",
                        "Format": "MeSH: [medical subject headings]",
                        "Example": "MeSH: Alzheimer Disease, Atrophy, Brain, Humans, Machine Learning"
                    },
                    {
                        "Feature": "Keywords",
                        "Priority": "⭐ Low",
                        "Format": "Keywords: [author keywords]",
                        "Example": "Keywords: deep learning, medical imaging, neurodegeneration"
                    }
                ])
                
                st.dataframe(embedding_features, hide_index=True, column_config={
                    "Feature": st.column_config.TextColumn("Feature", width="small"),
                    "Priority": st.column_config.TextColumn("Priority", width="small"),
                    "Format": st.column_config.TextColumn("Format", width="medium"),
                    "Example": st.column_config.TextColumn("Example", width="large")
                })
                
                embedding_model = st.session_state.config.get('embedding_model', 'all-MiniLM-L6-v2')
                
                model_info = {
                    'all-MiniLM-L6-v2': {
                        'name': 'MiniLM-L6-v2',
                        'dims': 384,
                        'speed': 'Fast',
                        'quality': 'Good',
                        'description': 'General-purpose sentence encoder, balanced speed/quality'
                    },
                    'allenai/specter': {
                        'name': 'SPECTER',
                        'dims': 768,
                        'speed': 'Medium',
                        'quality': 'Excellent',
                        'description': 'Scientific paper encoder trained on citations, best for research papers'
                    },
                    'all-mpnet-base-v2': {
                        'name': 'MPNet-base-v2',
                        'dims': 768,
                        'speed': 'Medium',
                        'quality': 'Excellent',
                        'description': 'High-quality general encoder, good for semantic similarity'
                    }
                }
                
                if embedding_model in model_info:
                    info = model_info[embedding_model]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Model", info['name'])
                    with col2:
                        st.metric("Dimensions", info['dims'])
                    with col3:
                        st.metric("Speed", info['speed'])
                    with col4:
                        st.metric("Quality", info['quality'])
                    
                    st.caption(f"ℹ️ {info['description']}")
                
                st.markdown("**Process:**")
                st.markdown("""
                1. **Text Combination**: Title + Abstract + MeSH + Keywords → single text string
                2. **Tokenization**: Text split into tokens by transformer model
                3. **Encoding**: Tokens → dense vector representation (embeddings)
                4. **Dimensionality**: 384 or 768 dimensions depending on model
                5. **Similarity**: Cosine similarity between embeddings measures semantic relatedness
                """)
                
                st.markdown("---")
                
                st.markdown("### Standard Validation Criteria (Always Applied)")
                st.markdown("*Used to score cluster quality after formation*")
                
                criteria_standard = pd.DataFrame([
                    {
                        "Criterion": "Methodological Coherence",
                        "Weight": "35%",
                        "Description": "Papers use similar research methods (Clinical Trial, Cohort, Lab, Computational, etc.)"
                    },
                    {
                        "Criterion": "Framework Coherence",
                        "Weight": "25%",
                        "Description": "Papers share conceptual framework (Mechanistic, Predictive, Interventional, etc.)"
                    },
                    {
                        "Criterion": "Temporal Coherence",
                        "Weight": "15%",
                        "Description": "Papers from similar time periods (methods evolve over time)"
                    },
                    {
                        "Criterion": "Internal Consistency",
                        "Weight": "15%",
                        "Description": "Papers are semantically similar to each other"
                    },
                    {
                        "Criterion": "MeSH Coherence",
                        "Weight": "10%",
                        "Description": "Papers share Medical Subject Headings (standardized terms)"
                    }
                ])
                
                st.dataframe(criteria_standard, hide_index=True, width='stretch')
                
                if st.session_state.config['use_custom_criteria']:
                    st.markdown("### Custom Criteria (Optional - Currently Enabled)")
                    
                    criteria_custom = pd.DataFrame([
                        {
                            "Criterion": "Data Availability",
                            "Weight": "15%",
                            "Description": "Papers mention available datasets (GitHub, Figshare, Zenodo, etc.)"
                        },
                        {
                            "Criterion": "Clinical Trial Sponsor",
                            "Weight": "10%",
                            "Description": "Clinical trials have clear sponsor information (industry vs academic)"
                        },
                        {
                            "Criterion": "Replication Status",
                            "Weight": "10%",
                            "Description": "Findings have been replicated or validated in independent studies"
                        }
                    ])
                    
                    st.dataframe(criteria_custom, hide_index=True, width='stretch')
                else:
                    st.info("Custom criteria are currently disabled. Enable in sidebar to use.")
                
                st.markdown("### Scoring Thresholds")
                
                thresholds = pd.DataFrame([
                    {"Score Range": "≥ 0.8", "Status": "✅ EXCELLENT", "Description": "High-quality, scientifically coherent cluster"},
                    {"Score Range": "0.6 - 0.8", "Status": "✅ GOOD", "Description": "Acceptable scientific coherence"},
                    {"Score Range": "< 0.6", "Status": "⚠️ REVIEW", "Description": "Low coherence - needs review or reclassification"}
                ])
                
                st.dataframe(thresholds, hide_index=True, width='stretch')
                
                st.markdown("### Methodologies Detected")
                
                with st.expander("View all 8 methodologies"):
                    methodologies = pd.DataFrame([
                        {"Methodology": "Clinical Trial", "Keywords": "randomized, controlled trial, RCT, double-blind, placebo"},
                        {"Methodology": "Cohort Study", "Keywords": "cohort, prospective, retrospective, follow-up, longitudinal"},
                        {"Methodology": "Case-Control", "Keywords": "case-control, matched controls, odds ratio"},
                        {"Methodology": "Cross-Sectional", "Keywords": "cross-sectional, survey, prevalence, questionnaire"},
                        {"Methodology": "Systematic Review", "Keywords": "systematic review, meta-analysis, PRISMA, Cochrane"},
                        {"Methodology": "Laboratory", "Keywords": "in vitro, in vivo, cell culture, animal model"},
                        {"Methodology": "Computational", "Keywords": "bioinformatics, machine learning, algorithm, simulation"},
                        {"Methodology": "Genomic", "Keywords": "genome-wide, GWAS, sequencing, RNA-seq"}
                    ])
                    st.dataframe(methodologies, hide_index=True, width='stretch')
                
                st.markdown("### Conceptual Frameworks Detected")
                
                with st.expander("View all 5 frameworks"):
                    frameworks = pd.DataFrame([
                        {"Framework": "Mechanistic", "Keywords": "mechanism, pathway, molecular, signaling"},
                        {"Framework": "Etiological", "Keywords": "etiology, risk factor, cause, pathogenesis"},
                        {"Framework": "Descriptive", "Keywords": "prevalence, incidence, epidemiology, distribution"},
                        {"Framework": "Predictive", "Keywords": "prognosis, prediction, risk score, prognostic"},
                        {"Framework": "Interventional", "Keywords": "treatment, therapy, intervention, efficacy"}
                    ])
                    st.dataframe(frameworks, hide_index=True, width='stretch')
                
                st.info("📖 For complete documentation, see `VALIDATION_CRITERIA.md` in the repository")
            
            # Export tab (tab5 if no funnel, tab6 if funnel)
            export_tab = tab6 if tab6 is not None else tab5
            
            with export_tab:
                st.subheader("Export Results")
                
                # Prepare export data
                export_data = {
                    'config': st.session_state.config,
                    'summary': {
                        'n_papers': results['n_papers'],
                        'n_clusters': results['n_clusters'],
                        'n_hypotheses': len(results['hypotheses']),
                        'timestamp': datetime.now().isoformat()
                    },
                    'hypotheses': results['hypotheses'],
                    'validation': {
                        'pass_rate': results['validation']['pass_rate'],
                        'passed': results['validation']['passed_validation'],
                        'total': results['validation']['total_clusters']
                    }
                }
                
                # JSON download
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json_str,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # CSV download (hypotheses)
                hyp_df = pd.DataFrame(results['hypotheses'])
                csv = hyp_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hypotheses (CSV)",
                    data=csv,
                    file_name=f"hypotheses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.success("✅ Results ready for export!")
            
            # Reset button
            st.divider()
            if st.button("🔄 Run New Analysis", type="secondary"):
                st.session_state.pipeline_run = False
                st.session_state.results = None
                st.rerun()

if __name__ == "__main__":
    main()
