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
from clustering import SemanticClusterer
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
    reduced = clusterer.reduce_dimensions(
        valid_embeddings,
        n_components=config['umap_components']
    )
    labels = clusterer.cluster_hdbscan(
        reduced,
        min_cluster_size=config['min_cluster_size'],
        min_samples=config['min_samples']
    )
    
    results['labels'] = labels
    results['reduced_embeddings'] = reduced
    results['n_clusters'] = len([l for l in np.unique(labels) if l != -1])
    results['n_noise'] = (labels == -1).sum()
    
    # Stage 3: Validation
    status_text.text("✅ Stage 3/5: Validating clusters...")
    progress_bar.progress(0.6)
    
    validator = ClassificationValidator()
    validation_results = validator.validate_all_clusters(valid_df, labels)
    results['validation'] = validation_results
    
    # Stage 4: Custom Criteria
    if config['use_custom_criteria']:
        status_text.text("🔍 Stage 4/5: Applying custom criteria...")
        progress_bar.progress(0.8)
        
        custom_validator = CustomCriteriaValidator()
        custom_validator.add_criterion(DataAvailabilityCriterion(), weight=0.15)
        custom_validator.add_criterion(ClinicalTrialSponsorCriterion(), weight=0.10)
        custom_validator.add_criterion(ReplicationStatusCriterion(), weight=0.10)
        
        custom_results = custom_validator.evaluate_all_clusters(valid_df, labels)
        results['custom_validation'] = custom_results
    
    # Stage 5: Generate Hypotheses
    status_text.text("💡 Stage 5/5: Generating hypotheses...")
    progress_bar.progress(0.9)
    
    hypotheses = generate_hypotheses(valid_df, labels, validation_results)
    results['hypotheses'] = hypotheses
    results['df'] = valid_df
    
    progress_bar.progress(1.0)
    status_text.text("✅ Pipeline completed!")
    
    return results

def generate_hypotheses(df, labels, validation_results):
    """Generate hypotheses from clusters"""
    
    hypotheses = []
    
    for cluster_name, cluster_data in validation_results['cluster_reports'].items():
        cluster_id = int(cluster_name.replace('cluster_', ''))
        
        if cluster_id == -1:  # Skip noise
            continue
        
        cluster_mask = labels == cluster_id
        cluster_papers = df[cluster_mask]
        
        if len(cluster_papers) < 5:
            continue
        
        # Calculate scores
        reproducibility = cluster_data['overall_score']
        size = len(cluster_papers)
        
        # Generate hypothesis
        sample_titles = cluster_papers['title'].head(3).tolist()
        
        hypothesis = {
            'id': len(hypotheses) + 1,
            'cluster_id': cluster_id,
            'type': 'Meta-Analysis' if size > 20 else 'ML Application',
            'title': f"Research Opportunity in Cluster {cluster_id}",
            'description': f"Cluster with {size} papers showing {reproducibility:.2f} validation score",
            'sample_papers': sample_titles,
            'reproducibility': reproducibility,
            'size': size,
            'priority_score': (reproducibility * 0.4 + min(size/50, 1) * 0.3 + 0.3) * 10
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
        min_cluster_size = st.slider(
            "Min Cluster Size",
            min_value=5,
            max_value=30,
            value=15,
            help="Minimum papers per cluster"
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
        
        st.subheader("Validation")
        use_custom_criteria = st.checkbox(
            "Use Custom Criteria",
            value=True,
            help="Apply custom validation criteria (data availability, sponsors, etc.)"
        )
        
        st.divider()
        
        # Update config
        st.session_state.config = {
            'dataset_size': dataset_size,
            'computational_only': computational_only,
            'embedding_model': embedding_model,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'umap_components': umap_components,
            'use_custom_criteria': use_custom_criteria
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
            {"Parameter": "Dataset Size", "Value": st.session_state.config.get('dataset_size', 200)},
            {"Parameter": "Embedding Model", "Value": st.session_state.config['embedding_model']},
            {"Parameter": "Min Cluster Size", "Value": st.session_state.config['min_cluster_size']},
            {"Parameter": "Custom Criteria", "Value": "✅" if st.session_state.config['use_custom_criteria'] else "❌"}
        ])
        st.dataframe(config_df, hide_index=True, use_container_width=True)
        
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
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Clusters", "✅ Validation", "💡 Hypotheses", "📥 Export"])
            
            with tab1:
                st.subheader("Cluster Visualization")
                
                # 2D scatter plot
                df_plot = pd.DataFrame({
                    'x': results['reduced_embeddings'][:, 0],
                    'y': results['reduced_embeddings'][:, 1],
                    'cluster': [f"Cluster {l}" if l != -1 else "Noise" for l in results['labels']],
                    'title': results['df']['title'].values
                })
                
                fig = px.scatter(
                    df_plot,
                    x='x',
                    y='y',
                    color='cluster',
                    hover_data=['title'],
                    title='Paper Clusters (UMAP Projection)',
                    width=900,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster sizes
                st.subheader("Cluster Sizes")
                
                cluster_sizes = []
                for label in np.unique(results['labels']):
                    if label != -1:
                        size = (results['labels'] == label).sum()
                        cluster_sizes.append({'Cluster': f"Cluster {label}", 'Size': size})
                
                cluster_df = pd.DataFrame(cluster_sizes).sort_values('Size', ascending=False)
                
                fig_bar = px.bar(
                    cluster_df,
                    x='Cluster',
                    y='Size',
                    title='Papers per Cluster',
                    color='Size',
                    color_continuous_scale='Blues'
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with tab2:
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
                        'Status': '✅ Pass' if cluster_data['passed'] else '❌ Fail'
                    })
                
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
                
                st.plotly_chart(fig_scores, use_container_width=True)
                
                # Detailed scores table
                st.subheader("Detailed Scores")
                st.dataframe(scores_df, hide_index=True, use_container_width=True)
            
            with tab3:
                st.subheader("Generated Hypotheses")
                
                hypotheses = results['hypotheses']
                
                if not hypotheses:
                    st.warning("No hypotheses generated. Try adjusting parameters.")
                else:
                    # Display each hypothesis
                    for hyp in hypotheses[:10]:  # Top 10
                        with st.expander(f"**Hypothesis #{hyp['id']}: {hyp['title']}** (Score: {hyp['priority_score']:.2f})"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Description:** {hyp['description']}")
                                st.markdown(f"**Type:** {hyp['type']}")
                                st.markdown(f"**Cluster:** {hyp['cluster_id']}")
                                
                                st.markdown("**Sample Papers:**")
                                for i, title in enumerate(hyp['sample_papers'], 1):
                                    st.markdown(f"{i}. {title}")
                            
                            with col2:
                                st.metric("Priority Score", f"{hyp['priority_score']:.2f}/10")
                                st.metric("Reproducibility", f"{hyp['reproducibility']:.2f}")
                                st.metric("Cluster Size", hyp['size'])
            
            with tab4:
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
