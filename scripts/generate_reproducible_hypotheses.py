#!/usr/bin/env python3
"""
Generate Reproducible Hypotheses
Focus on hypotheses that can be tested with existing datasets
without need for clinical trials or lab experiments
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from embeddings import PaperEmbedder
from clustering import SemanticClusterer
from extraction import ClassificationValidator, CustomCriteriaValidator
from extraction.custom_criteria import DataAvailabilityCriterion


def filter_computational_papers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter papers that are likely computational/data-driven"""
    
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


def identify_reproducible_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    validation_results: dict
) -> list:
    """
    Identify clusters with high reproducibility potential
    
    Criteria:
    - High data availability mentions
    - Computational/analytical methods
    - No clinical trial requirements
    - Existing datasets mentioned
    """
    
    reproducible_clusters = []
    
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        
        mask = labels == cluster_id
        cluster_df = df[mask]
        
        # Check for computational methods
        computational_score = 0
        data_availability_score = 0
        
        all_text = ' '.join(
            cluster_df['abstract_text'].fillna('').astype(str).str.lower()
        )
        
        # Computational indicators
        comp_keywords = ['dataset', 'database', 'computational', 'algorithm', 
                        'machine learning', 'statistical', 'retrospective']
        computational_score = sum(1 for kw in comp_keywords if kw in all_text) / len(comp_keywords)
        
        # Data availability indicators
        data_keywords = ['data available', 'publicly available', 'github', 
                        'repository', 'supplementary data', 'open data']
        data_availability_score = sum(1 for kw in data_keywords if kw in all_text) / len(cluster_df)
        
        # Clinical trial indicators (we want LOW scores here)
        trial_keywords = ['clinical trial', 'randomized', 'rct', 'phase i', 'phase ii']
        trial_score = sum(1 for kw in trial_keywords if kw in all_text)
        
        # Lab experiment indicators (we want LOW scores here)
        lab_keywords = ['in vitro', 'in vivo', 'cell culture', 'animal model', 'mice', 'rats']
        lab_score = sum(1 for kw in lab_keywords if kw in all_text)
        
        # Calculate reproducibility score
        reproducibility_score = (
            computational_score * 0.4 +
            data_availability_score * 0.3 +
            (1 - min(trial_score / 10, 1.0)) * 0.15 +
            (1 - min(lab_score / 10, 1.0)) * 0.15
        )
        
        if reproducibility_score >= 0.3:  # Threshold
            reproducible_clusters.append({
                'cluster_id': int(cluster_id),
                'size': int(mask.sum()),
                'reproducibility_score': float(reproducibility_score),
                'computational_score': float(computational_score),
                'data_availability_score': float(data_availability_score),
                'trial_mentions': int(trial_score),
                'lab_mentions': int(lab_score),
                'sample_titles': cluster_df['title'].head(3).tolist()
            })
    
    # Sort by reproducibility score
    reproducible_clusters.sort(key=lambda x: x['reproducibility_score'], reverse=True)
    
    return reproducible_clusters


def generate_data_driven_hypotheses(
    df: pd.DataFrame,
    labels: np.ndarray,
    reproducible_clusters: list
) -> list:
    """Generate hypotheses that can be tested with existing data"""
    
    hypotheses = []
    
    for cluster_info in reproducible_clusters[:5]:  # Top 5 clusters
        cluster_id = cluster_info['cluster_id']
        mask = labels == cluster_id
        cluster_df = df[mask]
        
        # Analyze cluster content
        all_text = ' '.join(cluster_df['abstract_text'].fillna('').astype(str))
        
        # Generate hypotheses based on cluster characteristics
        
        # Hypothesis 1: ML/AI application
        if 'machine learning' in all_text.lower() or 'deep learning' in all_text.lower():
            hypotheses.append({
                'cluster_id': cluster_id,
                'type': 'ml_application',
                'hypothesis': f"Machine learning models trained on existing datasets from cluster {cluster_id} "
                             f"can be improved by incorporating features from related clusters",
                'reproducibility': 'HIGH',
                'requirements': [
                    'Access to public datasets mentioned in papers',
                    'Standard ML frameworks (scikit-learn, TensorFlow, PyTorch)',
                    'Computational resources (GPU optional)'
                ],
                'verification_plan': [
                    '1. Download datasets from papers in cluster',
                    '2. Reproduce baseline models from papers',
                    '3. Test improved models with cross-validation',
                    '4. Compare performance metrics (AUC, accuracy, F1)',
                    '5. Statistical significance testing'
                ],
                'estimated_time': '2-4 weeks',
                'difficulty': 'Medium',
                'impact': 'Medium-High'
            })
        
        # Hypothesis 2: Meta-analysis
        if len(cluster_df) >= 10:
            hypotheses.append({
                'cluster_id': cluster_id,
                'type': 'meta_analysis',
                'hypothesis': f"Meta-analysis of {len(cluster_df)} studies in cluster {cluster_id} "
                             f"will reveal consistent effect sizes and identify moderating variables",
                'reproducibility': 'VERY HIGH',
                'requirements': [
                    'Published papers with reported effect sizes',
                    'Meta-analysis software (R metafor, Python meta)',
                    'Statistical knowledge'
                ],
                'verification_plan': [
                    '1. Extract effect sizes from all papers',
                    '2. Calculate pooled effect size with random effects model',
                    '3. Assess heterogeneity (I², Q statistic)',
                    '4. Perform subgroup analyses',
                    '5. Check for publication bias (funnel plot, Egger test)'
                ],
                'estimated_time': '1-2 weeks',
                'difficulty': 'Low-Medium',
                'impact': 'Medium'
            })
        
        # Hypothesis 3: Replication with public data
        if cluster_info['data_availability_score'] > 0.1:
            hypotheses.append({
                'cluster_id': cluster_id,
                'type': 'replication',
                'hypothesis': f"Key findings from cluster {cluster_id} can be replicated using "
                             f"publicly available datasets, validating original results",
                'reproducibility': 'VERY HIGH',
                'requirements': [
                    'Public datasets (identified in papers)',
                    'Statistical software (R, Python, SPSS)',
                    'Original analysis code if available'
                ],
                'verification_plan': [
                    '1. Identify papers with public data',
                    '2. Download datasets from repositories',
                    '3. Reproduce original analyses',
                    '4. Compare results with published findings',
                    '5. Document any discrepancies'
                ],
                'estimated_time': '1-3 weeks',
                'difficulty': 'Low',
                'impact': 'High (validates existing research)'
            })
        
        # Hypothesis 4: Cross-cluster pattern
        if len(reproducible_clusters) > 1:
            other_cluster = reproducible_clusters[1] if cluster_info == reproducible_clusters[0] else reproducible_clusters[0]
            hypotheses.append({
                'cluster_id': cluster_id,
                'type': 'cross_cluster',
                'hypothesis': f"Methods from cluster {cluster_id} can be applied to data from "
                             f"cluster {other_cluster['cluster_id']}, revealing new insights",
                'reproducibility': 'HIGH',
                'requirements': [
                    'Datasets from both clusters',
                    'Understanding of methods from both domains',
                    'Computational tools'
                ],
                'verification_plan': [
                    '1. Identify compatible datasets',
                    '2. Adapt methods from cluster A to cluster B data',
                    '3. Compare with existing approaches',
                    '4. Evaluate improvement in metrics',
                    '5. Validate on held-out test set'
                ],
                'estimated_time': '3-6 weeks',
                'difficulty': 'Medium-High',
                'impact': 'High (novel application)'
            })
    
    # Rank by reproducibility and impact
    for i, hyp in enumerate(hypotheses):
        score = 0
        if hyp['reproducibility'] == 'VERY HIGH':
            score += 3
        elif hyp['reproducibility'] == 'HIGH':
            score += 2
        
        if hyp['impact'] == 'High':
            score += 3
        elif hyp['impact'] == 'Medium-High':
            score += 2
        elif hyp['impact'] == 'Medium':
            score += 1
        
        if hyp['difficulty'] == 'Low':
            score += 2
        elif hyp['difficulty'] == 'Low-Medium':
            score += 1.5
        elif hyp['difficulty'] == 'Medium':
            score += 1
        
        hyp['priority_score'] = score
        hyp['rank'] = i + 1
    
    hypotheses.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Update ranks
    for i, hyp in enumerate(hypotheses):
        hyp['rank'] = i + 1
    
    return hypotheses


def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("REPRODUCIBLE HYPOTHESIS GENERATOR")
    print("="*70)
    print("\nFocus: Hypotheses testable with existing datasets")
    print("No clinical trials or lab experiments required\n")
    
    # Setup
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data/aiscientist/data/pubmed_data_2000.csv'
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} papers")
    
    # Filter computational papers
    print("\nFiltering computational/data-driven papers...")
    comp_df = filter_computational_papers(df)
    print(f"✅ Found {len(comp_df)} computational papers ({len(comp_df)/len(df)*100:.1f}%)")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embedder = PaperEmbedder()
    embeddings = embedder.embed_papers(comp_df, batch_size=16, show_progress=True)
    
    # Filter valid embeddings
    valid_mask = ~np.isnan(embeddings).any(axis=1)
    valid_embeddings = embeddings[valid_mask]
    valid_df = comp_df[valid_mask].reset_index(drop=True)
    print(f"✅ Valid embeddings: {len(valid_df)}")
    
    # Cluster
    print("\nClustering...")
    clusterer = SemanticClusterer(method='hdbscan')
    reduced = clusterer.reduce_dimensions(valid_embeddings, n_components=10)
    labels = clusterer.cluster_hdbscan(reduced, min_cluster_size=10)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✅ Found {n_clusters} clusters")
    
    # Identify reproducible clusters
    print("\nIdentifying reproducible clusters...")
    reproducible = identify_reproducible_clusters(valid_df, labels, {})
    print(f"✅ Found {len(reproducible)} reproducible clusters")
    
    print("\nTop Reproducible Clusters:")
    for cluster in reproducible[:5]:
        print(f"\n  Cluster {cluster['cluster_id']} ({cluster['size']} papers)")
        print(f"    Reproducibility Score: {cluster['reproducibility_score']:.2f}")
        print(f"    Computational: {cluster['computational_score']:.2f}")
        print(f"    Data Availability: {cluster['data_availability_score']:.2f}")
        print(f"    Sample: {cluster['sample_titles'][0][:60]}...")
    
    # Generate hypotheses
    print("\n" + "="*70)
    print("GENERATING REPRODUCIBLE HYPOTHESES")
    print("="*70)
    
    hypotheses = generate_data_driven_hypotheses(valid_df, labels, reproducible)
    
    print(f"\n✅ Generated {len(hypotheses)} reproducible hypotheses")
    
    # Display top hypotheses
    print("\n" + "="*70)
    print("TOP REPRODUCIBLE HYPOTHESES")
    print("="*70)
    
    for hyp in hypotheses[:5]:
        print(f"\n{'='*70}")
        print(f"HYPOTHESIS #{hyp['rank']} (Priority Score: {hyp['priority_score']:.1f})")
        print(f"{'='*70}")
        print(f"\nType: {hyp['type'].upper()}")
        print(f"Cluster: {hyp['cluster_id']}")
        print(f"Reproducibility: {hyp['reproducibility']}")
        print(f"Difficulty: {hyp['difficulty']}")
        print(f"Impact: {hyp['impact']}")
        print(f"Estimated Time: {hyp['estimated_time']}")
        
        print(f"\n📋 HYPOTHESIS:")
        print(f"   {hyp['hypothesis']}")
        
        print(f"\n📦 REQUIREMENTS:")
        for req in hyp['requirements']:
            print(f"   • {req}")
        
        print(f"\n✅ VERIFICATION PLAN:")
        for step in hyp['verification_plan']:
            print(f"   {step}")
    
    # Save results
    output_file = output_dir / 'reproducible_hypotheses.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_papers': len(df),
                'computational_papers': len(comp_df),
                'valid_papers': len(valid_df),
                'n_clusters': n_clusters,
                'reproducible_clusters': len(reproducible),
                'hypotheses_generated': len(hypotheses)
            },
            'reproducible_clusters': reproducible,
            'hypotheses': hypotheses
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Hypotheses: {len(hypotheses)}")
    print(f"Very High Reproducibility: {sum(1 for h in hypotheses if h['reproducibility'] == 'VERY HIGH')}")
    print(f"High Reproducibility: {sum(1 for h in hypotheses if h['reproducibility'] == 'HIGH')}")
    print(f"Low Difficulty: {sum(1 for h in hypotheses if 'Low' in h['difficulty'])}")
    print(f"High Impact: {sum(1 for h in hypotheses if 'High' in h['impact'])}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
