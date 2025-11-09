#!/usr/bin/env python3
"""
Full Pipeline: Semantic Classification & Hypothesis Generation

This script runs the complete pipeline:
1. Load PubMed dataset
2. Generate embeddings
3. Cluster papers semantically
4. Analyze gaps
5. Generate hypotheses
"""

import sys
import os
from pathlib import Path
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from embeddings import PaperEmbedder
from clustering import SemanticClusterer
from extraction import GapAnalyzer, HypothesisGenerator


def main(args):
    """Run full pipeline"""
    
    print("\n" + "="*70)
    print("RESEARCH SEMANTIC POC - FULL PIPELINE")
    print("="*70 + "\n")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data_file
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("STEP 1: Loading dataset...")
    print("-" * 70)
    
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("\nPlease ensure you have cloned the aiscientist repo:")
        print("  cd data/")
        print("  git clone https://github.com/sergeicu/aiscientist")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} papers from {data_path.name}")
    print(f"   Columns: {list(df.columns)}")
    
    # =========================================================================
    # STEP 2: Generate Embeddings
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Generating semantic embeddings...")
    print("-" * 70)
    
    embeddings_file = output_dir / 'embeddings.pkl'
    
    if embeddings_file.exists() and not args.force:
        print(f"Loading existing embeddings from {embeddings_file}")
        embedder = PaperEmbedder()
        embedding_data = embedder.load_embeddings(embeddings_file)
        embeddings = embedding_data['embeddings']
    else:
        embedder = PaperEmbedder(model_name=args.model)
        embeddings = embedder.embed_papers(df, batch_size=args.batch_size)
        embedder.save_embeddings(
            embeddings, 
            str(embeddings_file),
            metadata={
                'n_papers': len(df),
                'data_file': str(data_path)
            }
        )
    
    print(f"✅ Embeddings shape: {embeddings.shape}")
    
    # =========================================================================
    # STEP 3: Cluster Papers
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: Clustering papers semantically...")
    print("-" * 70)
    
    clusterer = SemanticClusterer(method=args.cluster_method)
    
    # Filter out NaN embeddings
    import numpy as np
    valid_mask = ~np.isnan(embeddings).any(axis=1)
    valid_embeddings = embeddings[valid_mask]
    valid_df = df[valid_mask].reset_index(drop=True)
    
    print(f"Valid embeddings: {valid_embeddings.shape[0]} / {embeddings.shape[0]}")
    
    # Reduce dimensions first
    reduced_embeddings = clusterer.reduce_dimensions(valid_embeddings, n_components=10)
    
    # Then cluster
    if args.cluster_method == 'hdbscan':
        labels = clusterer.cluster_hdbscan(
            reduced_embeddings,
            min_cluster_size=args.min_cluster_size
        )
    else:
        labels = clusterer.cluster(reduced_embeddings)
    
    # Use valid_df instead of df for rest of pipeline
    df = valid_df
    
    clusters_file = output_dir / 'clusters.json'
    clusterer.save_clusters(str(clusters_file), df, labels)
    
    print(f"✅ Clustering complete")
    
    # =========================================================================
    # STEP 4: Gap Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Analyzing research gaps...")
    print("-" * 70)
    
    gap_analyzer = GapAnalyzer()
    gap_report = gap_analyzer.generate_gap_report(df, labels, embeddings)
    
    gap_file = output_dir / 'gap_analysis.json'
    gap_analyzer.save_report(gap_report, str(gap_file))
    
    # =========================================================================
    # STEP 5: Generate Hypotheses
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Generating research hypotheses...")
    print("-" * 70)
    
    hyp_generator = HypothesisGenerator()
    
    # Get cluster info
    cluster_info = clusterer.get_cluster_stats(df, labels)
    
    hypotheses = hyp_generator.generate_all_hypotheses(
        gap_report,
        df,
        labels,
        embeddings,
        cluster_info
    )
    
    hyp_file = output_dir / 'hypotheses.json'
    hyp_generator.save_hypotheses(hypotheses, str(hyp_file))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print(f"  • Papers analyzed: {len(df)}")
    print(f"  • Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"  • Gaps identified: {len(gap_report['temporal_gaps'])} temporal, "
          f"{len(gap_report['methodological_gaps'])} methodological")
    print(f"  • Contradictions found: {len(gap_report['contradictions'])}")
    print(f"  • Opportunities: {len(gap_report['cross_cluster_opportunities'])}")
    print(f"  • Hypotheses generated: {hypotheses['metadata']['total_hypotheses']}")
    
    print("\n📁 Output files:")
    print(f"  • {embeddings_file}")
    print(f"  • {clusters_file}")
    print(f"  • {gap_file}")
    print(f"  • {hyp_file}")
    
    print("\n🎯 Next steps:")
    print("  1. Review hypotheses in output/hypotheses.json")
    print("  2. Examine gap analysis in output/gap_analysis.json")
    print("  3. Explore clusters in output/clusters.json")
    print("  4. Use notebooks/ for interactive analysis")
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run full semantic classification and hypothesis generation pipeline'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='data/aiscientist/pubmed_data_2000.csv',
        help='Path to input CSV file (default: pubmed_data_2000.csv for testing)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Embedding model name (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--cluster-method',
        type=str,
        default='hdbscan',
        choices=['hdbscan', 'dbscan', 'hierarchical'],
        help='Clustering method (default: hdbscan)'
    )
    
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=10,
        help='Minimum cluster size (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration of embeddings even if they exist'
    )
    
    args = parser.parse_args()
    
    main(args)
