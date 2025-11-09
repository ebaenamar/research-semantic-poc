#!/usr/bin/env python3
"""
Step-by-Step Testing Script
Test each component individually before running full pipeline
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
from extraction import ClassificationValidator


def test_data_loading(data_path: Path):
    """TEST 1: Data loading"""
    print("\n" + "="*70)
    print("TEST 1: DATA LOADING")
    print("="*70)
    
    if not data_path.exists():
        print(f"❌ FAILED: Data file not found at {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ PASSED: Loaded {len(df)} papers")
    print(f"   Columns: {list(df.columns)[:10]}...")
    
    # Check required columns
    required = ['title', 'abstract']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"⚠️  WARNING: Missing columns: {missing}")
    else:
        print(f"✅ PASSED: All required columns present")
    
    # Sample data
    print(f"\n📄 Sample paper:")
    sample = df.iloc[0]
    print(f"   Title: {sample.get('title', 'N/A')[:80]}...")
    print(f"   Abstract length: {len(str(sample.get('abstract', '')))}")
    
    return df


def test_embeddings(df: pd.DataFrame, output_dir: Path):
    """TEST 2: Embedding generation"""
    print("\n" + "="*70)
    print("TEST 2: EMBEDDING GENERATION")
    print("="*70)
    
    # Test with small sample first
    sample_size = min(100, len(df))
    df_sample = df.head(sample_size)
    
    print(f"Testing with {sample_size} papers...")
    
    try:
        embedder = PaperEmbedder()
        embeddings = embedder.embed_papers(df_sample, batch_size=16, show_progress=True)
        
        print(f"✅ PASSED: Generated embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dimension: {embeddings.shape[1]}")
        print(f"   NaN values: {np.isnan(embeddings).sum()}")
        
        # Check embedding quality
        non_nan_embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]
        if len(non_nan_embeddings) > 1:
            # Compute pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(non_nan_embeddings[:10])
            
            print(f"\n📊 Embedding Quality Check:")
            print(f"   Avg similarity (top 10 papers): {similarities.mean():.3f}")
            print(f"   Min similarity: {similarities[similarities > 0].min():.3f}")
            print(f"   Max similarity: {similarities.max():.3f}")
        
        # Save test embeddings
        test_emb_path = output_dir / 'test_embeddings.pkl'
        embedder.save_embeddings(
            embeddings,
            str(test_emb_path),
            metadata={'test': True, 'sample_size': sample_size}
        )
        print(f"✅ PASSED: Saved test embeddings to {test_emb_path}")
        
        return embeddings, df_sample
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_clustering(embeddings: np.ndarray, df: pd.DataFrame):
    """TEST 3: Clustering"""
    print("\n" + "="*70)
    print("TEST 3: SEMANTIC CLUSTERING")
    print("="*70)
    
    try:
        clusterer = SemanticClusterer(method='hdbscan')
        labels = clusterer.cluster(
            embeddings,
            reduce_dims=True,
            n_components=5,
            min_cluster_size=5  # Lower for small test
        )
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        print(f"✅ PASSED: Clustering complete")
        print(f"   Clusters found: {n_clusters}")
        print(f"   Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        
        # Print cluster sizes
        print(f"\n📊 Cluster Sizes:")
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            size = (labels == cluster_id).sum()
            print(f"   Cluster {cluster_id}: {size} papers")
        
        return labels
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_validation(df: pd.DataFrame, labels: np.ndarray, output_dir: Path):
    """TEST 4: Classification validation"""
    print("\n" + "="*70)
    print("TEST 4: CLASSIFICATION VALIDATION")
    print("="*70)
    
    try:
        validator = ClassificationValidator()
        
        # Validate all clusters
        validation_summary = validator.validate_all_clusters(df, labels)
        
        # Save validation report
        validation_path = output_dir / 'test_validation.json'
        validator.save_validation_report(validation_summary, str(validation_path))
        
        print(f"✅ PASSED: Validation complete")
        print(f"   Report saved to {validation_path}")
        
        # Check if any clusters passed
        pass_rate = validation_summary['pass_rate']
        if pass_rate >= 0.7:
            print(f"✅ EXCELLENT: {pass_rate*100:.1f}% clusters passed validation")
        elif pass_rate >= 0.5:
            print(f"⚠️  ACCEPTABLE: {pass_rate*100:.1f}% clusters passed validation")
        else:
            print(f"🚩 CONCERNING: Only {pass_rate*100:.1f}% clusters passed validation")
            print(f"   Consider adjusting clustering parameters")
        
        return validation_summary
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(args):
    """Run all tests"""
    
    print("\n" + "="*70)
    print("STEP-BY-STEP COMPONENT TESTING")
    print("="*70)
    print("\nThis will test each component independently before running full pipeline.")
    print("Tests will run on a small sample (100 papers) for speed.\n")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data_file
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Track test results
    results = {
        'data_loading': False,
        'embeddings': False,
        'clustering': False,
        'validation': False
    }
    
    # TEST 1: Data Loading
    df = test_data_loading(data_path)
    if df is not None:
        results['data_loading'] = True
    else:
        print("\n❌ Cannot proceed without data. Exiting.")
        return
    
    input("\nPress Enter to continue to Test 2 (Embeddings)...")
    
    # TEST 2: Embeddings
    import numpy as np  # Import here after confirming data
    embeddings, df_sample = test_embeddings(df, output_dir)
    if embeddings is not None:
        results['embeddings'] = True
    else:
        print("\n❌ Cannot proceed without embeddings. Exiting.")
        return
    
    input("\nPress Enter to continue to Test 3 (Clustering)...")
    
    # TEST 3: Clustering
    labels = test_clustering(embeddings, df_sample)
    if labels is not None:
        results['clustering'] = True
    else:
        print("\n❌ Cannot proceed without clustering. Exiting.")
        return
    
    input("\nPress Enter to continue to Test 4 (Validation)...")
    
    # TEST 4: Validation
    validation_summary = test_validation(df_sample, labels, output_dir)
    if validation_summary is not None:
        results['validation'] = True
    
    # FINAL SUMMARY
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou can now:")
        print("1. Run full pipeline: python scripts/run_full_pipeline.py")
        print("2. Adjust parameters in config/pipeline_config.yaml")
        print("3. Use Claude agents for analysis")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Review errors above before proceeding to full pipeline.")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test components step by step')
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='data/aiscientist/pubmed_data_2000.csv',
        help='Path to input CSV file'
    )
    
    args = parser.parse_args()
    
    main(args)
