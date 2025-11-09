#!/usr/bin/env python3
"""
Automated Testing Script (Non-interactive)
Test each component without user prompts
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from embeddings import PaperEmbedder
from clustering import SemanticClusterer
from extraction import ClassificationValidator


def main():
    """Run all tests automatically"""
    
    print("\n" + "="*70)
    print("AUTOMATED COMPONENT TESTING")
    print("="*70)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data/aiscientist/data/pubmed_data_2000.csv'
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Track test results
    results = {}
    
    # TEST 1: Data Loading
    print("\n" + "="*70)
    print("TEST 1: DATA LOADING")
    print("="*70)
    
    try:
        if not data_path.exists():
            print(f"❌ FAILED: Data file not found at {data_path}")
            return
        
        df = pd.read_csv(data_path)
        print(f"✅ PASSED: Loaded {len(df)} papers")
        print(f"   Columns: {list(df.columns)[:10]}...")
        
        # Sample data
        print(f"\n📄 Sample paper:")
        sample = df.iloc[0]
        print(f"   Title: {str(sample.get('title', 'N/A'))[:80]}...")
        print(f"   Abstract length: {len(str(sample.get('abstract_text', '')))}")
        
        results['data_loading'] = True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results['data_loading'] = False
        return
    
    # TEST 2: Embeddings (small sample)
    print("\n" + "="*70)
    print("TEST 2: EMBEDDING GENERATION")
    print("="*70)
    
    try:
        sample_size = 50  # Small sample for quick test
        df_sample = df.head(sample_size)
        
        print(f"Testing with {sample_size} papers...")
        
        embedder = PaperEmbedder()
        
        # Prepare text manually since column names might differ
        texts = []
        for idx, row in df_sample.iterrows():
            parts = []
            if pd.notna(row.get('title')):
                parts.append(f"Title: {row['title']}")
            if pd.notna(row.get('abstract_text')):
                parts.append(f"Abstract: {row['abstract_text']}")
            texts.append(" ".join(parts))
        
        # Generate embeddings directly
        embeddings = embedder.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ PASSED: Generated embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dimension: {embeddings.shape[1]}")
        print(f"   NaN values: {np.isnan(embeddings).sum()}")
        
        # Check embedding quality
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings[:10])
        
        print(f"\n📊 Embedding Quality Check:")
        print(f"   Avg similarity (top 10 papers): {similarities.mean():.3f}")
        print(f"   Min similarity: {similarities[similarities > 0].min():.3f}")
        print(f"   Max similarity: {similarities.max():.3f}")
        
        results['embeddings'] = True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['embeddings'] = False
        return
    
    # TEST 3: Clustering
    print("\n" + "="*70)
    print("TEST 3: SEMANTIC CLUSTERING")
    print("="*70)
    
    try:
        clusterer = SemanticClusterer(method='hdbscan')
        
        # First reduce dimensions
        reduced_embeddings = clusterer.reduce_dimensions(
            embeddings,
            n_components=5
        )
        
        # Then cluster with HDBSCAN parameters
        labels = clusterer.cluster_hdbscan(
            reduced_embeddings,
            min_cluster_size=3,  # Very low for small test
            min_samples=2
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
        
        results['clustering'] = True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['clustering'] = False
        return
    
    # TEST 4: Validation
    print("\n" + "="*70)
    print("TEST 4: CLASSIFICATION VALIDATION")
    print("="*70)
    
    try:
        # Prepare dataframe with proper column name
        df_test = df_sample.copy()
        if 'abstract_text' in df_test.columns and 'abstract' not in df_test.columns:
            df_test['abstract'] = df_test['abstract_text']
        
        validator = ClassificationValidator()
        
        # Validate all clusters
        validation_summary = validator.validate_all_clusters(
            df_test, 
            labels,
            text_column='abstract'
        )
        
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
            print(f"   Note: Small sample size may affect validation scores")
        
        results['validation'] = True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['validation'] = False
    
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
        print("\nNext steps:")
        print("1. Run with more papers: python scripts/run_full_pipeline.py")
        print("2. Review validation report: cat output/test_validation.json")
        print("3. Use Claude agents for interactive analysis")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Review errors above before proceeding.")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
