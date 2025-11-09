#!/usr/bin/env python3
"""
Embedding Evaluation Script
Compare different embedding models for scientific paper similarity
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from embeddings import PaperEmbedder


def evaluate_embedding_model(
    model_name: str,
    df: pd.DataFrame,
    sample_size: int = 100
) -> dict:
    """
    Evaluate an embedding model for scientific papers
    
    Args:
        model_name: HuggingFace model name
        df: DataFrame with papers
        sample_size: Number of papers to test
        
    Returns:
        Evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")
    
    df_sample = df.head(sample_size)
    
    # Time embedding generation
    start_time = time.time()
    
    try:
        embedder = PaperEmbedder(model_name=model_name)
        
        # Prepare texts
        texts = []
        for idx, row in df_sample.iterrows():
            parts = []
            if pd.notna(row.get('title')):
                parts.append(f"Title: {row['title']}")
            if pd.notna(row.get('abstract_text')):
                parts.append(f"Abstract: {row['abstract_text']}")
            texts.append(" ".join(parts))
        
        # Generate embeddings
        embeddings = embedder.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        embedding_time = time.time() - start_time
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return {
            'model': model_name,
            'status': 'failed',
            'error': str(e)
        }
    
    # Calculate metrics
    print(f"\n📊 Computing metrics...")
    
    # 1. Embedding dimension
    dimension = embeddings.shape[1]
    
    # 2. Similarity distribution
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, 0)  # Exclude self-similarity
    
    sim_mean = similarities.mean()
    sim_std = similarities.std()
    sim_min = similarities.min()
    sim_max = similarities.max()
    
    # 3. Discriminative power (can it distinguish papers?)
    # Good: Low mean similarity (papers are distinguishable)
    # Bad: High mean similarity (everything looks similar)
    discriminative_score = 1 - sim_mean
    
    # 4. Semantic coherence test
    # Find most similar pairs and check if they make sense
    top_pairs = []
    for i in range(min(5, len(similarities))):
        for j in range(i+1, len(similarities)):
            top_pairs.append((i, j, similarities[i, j]))
    
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    top_5_pairs = top_pairs[:5]
    
    # 5. Speed
    papers_per_second = sample_size / embedding_time
    
    results = {
        'model': model_name,
        'status': 'success',
        'dimension': dimension,
        'embedding_time': embedding_time,
        'papers_per_second': papers_per_second,
        'similarity_stats': {
            'mean': sim_mean,
            'std': sim_std,
            'min': sim_min,
            'max': sim_max
        },
        'discriminative_score': discriminative_score,
        'top_similar_pairs': [
            {
                'paper1_title': df_sample.iloc[i]['title'][:60] + '...',
                'paper2_title': df_sample.iloc[j]['title'][:60] + '...',
                'similarity': float(sim)
            }
            for i, j, sim in top_5_pairs
        ]
    }
    
    # Print results
    print(f"\n✅ RESULTS:")
    print(f"   Dimension: {dimension}")
    print(f"   Time: {embedding_time:.2f}s ({papers_per_second:.1f} papers/sec)")
    print(f"   Similarity - Mean: {sim_mean:.3f}, Std: {sim_std:.3f}")
    print(f"   Discriminative Score: {discriminative_score:.3f} (higher=better)")
    
    print(f"\n🔍 Top 3 Most Similar Pairs:")
    for i, pair in enumerate(top_5_pairs[:3], 1):
        idx1, idx2, sim = pair
        print(f"\n   {i}. Similarity: {sim:.3f}")
        print(f"      Paper 1: {df_sample.iloc[idx1]['title'][:70]}...")
        print(f"      Paper 2: {df_sample.iloc[idx2]['title'][:70]}...")
    
    return results


def compare_models(df: pd.DataFrame, sample_size: int = 100):
    """
    Compare multiple embedding models
    """
    print("\n" + "="*70)
    print("EMBEDDING MODEL COMPARISON FOR SCIENTIFIC PAPERS")
    print("="*70)
    
    # Models to compare
    models = {
        'all-MiniLM-L6-v2': {
            'description': 'General purpose, fast (current)',
            'dimension': 384,
            'speed': 'very fast'
        },
        'allenai/specter': {
            'description': 'Scientific papers specialist',
            'dimension': 768,
            'speed': 'slower'
        },
        'sentence-transformers/all-mpnet-base-v2': {
            'description': 'General purpose, high quality',
            'dimension': 768,
            'speed': 'moderate'
        }
    }
    
    print(f"\n📋 Models to evaluate:")
    for model, info in models.items():
        print(f"   • {model}")
        print(f"     {info['description']}")
        print(f"     Dimension: {info['dimension']}, Speed: {info['speed']}")
    
    print(f"\n🔬 Testing with {sample_size} papers...")
    
    results = {}
    
    for model_name in models.keys():
        try:
            result = evaluate_embedding_model(model_name, df, sample_size)
            results[model_name] = result
        except Exception as e:
            print(f"\n❌ Failed to evaluate {model_name}: {e}")
            results[model_name] = {'status': 'failed', 'error': str(e)}
    
    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<40} {'Dim':<8} {'Speed':<12} {'Discrim':<10}")
    print("-" * 70)
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            print(f"{model_name:<40} "
                  f"{result['dimension']:<8} "
                  f"{result['papers_per_second']:<12.1f} "
                  f"{result['discriminative_score']:<10.3f}")
        else:
            print(f"{model_name:<40} FAILED")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("\n🎯 For Scientific Papers:")
    print("   1. allenai/specter - BEST for scientific similarity")
    print("      • Trained specifically on scientific papers")
    print("      • Better at capturing domain-specific semantics")
    print("      • Slower but more accurate")
    
    print("\n⚡ For Speed:")
    print("   2. all-MiniLM-L6-v2 - FASTEST (current)")
    print("      • Good general-purpose embeddings")
    print("      • 3-5x faster than SPECTER")
    print("      • Acceptable for initial exploration")
    
    print("\n⚖️  For Balance:")
    print("   3. all-mpnet-base-v2 - BALANCED")
    print("      • High quality general embeddings")
    print("      • Moderate speed")
    print("      • Good compromise")
    
    print("\n💡 Recommendation for Your Use Case:")
    print("   Use allenai/specter for:")
    print("   • Final production pipeline")
    print("   • When accuracy is critical")
    print("   • Scientific domain-specific clustering")
    
    print("\n   Use all-MiniLM-L6-v2 for:")
    print("   • Quick prototyping and testing")
    print("   • When speed matters")
    print("   • Initial exploration")
    
    return results


def main():
    """Run embedding evaluation"""
    
    # Load data
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data/aiscientist/data/pubmed_data_2000.csv'
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} papers")
    
    # Run comparison
    results = compare_models(df, sample_size=100)
    
    # Save results
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)
    
    import json
    with open(output_dir / 'embedding_evaluation.json', 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\n✅ Results saved to output/embedding_evaluation.json")


if __name__ == "__main__":
    main()
