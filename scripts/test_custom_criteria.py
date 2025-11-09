#!/usr/bin/env python3
"""
Test Custom Criteria
Demonstrates how to use and extend the custom criteria system
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from embeddings import PaperEmbedder
from clustering import SemanticClusterer
from extraction.custom_criteria import (
    CustomCriteriaValidator,
    ClinicalTrialSponsorCriterion,
    DataAvailabilityCriterion,
    ReplicationStatusCriterion,
    ValidationCriterion
)


# Example: Create a custom criterion for your specific needs
class GeographicDiversityCriterion(ValidationCriterion):
    """
    Example custom criterion: Check geographic diversity in research
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__("geographic_diversity", weight)
        
        self.regions = {
            'north_america': ['usa', 'canada', 'united states', 'american'],
            'europe': ['uk', 'germany', 'france', 'italy', 'spain', 'european'],
            'asia': ['china', 'japan', 'india', 'korea', 'asian'],
            'other': ['australia', 'brazil', 'africa']
        }
    
    def evaluate(self, cluster_df: pd.DataFrame, text_column: str = 'abstract') -> dict:
        """Evaluate geographic diversity"""
        
        all_text = ' '.join(
            cluster_df[text_column].dropna().astype(str).str.lower()
        )
        
        region_counts = {}
        for region, keywords in self.regions.items():
            count = sum(1 for kw in keywords if kw in all_text)
            region_counts[region] = count
        
        regions_mentioned = sum(1 for count in region_counts.values() if count > 0)
        
        # Score based on diversity
        if regions_mentioned >= 3:
            score = 1.0
            interpretation = "High geographic diversity"
        elif regions_mentioned == 2:
            score = 0.8
            interpretation = "Moderate geographic diversity"
        elif regions_mentioned == 1:
            score = 0.6
            interpretation = "Limited to one region"
        else:
            score = 0.5
            interpretation = "No clear geographic information"
        
        return {
            'score': score,
            'details': {
                'regions_mentioned': regions_mentioned,
                'region_counts': region_counts
            },
            'interpretation': interpretation
        }


def main():
    """Demonstrate custom criteria usage"""
    
    print("\n" + "="*70)
    print("CUSTOM CRITERIA DEMONSTRATION")
    print("="*70)
    
    # Load data
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data/aiscientist/data/pubmed_data_2000.csv'
    
    if not data_path.exists():
        print(f"❌ Data file not found")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} papers")
    
    # Use small sample for demo
    df_sample = df.head(50)
    
    # Generate embeddings and cluster
    print("\n📊 Generating embeddings and clustering...")
    embedder = PaperEmbedder()
    
    texts = []
    for idx, row in df_sample.iterrows():
        parts = []
        if pd.notna(row.get('title')):
            parts.append(f"Title: {row['title']}")
        if pd.notna(row.get('abstract_text')):
            parts.append(f"Abstract: {row['abstract_text']}")
        texts.append(" ".join(parts))
    
    embeddings = embedder.model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    clusterer = SemanticClusterer(method='hdbscan')
    reduced_embeddings = clusterer.reduce_dimensions(embeddings, n_components=5)
    labels = clusterer.cluster_hdbscan(reduced_embeddings, min_cluster_size=3, min_samples=2)
    
    print(f"✅ Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    
    # Prepare dataframe
    df_test = df_sample.copy()
    if 'abstract_text' in df_test.columns and 'abstract' not in df_test.columns:
        df_test['abstract'] = df_test['abstract_text']
    
    # Create custom criteria validator
    print("\n" + "="*70)
    print("SETTING UP CUSTOM CRITERIA")
    print("="*70)
    
    validator = CustomCriteriaValidator()
    
    # Add built-in criteria
    print("\n📋 Adding built-in criteria:")
    validator.add_criterion(ClinicalTrialSponsorCriterion(weight=0.2))
    validator.add_criterion(DataAvailabilityCriterion(weight=0.15))
    validator.add_criterion(ReplicationStatusCriterion(weight=0.15))
    
    # Add custom criterion
    print("\n📋 Adding custom criterion:")
    validator.add_criterion(GeographicDiversityCriterion(weight=0.1))
    
    # List all criteria
    validator.list_criteria()
    
    # Evaluate clusters
    results = validator.evaluate_all_clusters(df_test, labels, text_column='abstract')
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for cluster_name, cluster_results in results['custom_criteria_results'].items():
        print(f"\n{cluster_name.upper()} ({cluster_results['size']} papers)")
        print(f"Overall Custom Score: {cluster_results['overall_custom_score']:.2f}")
        
        print("\nCriteria Breakdown:")
        for criterion_name, criterion_result in cluster_results['criteria_results'].items():
            print(f"  • {criterion_name}: {criterion_result['score']:.2f}")
            print(f"    {criterion_result['interpretation']}")
    
    # Save results
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)
    
    import json
    
    def convert_to_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json(item) for item in obj]
        return obj
    
    with open(output_dir / 'custom_criteria_results.json', 'w') as f:
        json.dump(convert_to_json(results), f, indent=2)
    
    print(f"\n✅ Results saved to output/custom_criteria_results.json")
    
    # How to create your own criterion
    print("\n" + "="*70)
    print("HOW TO CREATE YOUR OWN CRITERION")
    print("="*70)
    
    print("""
1. Inherit from ValidationCriterion:

    class MyCriterion(ValidationCriterion):
        def __init__(self, weight=0.1):
            super().__init__("my_criterion", weight)
        
        def evaluate(self, cluster_df, text_column='abstract'):
            # Your logic here
            score = 0.8  # Calculate your score (0-1)
            
            return {
                'score': score,
                'details': {'your': 'data'},
                'interpretation': 'Your explanation'
            }

2. Add to validator:

    validator.add_criterion(MyCriterion(weight=0.15))

3. Run evaluation:

    results = validator.evaluate_all_clusters(df, labels)

That's it! The system is fully modular and extensible.
    """)


if __name__ == "__main__":
    main()
