#!/usr/bin/env python3
"""
Test PubMed Enrichment with Problematic Cluster
Tests the PMIDs from the invalid meta-analysis cluster
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from external import PubMedClient


def main():
    print("\n" + "="*70)
    print("TESTING PUBMED ENRICHMENT")
    print("="*70)
    
    # PMIDs from the problematic cluster (mixed topics)
    pmids = [
        '36042322',  # Cortical atrophy CT
        '39792693',  # HER2 gastric cancer
        '33894656',  # smFRET
        '33343224',  # Genetic programming
        '35209064'   # Metalloproteins
    ]
    
    print(f"\nTesting {len(pmids)} PMIDs from cluster with mixed topics...")
    
    # Initialize client
    client = PubMedClient()
    
    # Enrich papers
    print("\n" + "-"*70)
    print("FETCHING METADATA (will use cache if available)")
    print("-"*70)
    
    papers = [{'pmid': pmid} for pmid in pmids]
    enriched = client.enrich_papers(papers, use_cache=True)
    
    # Display details for each paper
    print("\n" + "="*70)
    print("PAPER DETAILS")
    print("="*70)
    
    for i, paper in enumerate(enriched, 1):
        print(f"\n{'─'*70}")
        print(f"Paper {i}")
        print(f"{'─'*70}")
        
        print(f"PMID: {paper.get('pmid', 'N/A')}")
        print(f"Title: {paper.get('title', 'N/A')[:80]}...")
        print(f"Journal: {paper.get('journal', 'N/A')}")
        print(f"Year: {paper.get('year', 'N/A')}")
        
        pub_types = paper.get('publication_types', [])
        if pub_types:
            print(f"Publication Types: {', '.join(pub_types)}")
        
        mesh = paper.get('mesh_terms', [])
        if mesh:
            print(f"MeSH Terms ({len(mesh)} total):")
            for term in mesh[:5]:
                print(f"  • {term}")
            if len(mesh) > 5:
                print(f"  ... and {len(mesh) - 5} more")
        
        # Check if from cache
        if paper.get('_from_cache'):
            print("✅ Retrieved from cache")
        else:
            print("🌐 Fetched from PubMed API")
    
    # Validate for meta-analysis
    print("\n" + "="*70)
    print("META-ANALYSIS VALIDATION")
    print("="*70)
    
    validation = client.validate_for_meta_analysis(enriched)
    
    print(f"\nValid for meta-analysis: {validation['valid']}")
    
    if validation['valid']:
        print(f"✅ Papers are suitable for meta-analysis")
        print(f"   N papers: {validation['n_papers']}")
        print(f"   MeSH coverage: {validation['mesh_coverage']:.1%}")
        print(f"   {validation['recommendation']}")
    else:
        print(f"❌ Papers NOT suitable for meta-analysis")
        print(f"   Reason: {validation['reason']}")
        if 'recommendation' in validation:
            print(f"   {validation['recommendation']}")
        
        # Show statistics
        if 'n_papers' in validation:
            print(f"\nStatistics:")
            print(f"   Papers checked: {validation.get('n_papers', 0)}")
            print(f"   Reviews: {validation.get('review_count', 0)}")
            print(f"   Methods papers: {validation.get('method_count', 0)}")
            if 'mesh_coverage' in validation:
                print(f"   MeSH overlap: {validation['mesh_coverage']:.1%}")
    
    # Analyze MeSH heterogeneity
    print("\n" + "="*70)
    print("MESH TERM ANALYSIS")
    print("="*70)
    
    all_mesh = {}
    for paper in enriched:
        for term in paper.get('mesh_terms', []):
            all_mesh[term] = all_mesh.get(term, 0) + 1
    
    print(f"\nTotal unique MeSH terms: {len(all_mesh)}")
    
    # Find common terms (in >1 paper)
    common_mesh = {k: v for k, v in all_mesh.items() if v > 1}
    
    if common_mesh:
        print(f"Shared MeSH terms ({len(common_mesh)}):")
        sorted_mesh = sorted(common_mesh.items(), key=lambda x: x[1], reverse=True)
        for term, count in sorted_mesh[:10]:
            print(f"  • {term}: {count}/{len(enriched)} papers")
    else:
        print("⚠️  NO shared MeSH terms - papers are completely heterogeneous")
    
    # Check publication types
    print("\n" + "="*70)
    print("PUBLICATION TYPES ANALYSIS")
    print("="*70)
    
    all_types = {}
    for paper in enriched:
        for ptype in paper.get('publication_types', []):
            all_types[ptype] = all_types.get(ptype, 0) + 1
    
    if all_types:
        print("\nPublication types distribution:")
        sorted_types = sorted(all_types.items(), key=lambda x: x[1], reverse=True)
        for ptype, count in sorted_types:
            print(f"  • {ptype}: {count}/{len(enriched)} papers")
            
            # Flag problematic types
            if 'review' in ptype.lower() or 'meta-analysis' in ptype.lower():
                print(f"    ⚠️  Review/Meta-analysis detected")
            elif 'method' in ptype.lower() or 'protocol' in ptype.lower():
                print(f"    ⚠️  Methods paper detected")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if not validation['valid']:
        print("\n❌ This cluster should NOT generate a meta-analysis hypothesis")
        print(f"   Reason: {validation['reason']}")
        print("\n✅ Recommended hypothesis types instead:")
        print("   - Replication (if data available)")
        print("   - ML Application (if computational methods present)")
        print("   - Cross-Cluster Transfer (if compatible with other clusters)")
    else:
        print("\n✅ This cluster is suitable for meta-analysis")
        print("   Can proceed with meta-analysis hypothesis generation")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
