"""
Hypothesis Generator Module
Generates novel, testable hypotheses based on gap analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from datetime import datetime


class HypothesisGenerator:
    """
    Generate research hypotheses that go beyond state-of-the-art
    """
    
    def __init__(self):
        self.hypotheses = []
        
    def generate_from_temporal_gaps(
        self,
        temporal_gaps: Dict,
        cluster_info: Dict
    ) -> List[Dict]:
        """
        Generate hypotheses for outdated research areas
        
        Args:
            temporal_gaps: Temporal gap analysis
            cluster_info: Cluster characteristics
            
        Returns:
            List of hypotheses
        """
        hypotheses = []
        
        for cluster_name, gap_data in temporal_gaps.items():
            if gap_data.get('is_outdated', False):
                hypothesis = {
                    'type': 'temporal_update',
                    'cluster': cluster_name,
                    'hypothesis': (
                        f"Re-examining findings from {cluster_name} with modern "
                        f"methods and recent data will reveal new insights, as "
                        f"the most recent work is {gap_data['years_since_peak']} "
                        f"years old."
                    ),
                    'rationale': (
                        f"Research in this area peaked in {gap_data['most_recent']}. "
                        f"New technologies, methods, and data sources are now available."
                    ),
                    'verification_approach': [
                        "Apply modern ML/computational methods to historical questions",
                        "Use updated datasets not available at time of original studies",
                        "Replicate key findings with larger sample sizes"
                    ],
                    'feasibility': 'high',
                    'novelty': 'medium',
                    'impact_potential': 'high'
                }
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_from_methodological_gaps(
        self,
        method_gaps: Dict,
        cluster_info: Dict
    ) -> List[Dict]:
        """
        Generate hypotheses about applying underused methodologies
        
        Args:
            method_gaps: Methodological gap analysis
            cluster_info: Cluster characteristics
            
        Returns:
            List of hypotheses
        """
        hypotheses = []
        
        for cluster_name, gap_data in method_gaps.items():
            underused = gap_data.get('underused_methods', [])
            
            for method in underused:
                if method in ['machine_learning', 'computational']:
                    hypothesis = {
                        'type': 'methodological_innovation',
                        'cluster': cluster_name,
                        'method': method,
                        'hypothesis': (
                            f"Applying {method.replace('_', ' ')} techniques to "
                            f"problems in {cluster_name} will uncover patterns "
                            f"not detectable by traditional methods."
                        ),
                        'rationale': (
                            f"Only {gap_data['method_usage'].get(method, 0):.1f}% "
                            f"of papers in this cluster use {method}, despite its "
                            f"proven effectiveness in related domains."
                        ),
                        'verification_approach': [
                            f"Apply {method} to existing datasets from this cluster",
                            "Compare performance against traditional methods",
                            "Identify novel patterns or predictions"
                        ],
                        'data_requirements': "Existing datasets from cluster papers",
                        'feasibility': 'high',
                        'novelty': 'high',
                        'impact_potential': 'high'
                    }
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_from_contradictions(
        self,
        contradictions: List[Dict],
        df: pd.DataFrame
    ) -> List[Dict]:
        """
        Generate hypotheses to resolve contradictions
        
        Args:
            contradictions: List of detected contradictions
            df: DataFrame with paper data
            
        Returns:
            List of hypotheses
        """
        hypotheses = []
        
        for i, contradiction in enumerate(contradictions[:5]):  # Top 5
            paper1 = df.iloc[contradiction['paper_1_idx']]
            paper2 = df.iloc[contradiction['paper_2_idx']]
            
            hypothesis = {
                'type': 'contradiction_resolution',
                'contradiction_id': i,
                'hypothesis': (
                    f"The apparent contradiction between findings on similar topics "
                    f"can be explained by differences in methodology, population, "
                    f"or unmeasured confounders."
                ),
                'papers_involved': [
                    contradiction['paper_1_title'],
                    contradiction['paper_2_title']
                ],
                'rationale': (
                    "High semantic similarity suggests these papers address the "
                    "same topic but potentially reach different conclusions."
                ),
                'verification_approach': [
                    "Systematic comparison of methodologies",
                    "Meta-analysis incorporating both findings",
                    "New study designed to test boundary conditions",
                    "Identify moderating variables"
                ],
                'feasibility': 'medium',
                'novelty': 'high',
                'impact_potential': 'very_high',
                'note': 'Resolving contradictions advances scientific consensus'
            }
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_from_cross_cluster_opportunities(
        self,
        opportunities: List[Dict],
        cluster_info: Dict
    ) -> List[Dict]:
        """
        Generate hypotheses from cross-cluster connections
        
        Args:
            opportunities: Cross-cluster opportunities
            cluster_info: Cluster characteristics
            
        Returns:
            List of hypotheses
        """
        hypotheses = []
        
        for opp in opportunities[:5]:  # Top 5
            hypothesis = {
                'type': 'cross_domain_synthesis',
                'clusters_involved': [opp['cluster_1'], opp['cluster_2']],
                'hypothesis': (
                    f"Integrating methods from cluster {opp['cluster_1']} with "
                    f"problems from cluster {opp['cluster_2']} will yield novel "
                    f"insights not achievable within either domain alone."
                ),
                'rationale': (
                    f"Found {opp['n_connections']} connections between clusters "
                    f"suggesting potential for methodological transfer."
                ),
                'verification_approach': [
                    "Identify successful methods from cluster 1",
                    "Adapt methods to address questions in cluster 2",
                    "Benchmark against within-cluster approaches",
                    "Demonstrate superior performance or novel insights"
                ],
                'example': f"{opp['sample_papers']}",
                'feasibility': 'medium',
                'novelty': 'very_high',
                'impact_potential': 'very_high',
                'note': 'Cross-domain innovations often lead to breakthroughs'
            }
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_data_driven_hypotheses(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        embeddings: np.ndarray,
        cluster_info: Dict
    ) -> List[Dict]:
        """
        Generate hypotheses by analyzing data patterns
        
        Args:
            df: DataFrame with paper data
            labels: Cluster labels
            embeddings: Paper embeddings
            cluster_info: Cluster characteristics
            
        Returns:
            List of hypotheses
        """
        hypotheses = []
        
        # Analyze citation patterns
        if 'citations' in df.columns:
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                
                mask = labels == cluster_id
                cluster_df = df[mask]
                
                citations = pd.to_numeric(
                    cluster_df['citations'], 
                    errors='coerce'
                ).dropna()
                
                if len(citations) > 10:
                    high_impact = citations > citations.quantile(0.75)
                    
                    if high_impact.sum() > 5:
                        hypothesis = {
                            'type': 'impact_analysis',
                            'cluster': f'cluster_{cluster_id}',
                            'hypothesis': (
                                f"Papers with certain methodological characteristics "
                                f"in cluster {cluster_id} achieve significantly "
                                f"higher impact. Identifying and applying these "
                                f"characteristics will increase research impact."
                            ),
                            'rationale': (
                                f"Top quartile papers have {citations.quantile(0.75):.0f}+ "
                                f"citations vs median of {citations.median():.0f}."
                            ),
                            'verification_approach': [
                                "Identify common features of high-impact papers",
                                "Apply features to new research design",
                                "Track citation accumulation prospectively"
                            ],
                            'feasibility': 'high',
                            'novelty': 'medium',
                            'impact_potential': 'high'
                        }
                        hypotheses.append(hypothesis)
        
        return hypotheses
    
    def rank_hypotheses(
        self,
        hypotheses: List[Dict]
    ) -> List[Dict]:
        """
        Rank hypotheses by novelty, feasibility, and impact
        
        Args:
            hypotheses: List of hypotheses
            
        Returns:
            Ranked list of hypotheses
        """
        score_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'very_high': 4
        }
        
        for h in hypotheses:
            novelty_score = score_map.get(h.get('novelty', 'medium'), 2)
            feasibility_score = score_map.get(h.get('feasibility', 'medium'), 2)
            impact_score = score_map.get(h.get('impact_potential', 'medium'), 2)
            
            # Weighted score: impact > novelty > feasibility
            h['score'] = (impact_score * 2 + novelty_score * 1.5 + feasibility_score) / 4.5
        
        return sorted(hypotheses, key=lambda x: x['score'], reverse=True)
    
    def generate_all_hypotheses(
        self,
        gap_report: Dict,
        df: pd.DataFrame,
        labels: np.ndarray,
        embeddings: np.ndarray,
        cluster_info: Dict
    ) -> Dict:
        """
        Generate all types of hypotheses
        
        Args:
            gap_report: Complete gap analysis report
            df: DataFrame with paper data
            labels: Cluster labels
            embeddings: Paper embeddings
            cluster_info: Cluster characteristics
            
        Returns:
            Dictionary with all hypotheses
        """
        print("\n" + "="*60)
        print("GENERATING RESEARCH HYPOTHESES")
        print("="*60 + "\n")
        
        all_hypotheses = []
        
        # From temporal gaps
        print("Generating hypotheses from temporal gaps...")
        temporal_hyp = self.generate_from_temporal_gaps(
            gap_report['temporal_gaps'],
            cluster_info
        )
        all_hypotheses.extend(temporal_hyp)
        print(f"  Generated {len(temporal_hyp)} hypotheses")
        
        # From methodological gaps
        print("Generating hypotheses from methodological gaps...")
        method_hyp = self.generate_from_methodological_gaps(
            gap_report['methodological_gaps'],
            cluster_info
        )
        all_hypotheses.extend(method_hyp)
        print(f"  Generated {len(method_hyp)} hypotheses")
        
        # From contradictions
        print("Generating hypotheses from contradictions...")
        contradiction_hyp = self.generate_from_contradictions(
            gap_report['contradictions'],
            df
        )
        all_hypotheses.extend(contradiction_hyp)
        print(f"  Generated {len(contradiction_hyp)} hypotheses")
        
        # From cross-cluster opportunities
        print("Generating hypotheses from cross-cluster opportunities...")
        cross_hyp = self.generate_from_cross_cluster_opportunities(
            gap_report['cross_cluster_opportunities'],
            cluster_info
        )
        all_hypotheses.extend(cross_hyp)
        print(f"  Generated {len(cross_hyp)} hypotheses")
        
        # Data-driven hypotheses
        print("Generating data-driven hypotheses...")
        data_hyp = self.generate_data_driven_hypotheses(
            df, labels, embeddings, cluster_info
        )
        all_hypotheses.extend(data_hyp)
        print(f"  Generated {len(data_hyp)} hypotheses")
        
        # Rank all hypotheses
        print("\nRanking hypotheses by score...")
        ranked_hypotheses = self.rank_hypotheses(all_hypotheses)
        
        result = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_hypotheses': len(ranked_hypotheses),
                'data_source': 'Boston Children\'s Hospital PubMed dataset'
            },
            'summary': {
                'by_type': self._count_by_type(ranked_hypotheses),
                'top_5': ranked_hypotheses[:5]
            },
            'all_hypotheses': ranked_hypotheses
        }
        
        print(f"\n✅ Generated {len(ranked_hypotheses)} total hypotheses")
        
        return result
    
    def _count_by_type(self, hypotheses: List[Dict]) -> Dict:
        """Count hypotheses by type"""
        from collections import Counter
        types = [h['type'] for h in hypotheses]
        return dict(Counter(types))
    
    def save_hypotheses(self, hypotheses: Dict, filepath: str):
        """Save hypotheses to JSON"""
        with open(filepath, 'w') as f:
            json.dump(hypotheses, f, indent=2)
        
        print(f"\n✅ Hypotheses saved to {filepath}")
        print("\nTop 3 Hypotheses:")
        for i, h in enumerate(hypotheses['summary']['top_5'][:3], 1):
            print(f"\n{i}. [{h['type'].upper()}] Score: {h['score']:.2f}")
            print(f"   {h['hypothesis']}")


if __name__ == "__main__":
    print("Hypothesis Generator Module")
    print("=" * 50)
    print("\nGenerates testable hypotheses from:")
    print("  - Temporal gaps")
    print("  - Methodological gaps")
    print("  - Contradictions")
    print("  - Cross-cluster opportunities")
    print("  - Data patterns")
