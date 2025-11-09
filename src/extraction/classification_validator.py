"""
Classification Validator Module
Validates semantic classifications using scientific criteria
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import json
from collections import Counter
import re


class ClassificationValidator:
    """
    Validate cluster classifications using scientific domain criteria
    """
    
    def __init__(self):
        self.validation_results = {}
        
        # Scientific research methodologies
        self.methodologies = {
            'clinical_trial': [
                'randomized', 'controlled trial', 'phase i', 'phase ii', 
                'phase iii', 'rct', 'double-blind', 'placebo'
            ],
            'cohort_study': [
                'cohort', 'prospective', 'retrospective', 'follow-up',
                'longitudinal', 'observational'
            ],
            'case_control': [
                'case-control', 'matched controls', 'odds ratio', 'case control'
            ],
            'cross_sectional': [
                'cross-sectional', 'survey', 'prevalence', 'questionnaire'
            ],
            'systematic_review': [
                'systematic review', 'meta-analysis', 'prisma', 'cochrane'
            ],
            'laboratory': [
                'in vitro', 'in vivo', 'cell culture', 'animal model',
                'mouse model', 'biochemical assay'
            ],
            'computational': [
                'bioinformatics', 'machine learning', 'deep learning',
                'computational', 'algorithm', 'modeling', 'simulation'
            ],
            'genomic': [
                'genome-wide', 'gwas', 'sequencing', 'rna-seq',
                'genomic', 'transcriptomic', 'whole exome'
            ]
        }
        
        # Conceptual frameworks
        self.frameworks = {
            'mechanistic': ['mechanism', 'pathway', 'molecular', 'signaling'],
            'etiological': ['etiology', 'risk factor', 'cause', 'pathogenesis'],
            'descriptive': ['prevalence', 'incidence', 'epidemiology', 'distribution'],
            'predictive': ['prognosis', 'prediction', 'risk score', 'prognostic'],
            'interventional': ['treatment', 'therapy', 'intervention', 'efficacy']
        }
        
    def validate_cluster(
        self,
        cluster_df: pd.DataFrame,
        cluster_id: int,
        text_column: str = 'abstract'
    ) -> Dict:
        """
        Validate a single cluster using scientific criteria
        
        Args:
            cluster_df: DataFrame with papers in this cluster
            cluster_id: Cluster identifier
            text_column: Column containing text for analysis
            
        Returns:
            Validation report with scores and justifications
        """
        print(f"\n{'='*70}")
        print(f"VALIDATING CLUSTER {cluster_id} ({len(cluster_df)} papers)")
        print(f"{'='*70}")
        
        # Combine all text for analysis
        all_text = ' '.join(
            cluster_df[text_column].dropna().astype(str).str.lower()
        )
        
        # 1. Methodological coherence
        method_score, method_profile = self._assess_methodology(all_text, len(cluster_df))
        
        # 2. Conceptual framework
        framework_score, framework_profile = self._assess_framework(all_text)
        
        # 3. Temporal coherence (if year available)
        temporal_score, temporal_profile = self._assess_temporal(cluster_df)
        
        # 4. Internal consistency (title/abstract similarity)
        consistency_score = self._assess_consistency(cluster_df, text_column)
        
        # 5. Scientific validity (MeSH terms if available)
        mesh_score = self._assess_mesh_coherence(cluster_df)
        
        # Overall validation score
        overall_score = np.mean([
            method_score * 0.35,      # Most important
            framework_score * 0.25,
            temporal_score * 0.15,
            consistency_score * 0.15,
            mesh_score * 0.10
        ])
        
        # Generate justification
        justification = self._generate_justification(
            cluster_id,
            method_profile,
            framework_profile,
            temporal_profile,
            len(cluster_df)
        )
        
        # Quality flags
        quality_flags = self._generate_quality_flags(
            overall_score,
            method_score,
            framework_score,
            temporal_score
        )
        
        report = {
            'cluster_id': cluster_id,
            'size': len(cluster_df),
            'overall_score': float(overall_score),
            'validation_passed': overall_score >= 0.6,
            'scores': {
                'methodological_coherence': float(method_score),
                'framework_coherence': float(framework_score),
                'temporal_coherence': float(temporal_score),
                'internal_consistency': float(consistency_score),
                'mesh_coherence': float(mesh_score)
            },
            'profiles': {
                'methodology': method_profile,
                'framework': framework_profile,
                'temporal': temporal_profile
            },
            'justification': justification,
            'quality_flags': quality_flags,
            'recommended_action': self._recommend_action(overall_score, quality_flags)
        }
        
        self._print_validation_summary(report)
        
        return report
    
    def _assess_methodology(
        self, 
        text: str, 
        n_papers: int
    ) -> Tuple[float, Dict]:
        """Assess methodological coherence"""
        
        method_counts = {}
        for method, keywords in self.methodologies.items():
            count = sum(1 for keyword in keywords if keyword in text)
            method_counts[method] = count
        
        # Calculate dominance (one method should be clearly dominant)
        total_mentions = sum(method_counts.values())
        if total_mentions == 0:
            return 0.0, {'dominant': 'unknown', 'distribution': method_counts}
        
        # Find dominant methodology
        dominant_method = max(method_counts.items(), key=lambda x: x[1])
        dominant_percentage = dominant_method[1] / total_mentions
        
        # Score based on dominance (higher = more coherent)
        # 70%+ of one methodology = excellent (0.9-1.0)
        # 50-70% = good (0.7-0.9)
        # 30-50% = moderate (0.5-0.7)
        # <30% = poor (<0.5)
        
        if dominant_percentage >= 0.7:
            score = 0.9 + (dominant_percentage - 0.7) * 0.33
        elif dominant_percentage >= 0.5:
            score = 0.7 + (dominant_percentage - 0.5) * 1.0
        elif dominant_percentage >= 0.3:
            score = 0.5 + (dominant_percentage - 0.3) * 1.0
        else:
            score = dominant_percentage / 0.3 * 0.5
        
        profile = {
            'dominant_method': dominant_method[0],
            'dominance_percentage': float(dominant_percentage),
            'mentions': method_counts,
            'interpretation': self._interpret_methodology(dominant_method[0], dominant_percentage)
        }
        
        return min(score, 1.0), profile
    
    def _assess_framework(self, text: str) -> Tuple[float, Dict]:
        """Assess conceptual framework coherence"""
        
        framework_counts = {}
        for framework, keywords in self.frameworks.items():
            count = sum(1 for keyword in keywords if keyword in text)
            framework_counts[framework] = count
        
        total = sum(framework_counts.values())
        if total == 0:
            return 0.5, {'dominant': 'unclear', 'distribution': framework_counts}
        
        dominant = max(framework_counts.items(), key=lambda x: x[1])
        dominance = dominant[1] / total
        
        # Framework can be mixed (e.g., mechanistic + interventional)
        # So we're more lenient: 50%+ is good
        score = min(dominance * 1.5, 1.0)
        
        profile = {
            'dominant_framework': dominant[0],
            'dominance_percentage': float(dominance),
            'distribution': framework_counts
        }
        
        return score, profile
    
    def _assess_temporal(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """Assess temporal coherence"""
        
        if 'year' not in df.columns:
            return 0.7, {'status': 'year_not_available'}
        
        years = pd.to_numeric(df['year'], errors='coerce').dropna()
        
        if len(years) == 0:
            return 0.5, {'status': 'no_valid_years'}
        
        year_range = years.max() - years.min()
        median_year = years.median()
        
        # Temporal coherence is good if:
        # - Papers are within 10 years of each other (score: 0.8-1.0)
        # - Papers are within 15 years (score: 0.6-0.8)
        # - Papers >15 years apart (score: <0.6)
        
        if year_range <= 10:
            score = 0.8 + (10 - year_range) / 50  # 0.8 to 1.0
        elif year_range <= 15:
            score = 0.6 + (15 - year_range) / 25  # 0.6 to 0.8
        else:
            score = max(0.3, 0.6 - (year_range - 15) / 50)
        
        profile = {
            'year_range': int(year_range),
            'min_year': int(years.min()),
            'max_year': int(years.max()),
            'median_year': int(median_year),
            'is_coherent': year_range <= 10
        }
        
        return score, profile
    
    def _assess_consistency(
        self, 
        df: pd.DataFrame, 
        text_column: str
    ) -> float:
        """
        Assess internal consistency using keyword overlap
        Simplified version - full version would use embeddings
        """
        
        if len(df) < 2:
            return 1.0
        
        # Extract common words (simple approach)
        texts = df[text_column].dropna().astype(str).str.lower()
        
        # Count word frequencies across papers
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-z]{4,}\b', text)  # Words 4+ chars
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        
        # Papers are consistent if they share common vocabulary
        # Calculate what % of papers mention top terms
        
        if len(word_counts) == 0:
            return 0.5
        
        top_words = [word for word, count in word_counts.most_common(20)]
        
        # For each paper, count how many top words it contains
        overlaps = []
        for text in texts:
            text_lower = text.lower()
            overlap = sum(1 for word in top_words if word in text_lower)
            overlaps.append(overlap / len(top_words))
        
        # Average overlap score
        consistency_score = np.mean(overlaps)
        
        return consistency_score
    
    def _assess_mesh_coherence(self, df: pd.DataFrame) -> float:
        """Assess MeSH term coherence if available"""
        
        if 'mesh_terms' not in df.columns:
            return 0.7  # Neutral score if not available
        
        mesh_data = df['mesh_terms'].dropna()
        
        if len(mesh_data) == 0:
            return 0.7
        
        # Extract all MeSH terms
        all_mesh = []
        for terms in mesh_data:
            if isinstance(terms, str):
                all_mesh.extend(terms.split(';'))
        
        if len(all_mesh) == 0:
            return 0.7
        
        mesh_counts = Counter(all_mesh)
        
        # Good coherence if top MeSH terms appear in many papers
        total_papers = len(df)
        top_mesh_coverage = [
            count / total_papers 
            for _, count in mesh_counts.most_common(5)
        ]
        
        # Average coverage of top 5 MeSH terms
        if top_mesh_coverage:
            return min(np.mean(top_mesh_coverage) * 1.5, 1.0)
        
        return 0.7
    
    def _interpret_methodology(
        self, 
        method: str, 
        dominance: float
    ) -> str:
        """Generate interpretation of methodology"""
        
        method_descriptions = {
            'clinical_trial': 'Clinical interventional studies (RCTs, trials)',
            'cohort_study': 'Observational cohort studies with follow-up',
            'case_control': 'Case-control studies examining risk factors',
            'cross_sectional': 'Cross-sectional surveys or prevalence studies',
            'systematic_review': 'Systematic reviews and meta-analyses',
            'laboratory': 'Laboratory-based experimental research',
            'computational': 'Computational/bioinformatics analysis',
            'genomic': 'Genomic or transcriptomic studies'
        }
        
        desc = method_descriptions.get(method, 'Unknown methodology')
        
        if dominance >= 0.7:
            strength = "strongly characterized by"
        elif dominance >= 0.5:
            strength = "primarily characterized by"
        else:
            strength = "partially characterized by"
        
        return f"Cluster {strength} {desc}"
    
    def _generate_justification(
        self,
        cluster_id: int,
        method_profile: Dict,
        framework_profile: Dict,
        temporal_profile: Dict,
        size: int
    ) -> Dict:
        """Generate scientific justification for cluster"""
        
        justification = {
            'cluster_identity': f"Cluster {cluster_id} ({size} papers)",
            'methodological_basis': method_profile.get('interpretation', ''),
            'dominant_method': method_profile.get('dominant_method', 'unknown'),
            'conceptual_framework': framework_profile.get('dominant_framework', 'unknown'),
            'temporal_context': '',
            'scientific_validity': ''
        }
        
        # Temporal context
        if 'year_range' in temporal_profile:
            if temporal_profile['year_range'] <= 5:
                justification['temporal_context'] = (
                    f"Temporally coherent: papers from {temporal_profile['min_year']}-"
                    f"{temporal_profile['max_year']} ({temporal_profile['year_range']} year span)"
                )
            else:
                justification['temporal_context'] = (
                    f"Temporally dispersed: {temporal_profile['year_range']} year span "
                    f"({temporal_profile['min_year']}-{temporal_profile['max_year']})"
                )
        
        # Scientific validity
        method_dominance = method_profile.get('dominance_percentage', 0)
        if method_dominance >= 0.7:
            justification['scientific_validity'] = "Strong methodological coherence"
        elif method_dominance >= 0.5:
            justification['scientific_validity'] = "Moderate methodological coherence"
        else:
            justification['scientific_validity'] = "Weak methodological coherence - review recommended"
        
        return justification
    
    def _generate_quality_flags(
        self,
        overall: float,
        method: float,
        framework: float,
        temporal: float
    ) -> List[str]:
        """Generate quality flags for cluster"""
        
        flags = []
        
        if overall >= 0.8:
            flags.append("✅ EXCELLENT: High-quality, scientifically coherent cluster")
        elif overall >= 0.6:
            flags.append("✅ GOOD: Acceptable scientific coherence")
        else:
            flags.append("⚠️  REVIEW: Low coherence - consider reclassification")
        
        if method < 0.5:
            flags.append("🚩 METHODOLOGICAL: Mixed methodologies - may need splitting")
        
        if framework < 0.4:
            flags.append("🚩 CONCEPTUAL: Unclear conceptual framework")
        
        if temporal < 0.5:
            flags.append("⚠️  TEMPORAL: Wide time span - methods may have evolved")
        
        return flags
    
    def _recommend_action(
        self, 
        overall_score: float, 
        quality_flags: List[str]
    ) -> str:
        """Recommend action based on validation"""
        
        if overall_score >= 0.8:
            return "ACCEPT: Cluster is scientifically valid"
        elif overall_score >= 0.6:
            return "ACCEPT WITH REVIEW: Validate key papers manually"
        else:
            if any('METHODOLOGICAL' in flag for flag in quality_flags):
                return "SPLIT: Consider splitting by methodology"
            elif any('TEMPORAL' in flag for flag in quality_flags):
                return "SPLIT: Consider splitting by time period"
            else:
                return "RECLASSIFY: Low coherence across multiple dimensions"
    
    def _print_validation_summary(self, report: Dict):
        """Print validation summary"""
        
        print(f"\n🎯 Overall Score: {report['overall_score']:.2f}")
        print(f"📊 Status: {'PASSED ✅' if report['validation_passed'] else 'NEEDS REVIEW ⚠️'}")
        
        print(f"\n📈 Dimension Scores:")
        for dim, score in report['scores'].items():
            print(f"  • {dim}: {score:.2f}")
        
        print(f"\n🔬 Scientific Profile:")
        print(f"  • Methodology: {report['profiles']['methodology']['dominant_method']}")
        print(f"  • Framework: {report['profiles']['framework']['dominant_framework']}")
        
        print(f"\n💡 Quality Flags:")
        for flag in report['quality_flags']:
            print(f"  {flag}")
        
        print(f"\n🎬 Recommended Action: {report['recommended_action']}")
        print(f"{'='*70}\n")
    
    def validate_all_clusters(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        text_column: str = 'abstract'
    ) -> Dict:
        """Validate all clusters"""
        
        print("\n" + "="*70)
        print("COMPREHENSIVE CLUSTER VALIDATION")
        print("="*70)
        
        validation_reports = {}
        
        unique_clusters = [c for c in set(labels) if c != -1]
        
        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            cluster_df = df[mask]
            
            report = self.validate_cluster(cluster_df, cluster_id, text_column)
            validation_reports[f"cluster_{cluster_id}"] = report
        
        # Summary
        n_passed = sum(1 for r in validation_reports.values() if r['validation_passed'])
        n_total = len(validation_reports)
        
        summary = {
            'total_clusters': n_total,
            'passed_validation': n_passed,
            'failed_validation': n_total - n_passed,
            'pass_rate': n_passed / n_total if n_total > 0 else 0,
            'cluster_reports': validation_reports
        }
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Total Clusters: {n_total}")
        print(f"Passed: {n_passed} ({summary['pass_rate']*100:.1f}%)")
        print(f"Need Review: {n_total - n_passed}")
        print("="*70 + "\n")
        
        return summary
    
    def save_validation_report(self, validation_summary: Dict, filepath: str):
        """Save validation report to JSON"""
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_summary = convert_to_json_serializable(validation_summary)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        print(f"✅ Validation report saved to {filepath}")


if __name__ == "__main__":
    print("Classification Validator Module")
    print("=" * 50)
    print("\nValidates clusters using scientific criteria:")
    print("  - Methodological coherence")
    print("  - Conceptual framework")
    print("  - Temporal coherence")
    print("  - Internal consistency")
    print("  - MeSH term overlap")
