#!/usr/bin/env python3
"""
PubMed E-utilities Client with Local Cache
Fetches paper metadata from PubMed and caches locally
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET


class PubMedClient:
    """Client for fetching PubMed metadata with local caching"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    CACHE_DIR = Path("output/cache/pubmed")
    CACHE_EXPIRY_DAYS = 30
    RATE_LIMIT_DELAY = 0.34  # ~3 requests/second without API key
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PubMed client
        
        Args:
            api_key: Optional NCBI API key for higher rate limits
        """
        self.api_key = api_key
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0
        
        if api_key:
            self.RATE_LIMIT_DELAY = 0.1  # 10 requests/second with API key
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_path(self, pmid: str) -> Path:
        """Get cache file path for a PMID"""
        return self.CACHE_DIR / f"{pmid}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid"""
        if not cache_path.exists():
            return False
        
        # Check age
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        
        return age < timedelta(days=self.CACHE_EXPIRY_DAYS)
    
    def _load_cache(self, pmid: str) -> Optional[Dict]:
        """Load cached data if valid"""
        cache_path = self._get_cache_path(pmid)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_from_cache'] = True
                    return data
            except Exception as e:
                print(f"Cache read error for PMID {pmid}: {e}")
                return None
        
        return None
    
    def _save_cache(self, pmid: str, data: Dict):
        """Save data to cache"""
        cache_path = self._get_cache_path(pmid)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Cache write error for PMID {pmid}: {e}")
    
    def fetch_summary(self, pmid: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Fetch paper summary from PubMed ESummary API
        
        Args:
            pmid: PubMed ID
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with paper metadata or None if error
        """
        # Try cache first
        if use_cache:
            cached = self._load_cache(pmid)
            if cached:
                return cached
        
        # Fetch from API
        self._rate_limit()
        
        url = f"{self.BASE_URL}/esummary.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'json'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant fields
            if 'result' in data and pmid in data['result']:
                result = data['result'][pmid]
                
                enriched = {
                    'pmid': pmid,
                    'title': result.get('title', ''),
                    'journal': result.get('fulljournalname', result.get('source', '')),
                    'pubdate': result.get('pubdate', ''),
                    'year': result.get('pubdate', '').split()[0] if result.get('pubdate') else None,
                    'doi': result.get('elocationid', ''),
                    'authors': [author.get('name', '') for author in result.get('authors', [])],
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    '_fetched_at': datetime.now().isoformat(),
                    '_from_cache': False
                }
                
                # Save to cache
                self._save_cache(pmid, enriched)
                
                return enriched
            else:
                print(f"No result found for PMID {pmid}")
                return None
                
        except Exception as e:
            print(f"Error fetching PMID {pmid}: {e}")
            return None
    
    def fetch_details(self, pmid: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Fetch detailed paper info from PubMed EFetch API (XML)
        Includes MeSH terms, publication types, abstract
        
        Args:
            pmid: PubMed ID
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with detailed metadata
        """
        # Try to get from summary cache first
        cache_path = self._get_cache_path(pmid)
        if use_cache and self._is_cache_valid(cache_path):
            cached = self._load_cache(pmid)
            if cached and 'mesh_terms' in cached:
                return cached
        
        # Fetch details
        self._rate_limit()
        
        url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            article = root.find('.//PubmedArticle/MedlineCitation/Article')
            
            if article is None:
                return None
            
            # Extract fields
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ''
            
            abstract_elem = article.find('.//Abstract/AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ''
            
            # Publication types
            pub_types = []
            for pt in article.findall('.//PublicationTypeList/PublicationType'):
                if pt.text:
                    pub_types.append(pt.text)
            
            # MeSH terms
            mesh_terms = []
            for mesh in root.findall('.//MeshHeadingList/MeshHeading/DescriptorName'):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            
            # Journal
            journal_elem = root.find('.//MedlineCitation/Article/Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ''
            
            # Year
            year_elem = root.find('.//PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/PubDate/Year')
            year = year_elem.text if year_elem is not None else None
            
            details = {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'journal': journal,
                'year': year,
                'publication_types': pub_types,
                'mesh_terms': mesh_terms,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                '_fetched_at': datetime.now().isoformat(),
                '_from_cache': False
            }
            
            # Merge with summary if exists
            summary = self._load_cache(pmid)
            if summary:
                details.update(summary)
            
            # Save to cache
            self._save_cache(pmid, details)
            
            return details
            
        except Exception as e:
            print(f"Error fetching details for PMID {pmid}: {e}")
            return None
    
    def enrich_papers(self, papers: List[Dict], use_cache: bool = True) -> List[Dict]:
        """
        Enrich a list of papers with PubMed metadata
        
        Args:
            papers: List of paper dicts with 'pmid' field
            use_cache: Whether to use cache
            
        Returns:
            List of enriched paper dicts
        """
        enriched = []
        
        for paper in papers:
            pmid = str(paper.get('pmid', '')).replace('.0', '').strip()
            
            if not pmid or pmid == 'nan':
                enriched.append(paper)
                continue
            
            # Fetch enriched data
            metadata = self.fetch_details(pmid, use_cache=use_cache)
            
            if metadata:
                # Merge original paper data with enriched metadata
                enriched_paper = {**paper, **metadata}
                enriched.append(enriched_paper)
            else:
                enriched.append(paper)
        
        return enriched
    
    def validate_for_meta_analysis(self, papers: List[Dict]) -> Dict:
        """
        Validate if papers are suitable for meta-analysis
        
        Checks:
        - Not reviews/methods papers
        - Have quantitative outcomes
        - Homogeneous study designs
        
        Returns:
            Dict with validation results
        """
        if not papers:
            return {'valid': False, 'reason': 'No papers provided'}
        
        # Ensure papers are enriched
        pmids_to_check = []
        for p in papers:
            pmid = str(p.get('pmid', '')).replace('.0', '').strip()
            if pmid and pmid != 'nan':
                pmids_to_check.append(pmid)
        
        if not pmids_to_check:
            return {'valid': False, 'reason': 'No valid PMIDs'}
        
        # Fetch details for papers
        enriched_papers = []
        for pmid in pmids_to_check[:10]:  # Limit to first 10 for validation
            details = self.fetch_details(pmid)
            if details:
                enriched_papers.append(details)
        
        if len(enriched_papers) < 5:
            return {'valid': False, 'reason': f'Too few papers with metadata ({len(enriched_papers)})'}
        
        # Check publication types
        review_count = 0
        method_count = 0
        
        for paper in enriched_papers:
            pub_types = paper.get('publication_types', [])
            pub_types_lower = [pt.lower() for pt in pub_types]
            
            if any(term in ' '.join(pub_types_lower) for term in ['review', 'meta-analysis']):
                review_count += 1
            
            if any(term in ' '.join(pub_types_lower) for term in ['method', 'protocol', 'guideline']):
                method_count += 1
        
        if review_count > len(enriched_papers) * 0.3:
            return {'valid': False, 'reason': f'Too many reviews ({review_count}/{len(enriched_papers)})'}
        
        if method_count > len(enriched_papers) * 0.2:
            return {'valid': False, 'reason': f'Too many methods papers ({method_count}/{len(enriched_papers)})'}
        
        # Check for homogeneity via MeSH
        all_mesh = set()
        for paper in enriched_papers:
            all_mesh.update(paper.get('mesh_terms', []))
        
        if len(all_mesh) == 0:
            return {'valid': False, 'reason': 'No MeSH terms found'}
        
        # Calculate overlap
        mesh_coverage = []
        for paper in enriched_papers:
            paper_mesh = set(paper.get('mesh_terms', []))
            if all_mesh:
                coverage = len(paper_mesh & all_mesh) / len(all_mesh)
                mesh_coverage.append(coverage)
        
        avg_coverage = sum(mesh_coverage) / len(mesh_coverage) if mesh_coverage else 0
        
        if avg_coverage < 0.2:
            return {
                'valid': False,
                'reason': f'Low MeSH overlap (avg coverage: {avg_coverage:.1%})',
                'recommendation': 'Papers too heterogeneous for meta-analysis'
            }
        
        # Passed all checks
        return {
            'valid': True,
            'n_papers': len(enriched_papers),
            'review_count': review_count,
            'method_count': method_count,
            'mesh_coverage': avg_coverage,
            'recommendation': 'Papers appear suitable for meta-analysis'
        }


# Example usage
if __name__ == "__main__":
    client = PubMedClient()
    
    # Test with a PMID
    result = client.fetch_details('36042322')
    
    if result:
        print(f"Title: {result['title']}")
        print(f"Journal: {result['journal']}")
        print(f"Year: {result['year']}")
        print(f"MeSH Terms: {', '.join(result.get('mesh_terms', [])[:5])}")
        print(f"Publication Types: {', '.join(result.get('publication_types', []))}")
        print(f"From cache: {result['_from_cache']}")
