"""
arXiv Paper Classification: Data Loader & Preprocessing
Fetches papers from arXiv API and prepares dataset for classification
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict
import re

class ArxivDataLoader:
    """
    Fetch papers from arXiv API
    Rate limit: 3 requests/second
    """
    
    BASE_URL = "http://export.arxiv.org/api/query?"
    
    # Map arXiv categories to our 5 classes
    CATEGORY_MAPPING = {
        'cs.AI': 'AI',
        'cs.LG': 'ML',
        'cs.CL': 'NLP',
        'cs.CV': 'Computer Vision',
        'cs.RO': 'Robotics'
    }
    
    def __init__(self):
        self.papers = []
        self.failed_queries = []
    
    def fetch_papers_by_category(self, 
                                 category: str,
                                 max_results: int = 20000,
                                 start_date: str = '202301010000'):
        """
        Fetch papers from specific arXiv category
        
        Args:
            category: arXiv category code (e.g., 'cs.AI')
            max_results: Max papers per category
            start_date: YYYYMMDDHHMM format (default: Jan 1, 2023)
        
        Returns:
            List of papers with metadata
        """
        
        papers = []
        batch_size = 100  # arXiv API max per query
        
        for start in range(0, max_results, batch_size):
            # Construct query
            search_query = f'cat:{category} AND submittedDate:[{start_date}010000 TO 202412312359]'
            
            params = {
                'search_query': search_query,
                'start': start,
                'max_results': batch_size,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=100)
                response.raise_for_status()
                
                # Parse XML response
                entries = self._parse_arxiv_response(response.text, category)
                papers.extend(entries)
                
                print(f"✓ {category}: fetched {len(papers)} papers (batch {start//batch_size + 1})")
                
                # Rate limiting
                time.sleep(3)
                
                if len(papers) >= max_results:
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"✗ Error fetching {category} batch {start//batch_size}: {e}")
                self.failed_queries.append((category, start))
                time.sleep(1)
                continue
        
        self.papers.extend(papers[:max_results])
        return papers[:max_results]
    
    def _parse_arxiv_response(self, xml_text: str, category: str) -> List[Dict]:
        """Parse arXiv API XML response"""
        
        import xml.etree.ElementTree as ET
        
        papers = []
        root = ET.fromstring(xml_text)
        
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', namespace):
            try:
                paper = {
                    'arxiv_id': entry.find('atom:id', namespace).text.split('/abs/')[-1],
                    'title': entry.find('atom:title', namespace).text.strip(),
                    'abstract': entry.find('atom:summary', namespace).text.strip(),
                    'authors': [author.find('atom:name', namespace).text 
                               for author in entry.findall('atom:author', namespace)],
                    'submitted_date': entry.find('atom:published', namespace).text,
                    'primary_category': category,
                    'label': self.CATEGORY_MAPPING.get(category, 'Other'),
                }
                papers.append(paper)
            except AttributeError:
                continue
        
        return papers
    
    def fetch_all_categories(self, papers_per_category: int = 15000):
        """Fetch balanced dataset across all 5 categories"""
        
        print("Starting data collection from arXiv...")
        
        for category in self.CATEGORY_MAPPING.keys():
            self.fetch_papers_by_category(category, max_results=papers_per_category)
        
        print(f"\nTotal papers collected: {len(self.papers)}")
        
        return self.to_dataframe()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert papers to pandas DataFrame"""
        df = pd.DataFrame(self.papers)
        df['submitted_date'] = pd.to_datetime(df['submitted_date'])
        return df
    
    def save_to_csv(self, filepath: str):
        """Save dataset to CSV"""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {len(df)} papers to {filepath}")
        return df


class TextPreprocessor:
    """
    Preprocess abstract text for BERT-like models
    """
    
    def __init__(self):
        self.patterns = [
            (r'http\S+', ''),  # Remove URLs
            (r'arxiv:\s*[\d.]+', ''),  # Remove arXiv references
            (r'\s+', ' '),  # Collapse whitespace
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean abstract text"""
        
        # Remove newlines
        text = text.replace('\n', ' ')
        
        # Apply regex patterns
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def prepare_dataset(self, df: pd.DataFrame, output_path: str):
        """
        Prepare dataset for training
        - Clean text
        - Remove duplicates
        - Create train/val/test splits
        """
        
        print("Preprocessing dataset...")
        
        # Clean abstracts
        df['abstract_clean'] = df['abstract'].apply(self.clean_text)
        
        # Remove duplicates (by title)
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Remove very short abstracts
        df['abstract_length'] = df['abstract_clean'].str.split().str.len()
        df = df[df['abstract_length'] >= 20]
        
        print(f"After cleaning: {len(df)} papers")
        
        # Class distribution
        print("\nClass distribution:")
        print(df['label'].value_counts())
        
        # Save cleaned dataset
        df.to_csv(output_path, index=False)
        print(f"\n✓ Cleaned dataset saved to {output_path}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Option 1: Fetch from arXiv API (takes ~1 hour for 75K papers)
    # loader = ArxivDataLoader()
    # df = loader.fetch_all_categories(papers_per_category=15000)
    # loader.save_to_csv('data/raw/arxiv_papers.csv')
    
    # Option 2: Use this sample for testing
    print("Data loader ready. Use ArxivDataLoader().fetch_all_categories()")
