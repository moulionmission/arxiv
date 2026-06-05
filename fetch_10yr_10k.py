import requests
import time
import pandas as pd
import xml.etree.ElementTree as ET
import os
from datetime import datetime

CATEGORIES = {
    'cs.AI': 'AI',
    'cs.LG': 'ML',
    'cs.CL': 'NLP',
    'cs.CV': 'Computer Vision',
    'cs.RO': 'Robotics'
}

YEARS = list(range(2016, 2026))
PAPERS_PER_YEAR = 1000
BATCH_SIZE = 100
OUTPUT_FILE = 'data/raw/arxiv_10yr_10k_unbiased.csv'

def parse_xml_response(xml_text):
    """Parse arXiv API XML response and extract true primary categories"""
    root = ET.fromstring(xml_text)
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    papers = []
    
    for entry in root.findall('atom:entry', ns):
        try:
            # Extract true primary category term
            p_cat_elem = entry.find('arxiv:primary_category', ns)
            if p_cat_elem is None:
                continue
            primary_cat = p_cat_elem.attrib.get('term', '')
            
            # Map only if primary category is in our target categories
            if primary_cat not in CATEGORIES:
                continue
                
            paper = {
                'arxiv_id': entry.find('atom:id', ns).text.split('/abs/')[-1],
                'title': entry.find('atom:title', ns).text.strip(),
                'abstract': entry.find('atom:summary', ns).text.strip(),
                'submitted_date': entry.find('atom:published', ns).text,
                'label': CATEGORIES[primary_cat]
            }
            papers.append(paper)
        except Exception:
            continue
            
    return papers

def load_existing_progress():
    """Load existing papers and return a set of completed years"""
    if not os.path.exists(OUTPUT_FILE):
        return set(), []
        
    try:
        df = pd.read_csv(OUTPUT_FILE)
        df['year'] = pd.to_datetime(df['submitted_date']).dt.year
        
        # Count papers per year
        counts = df.groupby('year').size().to_dict()
        completed = set()
        
        for year, count in counts.items():
            if count >= 800:  # If we have at least 800 papers, consider that year complete
                completed.add(year)
                
        print(f"✓ Found existing output file with {len(df)} papers.")
        print(f"✓ {len(completed)} years already completed: {completed}")
        return completed, df.drop(columns=['year']).to_dict('records')
    except Exception as e:
        print(f"⚠️ Error loading existing file ({e}). Starting fresh.")
        return set(), []

def save_progress(papers):
    """Save collected papers to CSV"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = pd.DataFrame(papers)
    df = df.drop_duplicates(subset=['arxiv_id'])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved total of {len(df)} unique papers to {OUTPUT_FILE}")

def main():
    print(f"\n{'='*70}")
    print(f"UNBIASED ARXIV 10-YEAR DATA COLLECTION: 10,000 PAPERS")
    print(f"{'='*70}")
    
    completed_years, all_papers = load_existing_progress()
    
    # Construct category query union
    cat_query = " OR ".join([f"cat:{c}" for c in CATEGORIES.keys()])
    
    for year in YEARS:
        if year in completed_years:
            print(f"⏭️ Skipping Year {year} (already complete)")
            continue
            
        print(f"\nFetching papers for Year {year}...")
        year_papers = []
        
        for start in range(0, PAPERS_PER_YEAR, BATCH_SIZE):
            # Query for the union of categories in a specific year
            search_query = f"({cat_query}) AND submittedDate:[{year}01010000 TO {year}12312359]"
            params = {
                'search_query': search_query,
                'start': start,
                'max_results': BATCH_SIZE,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                response = requests.get("http://export.arxiv.org/api/query?", params=params, timeout=60)
                if response.status_code == 200:
                    parsed = parse_xml_response(response.text)
                    year_papers.extend(parsed)
                    print(f"  -> Fetched {len(parsed)} matching papers (start={start})")
                    
                    if not parsed:
                        break
                else:
                    print(f"  ✗ API Error {response.status_code}")
                    
            except Exception as e:
                print(f"  ✗ Request failed: {e}")
                
            time.sleep(3.5)  # Rate limiting
            
        if year_papers:
            all_papers.extend(year_papers)
            save_progress(all_papers)
        else:
            print(f"⚠️ No papers retrieved for year {year}")
            
    print(f"\n{'='*70}")
    print(f"✓ DATA COLLECTION COMPLETE!")
    print(f"Final dataset contains {len(all_papers)} unique papers.")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
