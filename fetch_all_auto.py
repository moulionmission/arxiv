import requests
import time
import pandas as pd
import xml.etree.ElementTree as ET
import os
from datetime import datetime

def fetch_category(category_code, category_label, num_papers=2000):
    """Fetch 2000 papers for ONE category"""
    
    BASE_URL = "http://export.arxiv.org/api/query?"
    
    print(f"\n{'='*70}")
    print(f"FETCHING {category_label.upper()}: {num_papers} papers")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    
    papers = []
    
    try:
        params = {
            'search_query': f'cat:{category_code}',
            'start': 0,
            'max_results': num_papers,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        print("Sending request...")
        response = requests.get(BASE_URL, params=params, timeout=180)
        
        if response.status_code == 200:
            print("✓ Got response")
            print("Parsing XML...")
            
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            count = 0
            for entry in root.findall('atom:entry', ns):
                try:
                    paper = {
                        'arxiv_id': entry.find('atom:id', ns).text.split('/abs/')[-1],
                        'title': entry.find('atom:title', ns).text.strip(),
                        'abstract': entry.find('atom:summary', ns).text.strip(),
                        'submitted_date': entry.find('atom:published', ns).text,
                        'label': category_label
                    }
                    papers.append(paper)
                    count += 1
                    
                    if count % 500 == 0:
                        print(f"  {count} papers processed...")
                except:
                    continue
            
            print(f"✓ Got {count} papers from {category_label}")
            return papers
            
        elif response.status_code == 429:
            print(f"⚠️  Rate limited (429). Waiting 180 seconds...")
            time.sleep(180)
            print("Retrying...")
            return fetch_category(category_code, category_label, num_papers)
        else:
            print(f"✗ Error: Status {response.status_code}")
            return []
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return []


def append_to_dataset(papers, category_label):
    """Append papers to master CSV"""
    
    os.makedirs('data/raw', exist_ok=True)
    csv_file = 'data/raw/arxiv_master.csv'
    
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        existing_ids = set(existing_df['arxiv_id'].values)
        print(f"Found {len(existing_ids)} existing papers")
    else:
        existing_df = None
        existing_ids = set()
    
    new_df = pd.DataFrame(papers)
    new_df = new_df[~new_df['arxiv_id'].isin(existing_ids)]
    
    print(f"Adding {len(new_df)} new papers from {category_label}")
    
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    
    combined.to_csv(csv_file, index=False)
    return combined


def print_progress(df):
    """Print progress"""
    print(f"\n{'='*70}")
    print(f"PROGRESS UPDATE")
    print(f"{'='*70}")
    print(f"Total papers: {len(df)}")
    print(f"Target: 10,000")
    print(f"Progress: {len(df)/100:.1f}%")
    print(f"\nBreakdown:")
    print(df['label'].value_counts().to_string())
    print(f"{'='*70}\n")


def wait_with_countdown(seconds, label):
    """Wait with countdown timer"""
    print(f"\n⏳ WAITING {seconds}s BEFORE {label}")
    print(f"Go grab coffee! ☕")
    print(f"{'='*70}\n")
    
    for i in range(seconds, 0, -10):
        remaining = min(10, i)
        mins = i // 60
        secs = i % 60
        print(f"  {mins}:{secs:02d} remaining... ({datetime.now().strftime('%H:%M:%S')})", end='\r')
        time.sleep(remaining)
    
    print(f"✓ READY FOR {label}!                              \n")


def main():
    print(f"\n{'='*70}")
    print(f"ARXIV AUTO-FETCHER: 10,000 PAPERS")
    print(f"{'='*70}")
    print("10 minute wait between each category")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")
    
    categories = [
        ('cs.CL', 'NLP'),
        ('cs.LG', 'ML'),
        ('cs.AI', 'AI'),
        ('cs.CV', 'CV'),
        ('cs.RO', 'Robotics')
    ]
    
    for i, (cat_code, cat_label) in enumerate(categories):
        # Fetch
        print(f"\n[{i+1}/5] Fetching {cat_label}...")
        papers = fetch_category(cat_code, cat_label, num_papers=2000)
        
        if papers:
            # Append
            df = append_to_dataset(papers, cat_label)
            print_progress(df)
            
            # Wait 10 minutes before next (except last)
            if i < len(categories) - 1:
                next_label = categories[i+1][1]
                wait_with_countdown(600, next_label)  # 600 seconds = 10 minutes
        else:
            print(f"⚠️  Failed to fetch {cat_label}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✓ FETCH COMPLETE!")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    
    final_df = pd.read_csv('data/raw/arxiv_master.csv')
    print(f"Total papers: {len(final_df)}")
    print(f"\nFinal breakdown:")
    print(final_df['label'].value_counts().to_string())
    print(f"\n✓ Ready for preprocessing!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
