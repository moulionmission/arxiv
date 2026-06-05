import requests
import time
import pandas as pd
from datetime import datetime

print("Fetching REAL papers from arXiv...")
print("="*60)

BASE_URL = "http://export.arxiv.org/api/query?"

categories = {
    'cs.AI': 'AI',
    'cs.LG': 'ML',
    'cs.CL': 'NLP',
    'cs.CV': 'Computer Vision',
    'cs.RO': 'Robotics'
}

all_papers = []

for category_code, category_label in categories.items():
    print(f"\nFetching {category_label} papers from arXiv...")
    
    # Fetch 100 papers per category (500 total = realistic)
    params = {
        'search_query': f'cat:{category_code}',
        'start': 0,
        'max_results': 100,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            # Parse XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            
            count = 0
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                try:
                    paper = {
                        'arxiv_id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/abs/')[-1],
                        'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
                        'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text.strip(),
                        'submitted_date': entry.find('{http://www.w3.org/2005/Atom}published').text,
                        'label': category_label
                    }
                    all_papers.append(paper)
                    count += 1
                except:
                    continue
            
            print(f"  ✓ Fetched {count} papers")
            
        time.sleep(3)  # Rate limiting
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

print("\n" + "="*60)
print(f"Total papers fetched: {len(all_papers)}")

# Save to CSV
import os
os.makedirs('data/raw', exist_ok=True)

df = pd.DataFrame(all_papers)
df.to_csv('data/raw/real_arxiv_papers.csv', index=False)

print(f"✓ Saved to: data/raw/real_arxiv_papers.csv")
print(f"\nClass distribution:")
print(df['label'].value_counts())
