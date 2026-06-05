import requests
import time
import pandas as pd
import xml.etree.ElementTree as ET

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
    print(f"Fetching {category_label} papers (200 each)...")
    
    try:
        params = {
            'search_query': f'cat:{category_code}',
            'start': 0,
            'max_results': 200,  # 2x more
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        response = requests.get(BASE_URL, params=params, timeout=60)
        
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                try:
                    paper = {
                        'arxiv_id': entry.find('atom:id', ns).text.split('/abs/')[-1],
                        'title': entry.find('atom:title', ns).text.strip(),
                        'abstract': entry.find('atom:summary', ns).text.strip(),
                        'submitted_date': entry.find('atom:published', ns).text,
                        'label': category_label
                    }
                    all_papers.append(paper)
                except:
                    continue
            
            print(f"  ✓ Got {len([p for p in all_papers if p['label'] == category_label])} papers")
        
        time.sleep(15)
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

print(f"\n✓ Total: {len(all_papers)} papers")

import os
os.makedirs('data/raw', exist_ok=True)

df = pd.DataFrame(all_papers)
df.to_csv('data/raw/real_arxiv_papers_1k.csv', index=False)
print("✓ Saved!")
print(df['label'].value_counts())
