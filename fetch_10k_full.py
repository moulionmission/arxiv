import requests
import time
import pandas as pd
import xml.etree.ElementTree as ET
import os

print("FETCHING 10,000 PAPERS NOW!")
print("="*60)
print("⏱️  Estimated time: 90-120 minutes")
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
papers_per_category = 2000

for category_code, category_label in categories.items():
    print(f"\n{'='*60}")
    print(f"Fetching {category_label} (2000 papers)...")
    print(f"{'='*60}")
    
    try:
        params = {
            'search_query': f'cat:{category_code} AND submittedDate:[200401010000 TO 202412312359]',
            'start': 0,
            'max_results': papers_per_category,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        print("Sending request (waiting for response)...")
        response = requests.get(BASE_URL, params=params, timeout=120)
        
        if response.status_code == 200:
            print(f"✓ Got response (status 200)")
            print("Parsing XML...")
            
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            count = 0
            for entry in root.findall('atom:entry', ns):
                try:
                    arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
                    title = entry.find('atom:title', ns).text.strip()
                    abstract = entry.find('atom:summary', ns).text.strip()
                    submitted_date = entry.find('atom:published', ns).text
                    
                    paper = {
                        'arxiv_id': arxiv_id,
                        'title': title,
                        'abstract': abstract,
                        'submitted_date': submitted_date,
                        'label': category_label
                    }
                    all_papers.append(paper)
                    count += 1
                    
                    if count % 500 == 0:
                        print(f"  {count} papers processed...")
                        
                except Exception as e:
                    continue
            
            print(f"✓ {category_label} complete: {count} papers")
            
        elif response.status_code == 429:
            print(f"⚠️  Rate limited (429). Waiting 60 seconds...")
            time.sleep(60)
            
        else:
            print(f"✗ Error: Status {response.status_code}")
        
        # Wait between categories
        print(f"Waiting 30 seconds before next category...")
        time.sleep(30)
        
    except requests.exceptions.Timeout:
        print(f"✗ Request timeout for {category_label}. Retrying...")
        time.sleep(60)
    except Exception as e:
        print(f"✗ Error: {e}")

print(f"\n{'='*60}")
print(f"FETCH COMPLETE!")
print(f"{'='*60}")
print(f"Total papers fetched: {len(all_papers)}")

if len(all_papers) > 0:
    os.makedirs('data/raw', exist_ok=True)
    
    df = pd.DataFrame(all_papers)
    df.to_csv('data/raw/arxiv_10k_papers.csv', index=False)
    
    print(f"\n✓ Saved to: data/raw/arxiv_10k_papers.csv")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nDate range:")
    print(f"  Earliest: {df['submitted_date'].min()}")
    print(f"  Latest: {df['submitted_date'].max()}")
    print(f"\n✓ Ready for preprocessing!")
else:
    print("✗ No papers fetched.")
