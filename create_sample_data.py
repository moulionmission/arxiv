import pandas as pd
import random
from datetime import datetime, timedelta

print("Creating sample dataset for testing...")

categories = ['AI', 'ML', 'NLP', 'Computer Vision', 'Robotics']

papers = []
for i in range(1000):
    category = random.choice(categories)
    date = datetime.now() - timedelta(days=random.randint(0, 730))
    
    abstract = f"""This paper presents a novel approach to {category} research. 
    We propose a method that significantly improves performance metrics. 
    Our experimental results demonstrate state-of-the-art performance on benchmark datasets. 
    The approach is evaluated on multiple datasets and compared with existing methods.
    We achieve substantial improvements over previous work in this domain."""
    
    papers.append({
        'arxiv_id': f'2024.{i:05d}',
        'title': f'Novel {category} Method {i}',
        'abstract': abstract,
        'authors': ['Author A', 'Author B'],
        'submitted_date': date.isoformat(),
        'primary_category': 'cs.AI',
        'label': category
    })

df = pd.DataFrame(papers)

# Create data folder if needed
import os
os.makedirs('data/raw', exist_ok=True)

df.to_csv('data/raw/sample_papers.csv', index=False)

print(f"✓ Created {len(df)} sample papers")
print(f"✓ Saved to: data/raw/sample_papers.csv")
print(f"\nClass distribution:")
print(df['label'].value_counts())