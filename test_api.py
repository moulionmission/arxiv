from data_loader import ArxivDataLoader

print("Testing arXiv API...")
loader = ArxivDataLoader()

# Fetch small test batch from AI category only
print("Fetching 100 papers from cs.AI...")
loader.fetch_papers_by_category('cs.AI', max_results=100)

print(f"✓ Success! Fetched {len(loader.papers)} papers")
print(f"Sample paper: {loader.papers[0]['title']}")