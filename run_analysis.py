import pandas as pd
from statistical_analysis import TopicEvolutionAnalysis

df = pd.read_csv('data/processed/real_arxiv_1k_cleaned.csv')

print("\n" + "="*70)
print("TOPIC EVOLUTION ANALYSIS")
print("="*70)

analysis = TopicEvolutionAnalysis(df)
analysis.plot_topic_trends(granularity='year')
analysis.print_summary()
