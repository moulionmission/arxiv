import pandas as pd
from statistical_analysis import TopicEvolutionAnalysis, CitationTrendAnalysis
import shutil
import os

print("\n" + "="*70)
print("RUNNING STATISTICAL ANALYSIS ON UNBIASED 10-YEAR DATASET")
print("="*70 + "\n")

processed_path = 'data/processed/arxiv_10yr_10k_unbiased_cleaned.csv'

if not os.path.exists(processed_path):
    print(f"❌ Error: Processed data file {processed_path} not found.")
    exit(1)

# Load data
df = pd.read_csv(processed_path)
print(f"Loaded {len(df)} papers.")

# Instantiate Topic Evolution Analysis
analysis = TopicEvolutionAnalysis(df)

# Run trend analysis and print summary
print("\nTopic Evolution Analysis (Granularity: Year):")
analysis.print_summary()

# Plot topic trends by year
analysis.plot_topic_trends(granularity='year')

# Rename the plot to a dataset-specific file name
source_plot = 'results/topic_evolution.png'
dest_plot = 'results/topic_evolution_10yr_10k_unbiased.png'
if os.path.exists(source_plot):
    shutil.copy(source_plot, dest_plot)
    print(f"✓ Corrected trend visualization saved to: {dest_plot}")

print("\n" + "="*70)
print("✓ STATISTICAL ANALYSIS COMPLETE!")
print("="*70 + "\n")
