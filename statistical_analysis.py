"""
arXiv Paper Classification: Statistical Analysis
Analyze topic evolution, citation trends, and research growth patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')


class TopicEvolutionAnalysis:
    """Analyze how research topics change over time"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with 'submitted_date' and 'label' columns
        """
        self.df = df.copy()
        self.df['submitted_date'] = pd.to_datetime(self.df['submitted_date'])
        self.df['year_month'] = self.df['submitted_date'].dt.to_period('M')
        self.df['year'] = self.df['submitted_date'].dt.year
        self.df['quarter'] = self.df['submitted_date'].dt.to_period('Q')
        
    def trend_by_category(self, granularity: str = 'month'):
        """
        Calculate % of papers in each category over time
        
        Args:
            granularity: 'month', 'quarter', or 'year'
        
        Returns:
            DataFrame with trends
        """
        
        if granularity == 'month':
            time_col = 'year_month'
        elif granularity == 'quarter':
            time_col = 'quarter'
        else:
            time_col = 'year'
        
        # Count papers per category per time period
        trends = self.df.groupby([time_col, 'label']).size().unstack(fill_value=0)
        
        # Convert to percentages
        trends_pct = trends.div(trends.sum(axis=1), axis=0) * 100
        
        return trends, trends_pct
    
    def mann_kendall_test(self, granularity: str = 'month'):
        """
        Mann-Kendall test for monotonic trend in each category
        Tests null hypothesis: no trend exists
        
        Args:
            granularity: 'month', 'quarter', or 'year'
        
        Returns:
            DataFrame with test results
        """
        
        trends, _ = self.trend_by_category(granularity)
        
        results = []
        
        for category in trends.columns:
            series = trends[category].values
            
            # Mann-Kendall test (non-parametric)
            n = len(series)
            s = 0
            
            for i in range(n-1):
                for j in range(i+1, n):
                    s += np.sign(series[j] - series[i])
            
            # Variance calculation
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
            
            # Direction
            trend_direction = 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
            
            results.append({
                'category': category,
                'z_statistic': z,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'direction': trend_direction
            })
        
        return pd.DataFrame(results)
    
    def growth_rate_by_category(self):
        """
        Calculate year-over-year growth rate for each category
        
        Returns:
            DataFrame with growth rates and statistics
        """
        
        yearly = self.df.groupby(['year', 'label']).size().unstack(fill_value=0)
        
        growth_rates = {}
        
        for category in yearly.columns:
            year_counts = yearly[category].values
            
            # Calculate YoY growth
            growth = []
            for i in range(1, len(year_counts)):
                if year_counts[i-1] > 0:
                    yoy_growth = ((year_counts[i] - year_counts[i-1]) / year_counts[i-1]) * 100
                    growth.append(yoy_growth)
            
            growth_rates[category] = {
                'mean_growth': np.mean(growth) if growth else 0,
                'std_growth': np.std(growth) if growth else 0,
                'current_year_count': year_counts[-1],
                'previous_year_count': year_counts[-2] if len(year_counts) > 1 else 0
            }
        
        return pd.DataFrame(growth_rates).T
    
    def diversity_index(self, granularity: str = 'year'):
        """
        Calculate Shannon entropy (diversity) of topic distribution over time
        High entropy = diverse research; Low entropy = concentrated
        
        Args:
            granularity: 'year', 'quarter', or 'month'
        
        Returns:
            DataFrame with diversity indices
        """
        
        if granularity == 'year':
            time_col = 'year'
        elif granularity == 'quarter':
            time_col = 'quarter'
        else:
            time_col = 'year_month'
        
        trends, trends_pct = self.trend_by_category(granularity)
        
        entropy_scores = []
        
        for idx in trends_pct.index:
            proportions = trends_pct.loc[idx] / 100  # Convert back to proportions
            # Remove zeros (log(0) undefined)
            proportions = proportions[proportions > 0]
            entropy = -np.sum(proportions * np.log(proportions))
            entropy_scores.append(entropy)
        
        diversity_df = pd.DataFrame({
            'time_period': trends_pct.index,
            'shannon_entropy': entropy_scores
        })
        
        return diversity_df
    
    def plot_topic_trends(self, granularity: str = 'month', figsize: tuple = (14, 6)):
        """Plot topic evolution over time"""
        
        _, trends_pct = self.trend_by_category(granularity)
        
        plt.figure(figsize=figsize)
        
        for category in trends_pct.columns:
            plt.plot(range(len(trends_pct)), trends_pct[category], 
                    marker='o', label=category, linewidth=2, markersize=4)
        
        plt.xlabel(f'Time ({granularity.capitalize()})')
        plt.ylabel('Percentage of Papers (%)')
        plt.title('Research Topic Evolution Over Time')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/topic_evolution.png', dpi=300)
        print("✓ Topic evolution plot saved")
        plt.close()
    
    def print_summary(self):
        """Print summary statistics"""
        
        print("\n" + "="*70)
        print("TOPIC EVOLUTION ANALYSIS SUMMARY")
        print("="*70)
        
        # Mann-Kendall results
        mk_results = self.mann_kendall_test('year')
        print("\nMann-Kendall Trend Test (Year-over-Year):")
        print(mk_results.to_string())
        
        # Growth rates
        print("\n\nYear-over-Year Growth Rates:")
        growth = self.growth_rate_by_category()
        print(growth.to_string())
        
        # Diversity
        print("\n\nShannon Entropy (Topic Diversity) - Last 3 Years:")
        diversity = self.diversity_index('year')
        print(diversity.tail(3).to_string())


class CitationTrendAnalysis:
    """Analyze citation patterns and trends"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with 'citation_count', 'submitted_date', 'label' columns
                Note: Requires external citation data from Semantic Scholar or S2 API
        """
        self.df = df.copy()
        self.df['submitted_date'] = pd.to_datetime(self.df['submitted_date'])
        self.df['year'] = self.df['submitted_date'].dt.year
        self.df['days_since_publication'] = (
            pd.Timestamp.now() - self.df['submitted_date']
        ).dt.days
    
    def citation_stats_by_category(self):
        """
        Calculate citation statistics per category
        
        Returns:
            DataFrame with citation metrics
        """
        
        stats_dict = {}
        
        for category in self.df['label'].unique():
            cat_data = self.df[self.df['label'] == category]
            
            # Handle missing citation data gracefully
            citations = cat_data['citation_count'].fillna(0)
            
            stats_dict[category] = {
                'mean_citations': citations.mean(),
                'median_citations': citations.median(),
                'std_citations': citations.std(),
                'max_citations': citations.max(),
                'papers_count': len(cat_data),
                'citation_velocity': (
                    citations.sum() / cat_data['days_since_publication'].mean()
                    if cat_data['days_since_publication'].mean() > 0 else 0
                ) * 365  # Citations per year
            }
        
        return pd.DataFrame(stats_dict).T
    
    def anova_citation_differences(self):
        """
        ANOVA test: Are citation counts significantly different across categories?
        Null hypothesis: All categories have same mean citations
        
        Returns:
            F-statistic, p-value, significant (bool)
        """
        
        groups = [
            self.df[self.df['label'] == cat]['citation_count'].fillna(0).values
            for cat in self.df['label'].unique()
        ]
        
        # Use Kruskal-Wallis (non-parametric) if data non-normal
        h_stat, p_value = kruskal(*groups)
        
        return {
            'test': 'Kruskal-Wallis H-test',
            'h_statistic': h_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05
        }
    
    def citation_velocity_trend(self):
        """
        Calculate citation accumulation speed by category
        Newer papers cited faster = higher velocity
        
        Returns:
            DataFrame with velocity metrics
        """
        
        velocities = []
        
        for category in self.df['label'].unique():
            cat_data = self.df[self.df['label'] == category]
            
            # Segment by publication year
            for year in sorted(cat_data['year'].unique())[-3:]:  # Last 3 years
                year_data = cat_data[cat_data['year'] == year]
                
                avg_citations = year_data['citation_count'].fillna(0).mean()
                avg_days = year_data['days_since_publication'].mean()
                
                # Velocity = citations per 100 days
                velocity = (avg_citations / avg_days * 100) if avg_days > 0 else 0
                
                velocities.append({
                    'category': category,
                    'year': year,
                    'citation_velocity': velocity,
                    'avg_citations': avg_citations
                })
        
        return pd.DataFrame(velocities)
    
    def plot_citation_distribution(self, figsize: tuple = (14, 6)):
        """Box plot of citation distribution by category"""
        
        plt.figure(figsize=figsize)
        
        data_for_plot = []
        labels_for_plot = []
        
        for category in sorted(self.df['label'].unique()):
            citations = self.df[self.df['label'] == category]['citation_count'].fillna(0)
            data_for_plot.append(citations)
            labels_for_plot.append(category)
        
        bp = plt.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        plt.ylabel('Citation Count')
        plt.title('Citation Distribution by Research Category')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('results/citation_distribution.png', dpi=300)
        print("✓ Citation distribution plot saved")
        plt.close()
    
    def print_summary(self):
        """Print citation analysis summary"""
        
        print("\n" + "="*70)
        print("CITATION TREND ANALYSIS SUMMARY")
        print("="*70)
        
        print("\nCitation Statistics by Category:")
        stats = self.citation_stats_by_category()
        print(stats.to_string())
        
        print("\n\nANOVA Test - Citation Differences Across Categories:")
        anova = self.anova_citation_differences()
        for key, val in anova.items():
            print(f"{key}: {val}")
        
        print("\n\nCitation Velocity (Citations per 100 Days):")
        velocity = self.citation_velocity_trend()
        print(velocity.to_string())


# Main execution
if __name__ == "__main__":
    
    print("Loading dataset...")
    df = pd.read_csv('data/processed/arxiv_papers_cleaned.csv')
    
    # Topic Evolution Analysis
    print("\n" + "="*70)
    print("ANALYZING TOPIC EVOLUTION...")
    print("="*70)
    
    topic_analysis = TopicEvolutionAnalysis(df)
    topic_analysis.plot_topic_trends(granularity='quarter')
    topic_analysis.print_summary()
    
    # Citation Analysis (if citation_count available)
    if 'citation_count' in df.columns:
        print("\n" + "="*70)
        print("ANALYZING CITATION TRENDS...")
        print("="*70)
        
        citation_analysis = CitationTrendAnalysis(df)
        citation_analysis.plot_citation_distribution()
        citation_analysis.print_summary()
    else:
        print("\nNote: Citation data not available. To enable citation analysis:")
        print("1. Integrate Semantic Scholar API or S2 API")
        print("2. Fetch citation counts for each arXiv paper")
        print("3. Add 'citation_count' column to dataset")
