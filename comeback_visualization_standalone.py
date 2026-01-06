'''
Author: Audrey Wang (with assistance in debugging and refactoring from AI)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ComebackVisualizationGenerator:
    """Generate comeback visualizations from existing multi-run results"""
    
    def __init__(self, results_dir="multirun_comeback_analysis"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("multirun_comeback_visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_existing_results(self, stats_csv_file=None, raw_csv_file=None):
        """Load results from existing CSV files"""
        
        print("Loading existing comeback results...")
        
        # If specific files not provided, find the most recent ones
        if stats_csv_file is None:
            stats_files = list(self.results_dir.glob("comeback_statistics_*.csv"))
            if stats_files:
                stats_csv_file = max(stats_files, key=lambda x: x.stat().st_mtime)
                print(f"Found statistics file: {stats_csv_file.name}")
            else:
                raise FileNotFoundError("No comeback statistics CSV files found")
        
        if raw_csv_file is None:
            raw_files = list(self.results_dir.glob("comeback_raw_results_*.csv"))
            if raw_files:
                raw_csv_file = max(raw_files, key=lambda x: x.stat().st_mtime)
                print(f"Found raw results file: {raw_csv_file.name}")
            else:
                raw_csv_file = None
        
        # Load statistics
        self.stats_df = pd.read_csv(stats_csv_file)
        print(f"Loaded statistics: {len(self.stats_df)} year entries")
        print(f"Years available: {sorted(self.stats_df['year'].unique())}")
        
        # Load raw results if available
        if raw_csv_file and raw_csv_file.exists():
            self.raw_df = pd.read_csv(raw_csv_file)
            print(f"Loaded raw results: {len(self.raw_df)} individual runs")
        else:
            self.raw_df = None
            print("Raw results file not available")
        
        return self.stats_df, self.raw_df
    
    def create_comprehensive_comeback_plots(self, style='publication', figsize=(18, 14)):
        """Create comprehensive 4-panel comeback visualization"""
                
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        stage_labels = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
        
        # Set style
        if style == 'publication':
            plt.style.use('default')
            plt.rcParams.update({
                'font.size': 11,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 11
            })
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Tennis Comeback Prediction Analysis: Multi-Run Results\n(1-2 → 3-2 Comeback Scenarios)', 
                     fontsize=16, fontweight='bold', y=0.96)
        
        # Plot 1: Mean accuracy by year and stage
        self._create_year_comparison_plot(ax1, stages, stage_labels)
        
        # Plot 2: Cross-year average with challenge identification
        self._create_cross_year_average_plot(ax2, stages, stage_labels)
        
        # Plot 3: Variability analysis
        self._create_variability_analysis_plot(ax3, stages, stage_labels)
        
        # Plot 4: Performance ranking
        self._create_performance_ranking_plot(ax4, stages, stage_labels)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save plots
        self._save_plot(fig, 'comeback_comprehensive_analysis')
        plt.close()
        
        print("Comprehensive comeback plots saved")
    
    def _create_year_comparison_plot(self, ax, stages, stage_labels):
        """Create year-by-year comparison subplot"""
        
        years = sorted(self.stats_df['year'].unique())
        x = np.arange(len(stage_labels))
        width = 0.18
        
        # Distinct colors for each year
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, year in enumerate(years):
            year_data = self.stats_df[self.stats_df['year'] == year]
            if year_data.empty:
                continue
                
            year_row = year_data.iloc[0]
            
            means = []
            stds = []
            
            for stage in stages:
                mean_key = f'{stage}_mean'
                std_key = f'{stage}_std'
                
                mean_val = year_row.get(mean_key, 0)
                std_val = year_row.get(std_key, 0)
                
                means.append(mean_val if pd.notna(mean_val) else 0)
                stds.append(std_val if pd.notna(std_val) else 0)
            
            # Create bars
            bars = ax.bar(x + i*width, means, width, yerr=stds, 
                         capsize=3, label=f'{year}', color=colors[i % len(colors)], 
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels (only for non-zero values)
            for bar, mean_val in zip(bars, means):
                if mean_val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                           f'{mean_val:.0%}', ha='center', va='bottom', 
                           fontsize=8, fontweight='bold')
        
        ax.set_title('Comeback Prediction by Year and Stage\n(Mean ± Standard Deviation)', 
                     fontweight='bold')
        ax.set_xlabel('Match Stage', fontweight='bold')
        ax.set_ylabel('Prediction Accuracy', fontweight='bold')
        ax.set_xticks(x + width * (len(years) - 1) / 2)
        ax.set_xticklabels(stage_labels, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(title='Year', loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
    
    def _create_cross_year_average_plot(self, ax, stages, stage_labels):
        """Create cross-year average subplot with challenge highlighting"""
        
        cross_year_means = []
        cross_year_stds = []
        sample_counts = []
        
        years = sorted(self.stats_df['year'].unique())
        
        for stage in stages:
            stage_data = []
            stage_stds = []
            
            for year in years:
                year_row = self.stats_df[self.stats_df['year'] == year]
                if not year_row.empty:
                    mean_val = year_row.iloc[0].get(f'{stage}_mean', np.nan)
                    std_val = year_row.iloc[0].get(f'{stage}_std', 0)
                    
                    if pd.notna(mean_val) and mean_val > 0:
                        stage_data.append(mean_val)
                        stage_stds.append(std_val)
            
            if stage_data:
                cross_year_means.append(np.mean(stage_data))
                cross_year_stds.append(np.mean(stage_stds))  # Average of std devs
                sample_counts.append(len(stage_data))
            else:
                cross_year_means.append(0)
                cross_year_stds.append(0)
                sample_counts.append(0)
        
        # Color bars by performance level
        bar_colors = []
        for mean_val in cross_year_means:
            if mean_val >= 0.8:
                bar_colors.append('#2ca02c')  # Green for good
            elif mean_val >= 0.6:
                bar_colors.append('#ff7f0e')  # Orange for moderate
            elif mean_val > 0:
                bar_colors.append('#d62728')  # Red for poor
            else:
                bar_colors.append('#cccccc')  # Gray for no data
        
        bars = ax.bar(range(len(stage_labels)), cross_year_means, 
                     yerr=cross_year_stds, capsize=5, 
                     color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Highlight challenging stages with special markers
        for i, (mean_val, count) in enumerate(zip(cross_year_means, sample_counts)):
            if mean_val > 0:
                # Add value label
                ax.text(i, mean_val + cross_year_stds[i] + 0.05,
                       f'{mean_val:.0%}', ha='center', va='bottom', 
                       fontsize=11, fontweight='bold')
                
                # Add sample size annotation
                ax.text(i, mean_val/2, f'n={count}', ha='center', va='center',
                       fontsize=9, color='white', fontweight='bold')
        
        # Find and annotate the most challenging stage
        if cross_year_means:
            # Exclude zero values when finding minimum
            non_zero_means = [(i, val) for i, val in enumerate(cross_year_means) if val > 0]
            if non_zero_means:
                min_idx, min_val = min(non_zero_means, key=lambda x: x[1])
                
                ax.annotate('MOST CHALLENGING\nfor Comebacks', 
                           xy=(min_idx, min_val), xytext=(min_idx + 1, 0.25),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           fontsize=10, ha='center', color='red', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
        
        ax.set_title('Cross-Year Average Performance\n(Identifying Challenge Points)', 
                     fontweight='bold')
        ax.set_xlabel('Match Stage', fontweight='bold')
        ax.set_ylabel('Average Accuracy', fontweight='bold')
        ax.set_xticks(range(len(stage_labels)))
        ax.set_xticklabels(stage_labels, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
    
    def _create_variability_analysis_plot(self, ax, stages, stage_labels):
        """Create variability analysis subplot"""
        
        # Calculate cross-year standard deviations
        cross_year_stds = []
        years = sorted(self.stats_df['year'].unique())
        
        for stage in stages:
            stage_means = []
            
            for year in years:
                year_row = self.stats_df[self.stats_df['year'] == year]
                if not year_row.empty:
                    mean_val = year_row.iloc[0].get(f'{stage}_mean', np.nan)
                    if pd.notna(mean_val) and mean_val > 0:
                        stage_means.append(mean_val)
            
            if len(stage_means) > 1:
                cross_year_stds.append(np.std(stage_means))
            elif len(stage_means) == 1:
                cross_year_stds.append(0)
            else:
                cross_year_stds.append(0)
        
        # Create bars with color coding by variability level
        var_colors = []
        for std_val in cross_year_stds:
            if std_val > 0.15:
                var_colors.append('#d62728')  # High variability - red
            elif std_val > 0.05:
                var_colors.append('#ff7f0e')  # Medium variability - orange
            else:
                var_colors.append('#2ca02c')  # Low variability - green
        
        bars = ax.bar(range(len(stage_labels)), cross_year_stds, 
                     color=var_colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, std_val in zip(bars, cross_year_stds):
            if std_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{std_val:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Add horizontal lines for interpretation
        ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Low Variability')
        ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='High Variability')
        
        ax.set_title('Cross-Year Prediction Variability\n(Standard Deviation Between Years)', 
                     fontweight='bold')
        ax.set_xlabel('Match Stage', fontweight='bold')
        ax.set_ylabel('Standard Deviation', fontweight='bold')
        ax.set_xticks(range(len(stage_labels)))
        ax.set_xticklabels(stage_labels, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _create_performance_ranking_plot(self, ax, stages, stage_labels):
        """Create performance ranking subplot"""
        
        # Calculate average performance for ranking
        stage_performances = []
        years = sorted(self.stats_df['year'].unique())
        
        for i, stage in enumerate(stages):
            stage_means = []
            
            for year in years:
                year_row = self.stats_df[self.stats_df['year'] == year]
                if not year_row.empty:
                    mean_val = year_row.iloc[0].get(f'{stage}_mean', np.nan)
                    if pd.notna(mean_val) and mean_val > 0:
                        stage_means.append(mean_val)
            
            if stage_means:
                avg_performance = np.mean(stage_means)
                stage_performances.append((stage_labels[i], avg_performance, len(stage_means)))
        
        # Sort by performance (descending)
        stage_performances.sort(key=lambda x: x[1], reverse=True)
        
        # Extract data for plotting
        labels = [x[0] for x in stage_performances]
        performances = [x[1] for x in stage_performances]
        sample_sizes = [x[2] for x in stage_performances]
        
        # Create horizontal bar chart
        y_pos = range(len(labels))
        
        # Color by rank
        colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#4169E1', '#8A2BE2']
        
        bars = ax.barh(y_pos, performances, 
                      color=colors[:len(labels)], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, perf, n_samples in zip(bars, performances, sample_sizes):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{perf:.0%} (n={n_samples})', ha='left', va='center', fontweight='bold')
        
        # Add ranking numbers
        for i, (bar, label) in enumerate(zip(bars, labels)):
            rank_symbol = ['1st', '2nd', '3rd', '4th', '5th'][i] if i < 5 else f'{i+1}th'
            ax.text(0.02, bar.get_y() + bar.get_height()/2,
                   rank_symbol, ha='left', va='center', fontweight='bold', 
                   fontsize=12, color='white')
        
        ax.set_title('Performance Ranking by Stage\n(Best to Worst for Comeback Prediction)', 
                     fontweight='bold')
        ax.set_xlabel('Average Accuracy', fontweight='bold')
        ax.set_ylabel('Match Stage (Ranked)', fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{label}" for label in labels])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, max(performances) * 1.2 if performances else 1)
    
    def create_comeback_heatmap(self, figsize=(12, 8)):
        """Create comeback prediction heatmap"""
                
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        stage_labels = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
        
        # Prepare heatmap data
        years = sorted(self.stats_df['year'].unique())
        heatmap_data = []
        
        for year in years:
            year_row = self.stats_df[self.stats_df['year'] == year]
            if year_row.empty:
                heatmap_data.append([0] * len(stages))
                continue
                
            year_data = []
            for stage in stages:
                mean_val = year_row.iloc[0].get(f'{stage}_mean', 0)
                year_data.append(mean_val if pd.notna(mean_val) else 0)
            
            heatmap_data.append(year_data)
        
        heatmap_data = np.array(heatmap_data)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, fmt='.0%', 
                   xticklabels=stage_labels,
                   yticklabels=years,
                   cmap='RdYlGn', 
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Comeback Prediction Accuracy'},
                   linewidths=2, linecolor='white',
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        # Highlight cells with special borders for extreme values
        for i in range(len(years)):
            for j in range(len(stage_labels)):
                value = heatmap_data[i, j]
                if value > 0.9: # Excellent performance
                    plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                                     edgecolor='gold', lw=3))
                elif value > 0 and value < 0.4: # Poor performance
                    plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                                     edgecolor='red', lw=3))
        
        plt.title('Comeback Prediction Accuracy Heatmap\n(1-2 → 3-2 Scenarios: Multi-Run Results)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Match Stage', fontsize=14, fontweight='bold')
        plt.ylabel('Year', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        self._save_plot(plt.gcf(), 'comeback_prediction_heatmap')
        plt.close()
        
        print("Comeback heatmap saved")
    
    def create_publication_summary_plot(self, figsize=(14, 8)):
        """Create a clean summary plot"""
                
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        stage_labels = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
        
        # Calculate cross-year statistics
        cross_year_means = []
        cross_year_cis = []  # Confidence intervals
        sample_counts = []
        
        years = sorted(self.stats_df['year'].unique())
        
        for stage in stages:
            stage_data = []
            
            for year in years:
                year_row = self.stats_df[self.stats_df['year'] == year]
                if not year_row.empty:
                    mean_val = year_row.iloc[0].get(f'{stage}_mean', np.nan)
                    if pd.notna(mean_val) and mean_val > 0:
                        stage_data.append(mean_val)
            
            if stage_data:
                mean_acc = np.mean(stage_data)
                std_acc = np.std(stage_data)
                n = len(stage_data)
                
                # 95% confidence interval
                ci = 1.96 * std_acc / np.sqrt(n) if n > 1 else 0
                
                cross_year_means.append(mean_acc)
                cross_year_cis.append(ci)
                sample_counts.append(n)
            else:
                cross_year_means.append(0)
                cross_year_cis.append(0)
                sample_counts.append(0)
        
        # Create publication plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bars with professional styling
        bar_colors = []
        for mean_val in cross_year_means:
            if mean_val >= 0.8:
                bar_colors.append('#2ca02c')  # Green for good
            elif mean_val >= 0.6:
                bar_colors.append('#ff7f0e')  # Orange for moderate
            elif mean_val > 0:
                bar_colors.append('#d62728')  # Red for poor
            else:
                bar_colors.append('#cccccc')  # Gray for no data
        
        bars = ax.bar(range(len(stage_labels)), cross_year_means, 
                     yerr=cross_year_cis, capsize=8, 
                     color=bar_colors, alpha=0.8, edgecolor='navy', linewidth=2)
        
        # Add value labels with confidence intervals
        for i, (bar, mean_val, ci, count) in enumerate(zip(bars, cross_year_means, cross_year_cis, sample_counts)):
            if mean_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 0.05,
                       f'{mean_val:.1%}\n±{ci:.1%}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12)
                """
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                       f'n={count}', ha='center', va='center', 
                       color='white', fontweight='bold', fontsize=11)
                """
        # Styling
        ax.set_title('Comeback Prediction Accuracy: Cross-Year Analysis\n(1-2 → 3-2 Recovery Scenarios)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Match Prediction Stage', fontsize=14, fontweight='bold')
        ax.set_ylabel('Prediction Accuracy (Mean ± 95% CI)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(stage_labels)))
        ax.set_xticklabels(stage_labels, rotation=45, ha='right', fontsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, 1.1)
        
        """
        # Add subtle background shading for performance zones
        ax.axhspan(0.8, 1.0, alpha=0.1, color='green', label='Excellent (>80%)')
        ax.axhspan(0.6, 0.8, alpha=0.1, color='yellow', label='Good (60-80%)')
        ax.axhspan(0.0, 0.6, alpha=0.1, color='red', label='Challenging (<60%)')
        
        # Add legend for zones
        ax.legend(loc='upper right', title='Performance Zones', framealpha=0.9)
        """
        
        plt.tight_layout()
        self._save_plot(fig, 'comeback_publication_summary')
        plt.close()
        
        print("Publication summary plot saved")

    def create_year_stage_plot(self, figsize=(14, 8)):
               
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        stage_labels = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']

        years = sorted(self.stats_df['year'].unique())
        x = np.arange(len(stage_labels))
        width = 0.18  # Narrower bars for better fit
       
        fig, ax = plt.subplots(figsize=figsize)

        # Use distinct colors for each year
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
       
        for i, year in enumerate(years):
           year_data = self.stats_df[self.stats_df['year'] == year]
           if year_data.empty:
               continue
               
           year_row = year_data.iloc[0]
           
           means = []
           stds = []
           
           for stage in stages:
               mean_key = f'{stage}_mean'
               std_key = f'{stage}_std'
               
               mean_val = year_row.get(mean_key, 0)
               std_val = year_row.get(std_key, 0)
               
               means.append(mean_val if pd.notna(mean_val) else 0)
               stds.append(std_val if pd.notna(std_val) else 0)
           
           # Create bars
           bars = ax.bar(x + i*width, means, width, yerr=stds, 
                        capsize=3, label=f'{year}', color=colors[i % len(colors)], 
                        alpha=0.8, edgecolor='black', linewidth=0.5)
           
           # Add value labels (only for non-zero values)
           for bar, mean_val in zip(bars, means):
               if mean_val > 0:
                   ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                          f'{mean_val:.0%}', ha='center', va='bottom', 
                          fontsize=8, fontweight='bold')
       
        ax.set_title('Comeback Prediction by Year and Stage\n(Mean ± Standard Deviation)', 
                    fontweight='bold')
        ax.set_xlabel('Match Stage', fontweight='bold')
        ax.set_ylabel('Prediction Accuracy', fontweight='bold')
        ax.set_xticks(x + width * (len(years) - 1) / 2)
        ax.set_xticklabels(stage_labels, rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(title='Year', loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        self._save_plot(fig, 'comeback_year_stage_comparison')
        plt.close()

        print("Comeback year-stage plot saved")
    
    def _save_plot(self, fig, filename):
        """Save plot in multiple formats"""
        fig.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{filename}.pdf', bbox_inches='tight')
    
    def generate_all_visualizations(self, stats_csv=None, raw_csv=None):
        """Generate all comeback visualizations from existing data"""
                
        # Load existing results
        self.load_existing_results(stats_csv, raw_csv)
        
        # Generate all plots
        self.create_comprehensive_comeback_plots()
        self.create_comeback_heatmap()
        self.create_publication_summary_plot()
        self.create_year_stage_plot()
        
        print(f"\nALL VISUALIZATIONS GENERATED")
        print(f"Output directory: {self.output_dir}")
        print(f"Files created:")
        
        for file in sorted(self.output_dir.glob('*')):
            print(f"{file.name}")
        
        return self.output_dir


def update_comeback_visualizations(results_dir="multirun_comeback_analysis", 
                                  stats_csv=None, raw_csv=None):
    visualizer = ComebackVisualizationGenerator(results_dir)
    return visualizer.generate_all_visualizations(stats_csv, raw_csv)


if __name__ == "__main__":
    print("Comeback Visualization Generator")
    print("Generate updated visualizations from existing comeback analysis results")
    
    # Get results directory
    #results_dir = input("Enter results directory (or press Enter for 'multirun_comeback_analysis'): ").strip()
    #if not results_dir:
    #    results_dir = "multirun_comeback_analysis"
    results_dir = "multirun_comeback_analysis"

    # Check if directory exists
    if not Path(results_dir).exists():
        print(f"Directory {results_dir} not found!")
        print("Available directories:")
        for d in Path('.').iterdir():
            if d.is_dir() and 'comeback' in d.name.lower():
                print(f"{d.name}")
        exit()
    
    print(f"Using results directory: {results_dir}")
    
    # Generate visualizations
    try:
        output_dir = update_comeback_visualizations(results_dir)
        print(f"\nVisualizations updated successfully")
        print(f"Check {output_dir} for updated plots.")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
