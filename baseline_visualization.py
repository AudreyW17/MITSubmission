'''
Author: Audrey Wang (with assistance in debugging and refactoring from AI)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_baseline_visualizations(publication_csv_file):
    """Create visualizations from baseline multi-run results"""
    
    # Read the data
    df = pd.read_csv(publication_csv_file)
    
    print(f"Creating visualizations from: {publication_csv_file}")
    print(f"Data shape: {df.shape}")
    print(f"Years: {sorted(df['Year'].unique())}")
    print(f"Stages: {df['Stage'].unique()}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir = Path("baseline_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate all visualization types
    create_combined_line_chart(df, output_dir)
    create_accuracy_heatmap(df, output_dir)
    create_summary_statistics_table(df, output_dir)
    create_error_bar_chart(df, output_dir)
    create_improvement_analysis(df, output_dir)
    
    print(f"\nAll visualizations saved in: {output_dir}/")
    return output_dir

def create_combined_line_chart(df, output_dir):
    """Create line chart showing all years together"""
        
    # Prepare data
    stage_order = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
    years = sorted(df['Year'].unique())
    #colors = plt.cm.Set2(np.linspace(0, 1, len(years)))
    colors =['orange', 'blue', 'magenta', 'gray']
   
    plt.figure(figsize=(12, 8))

    # Plot line for each year
    #for year in sorted(df['Year'].unique()):
    for i, year in enumerate(years):
        year_data = df[df['Year'] == year]
        
        # Order by stage
        year_data = year_data.set_index('Stage').reindex(stage_order).reset_index()
        
        plt.plot(stage_order, year_data['Mean_Accuracy'], 
                marker='o', color=colors[i], linewidth=2.5, markersize=8, label=f'{year}')
        
        # Add error bars
        plt.errorbar(stage_order, year_data['Mean_Accuracy'], 
                    yerr=year_data['Std_Accuracy'], 
                    alpha=0.3, linewidth=1)
    
    # Formatting
    plt.title('Progressive Tennis Prediction Accuracy Across Match Stages\n(Multi-Year Transformer Baseline)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Match Stage', fontsize=14, fontweight='bold')
    plt.ylabel('Prediction Accuracy (%)', fontsize=14, fontweight='bold')
    
    # Y-axis formatting
    plt.ylim(0.75, 1.0)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Year', fontsize=12, title_fontsize=13, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Annotations for key insights
    plt.annotate('Peak Performance\n(Set leads decisive)', 
                xy=('After Set 3', 0.97), xytext=(2.5, 0.98),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, ha='center', color='green')
    
    plt.annotate('Challenging Scenario\n(Extended matches)', 
                xy=('After Set 4', 0.91), xytext=(4.5, 0.85),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, ha='center', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_combined_line_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'baseline_combined_line_chart.pdf', bbox_inches='tight')
    plt.close()
    

def create_accuracy_heatmap(df, output_dir):
    """Create heatmap showing accuracy across years and stages"""
        
    # Prepare data for heatmap
    stage_order = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
    
    # Pivot the data
    heatmap_data = df.pivot(index='Year', columns='Stage', values='Mean_Accuracy')
    heatmap_data = heatmap_data.reindex(columns=stage_order)
    
    # Create the heatmap
    plt.figure(figsize=(12, 6))
    
    # Create with custom colormap
    sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='RdYlGn', 
                cbar_kws={'label': 'Prediction Accuracy'}, 
                linewidths=0.5, linecolor='white')
    
    plt.title('Tennis Prediction Accuracy Heatmap\n(Baseline Transformer Results)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Match Stage', fontsize=14, fontweight='bold')
    plt.ylabel('Year', fontsize=14, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'baseline_accuracy_heatmap.pdf', bbox_inches='tight')
    plt.close()
    

def create_summary_statistics_table(df, output_dir):
    """Create summary statistics table and visualization"""
        
    # Calculate cross-year statistics
    stage_order = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
    
    summary_stats = []
    for stage in stage_order:
        stage_data = df[df['Stage'] == stage]
        
        summary_stats.append({
            'Stage': stage,
            'Mean_Accuracy': stage_data['Mean_Accuracy'].mean(),
            'Std_Accuracy': stage_data['Mean_Accuracy'].std(),
            'Min_Accuracy': stage_data['Mean_Accuracy'].min(),
            'Max_Accuracy': stage_data['Mean_Accuracy'].max(),
            'Range': stage_data['Mean_Accuracy'].max() - stage_data['Mean_Accuracy'].min(),
            'Consistency': 1 - stage_data['Mean_Accuracy'].std()  # Higher = more consistent
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save summary table
    summary_df.to_csv(output_dir / 'baseline_summary_statistics.csv', index=False)
    
    # Create visualization of summary stats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean accuracy with error bars
    ax1.bar(range(len(stage_order)), summary_df['Mean_Accuracy'], 
           yerr=summary_df['Std_Accuracy'], capsize=5, 
           color='skyblue', edgecolor='navy', alpha=0.7)
    
    ax1.set_title('Average Accuracy Across All Years\n(with Standard Deviation)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Match Stage', fontweight='bold')
    ax1.set_ylabel('Mean Accuracy', fontweight='bold')
    ax1.set_xticks(range(len(stage_order)))
    ax1.set_xticklabels(stage_order, rotation=45, ha='right')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean_acc, std_acc) in enumerate(zip(summary_df['Mean_Accuracy'], summary_df['Std_Accuracy'])):
        ax1.text(i, mean_acc + std_acc + 0.005, f'{mean_acc:.1%}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Range and consistency
    ax2_twin = ax2.twinx()
    
    bars = ax2.bar(range(len(stage_order)), summary_df['Range'], 
                  color='lightcoral', alpha=0.7, label='Accuracy Range')
    line = ax2_twin.plot(range(len(stage_order)), summary_df['Consistency'], 
                        'o-', color='darkgreen', linewidth=2, markersize=8, 
                        label='Consistency Score')
    
    ax2.set_title('Cross-Year Variability Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Match Stage', fontweight='bold')
    ax2.set_ylabel('Accuracy Range (Max - Min)', fontweight='bold', color='darkred')
    ax2_twin.set_ylabel('Consistency Score', fontweight='bold', color='darkgreen')
    ax2.set_xticks(range(len(stage_order)))
    ax2.set_xticklabels(stage_order, rotation=45, ha='right')
    
    # Format axes
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax2_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    
    # Legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_summary_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'baseline_summary_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    return summary_df

def create_error_bar_chart(df, output_dir):
    """Create detailed error bar chart by year and stage"""
        
    stage_order = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
    years = sorted(df['Year'].unique())
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate positions for grouped bars
    x = np.arange(len(stage_order))
    width = 0.2
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(years)))
    
    for i, year in enumerate(years):
        year_data = df[df['Year'] == year]
        year_data = year_data.set_index('Stage').reindex(stage_order).reset_index()
        
        positions = x + i * width
        
        bars = ax.bar(positions, year_data['Mean_Accuracy'], width, 
                     yerr=year_data['Std_Accuracy'], capsize=3,
                     label=f'{year}', color=colors[i], alpha=0.8,
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for pos, mean_acc, std_acc in zip(positions, year_data['Mean_Accuracy'], year_data['Std_Accuracy']):
            ax.text(pos, mean_acc + std_acc + 0.005, f'{mean_acc:.1%}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_title('Multi-Run Baseline Results: Progressive Tennis Prediction\n(Mean ± Standard Deviation)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Match Stage', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(years) - 1) / 2)
    ax.set_xticklabels(stage_order)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_detailed_error_bars.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'baseline_detailed_error_bars.pdf', bbox_inches='tight')
    plt.close()
    

def create_improvement_analysis(df, output_dir):
    """Analyze and visualize the improvement patterns"""
        
    # Calculate improvement from pre-match to each stage
    stage_order = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
    
    improvement_data = []
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]
        year_data = year_data.set_index('Stage').reindex(stage_order).reset_index()
        
        pre_match_acc = year_data.loc[year_data['Stage'] == 'Pre-Match', 'Mean_Accuracy'].iloc[0]
        
        for _, row in year_data.iterrows():
            if row['Stage'] != 'Pre-Match':
                improvement = row['Mean_Accuracy'] - pre_match_acc
                improvement_data.append({
                    'Year': year,
                    'Stage': row['Stage'],
                    'Improvement': improvement,
                    'Relative_Improvement': improvement / pre_match_acc
                })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Create improvement visualization
    plt.figure(figsize=(12, 8))
    
    for year in sorted(improvement_df['Year'].unique()):
        year_data = improvement_df[improvement_df['Year'] == year]
        plt.plot(year_data['Stage'], year_data['Improvement'], 
                marker='o', linewidth=2.5, markersize=8, label=f'{year}')
    
    plt.title('Prediction Accuracy Improvement Over Pre-Match Baseline\n(Progressive Information Gain)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Match Stage', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy Improvement vs Pre-Match', fontsize=14, fontweight='bold')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1%}'))
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Year')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Highlight the "After Set 3" peak and "After Set 4" drop
    plt.annotate('Information Peak\n(+15-20%)', 
                xy=('After Set 3', 0.18), xytext=('After Set 2', 0.22),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=11, ha='center', color='green', fontweight='bold')
    
    plt.annotate('Challenging Drop\n(Extended matches)', 
                xy=('After Set 4', 0.13), xytext=('After Set 3', 0.08),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=11, ha='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'baseline_improvement_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # Save improvement data
    improvement_df.to_csv(output_dir / 'improvement_analysis_data.csv', index=False)
    
    print("Improvement analysis saved")
    
    return improvement_df

def create_summary_table(publication_csv_file, output_dir):
    """Create a clean summary table"""
    
    df = pd.read_csv(publication_csv_file)
    
    # Create summary
    stage_order = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
    
    pub_summary = []
    for stage in stage_order:
        stage_data = df[df['Stage'] == stage]
        
        # Calculate overall statistics
        mean_acc = stage_data['Mean_Accuracy'].mean()
        overall_std = stage_data['Mean_Accuracy'].std()
        min_acc = stage_data['Mean_Accuracy'].min()
        max_acc = stage_data['Mean_Accuracy'].max()
        
        # Create formatted string
        formatted_result = f"{mean_acc:.1%} ± {overall_std:.1%}"
        range_str = f"{min_acc:.1%}-{max_acc:.1%}"
        
        pub_summary.append({
            'Match_Stage': stage,
            'Mean_Accuracy': f"{mean_acc:.1%}",
            'Cross_Year_Std': f"{overall_std:.1%}",
            'Range': range_str,
            'Formatted_Result': formatted_result,
            'Sample_Size': 'n=10 per year',
            'Years_Tested': '2021-2024'
        })
    
    pub_df = pd.DataFrame(pub_summary)
    pub_df.to_csv(output_dir / 'publication_ready_summary.csv', index=False)
    
    return pub_df

# Main function
def analyze_baseline_results(publication_csv_file):
    print("BASELINE RESULTS ANALYSIS")
    
    if not Path(publication_csv_file).exists():
        print(f"File not found: {publication_csv_file}")
        return None
    
    # Create all visualizations
    output_dir = create_baseline_visualizations(publication_csv_file)
    
    # Create publication summary
    create_summary_table(publication_csv_file, output_dir)
    
    print("\nANALYSIS COMPLETE")
    print(f"All visualizations and tables saved in: {output_dir}/")
    
    print("\nFILES CREATED:")
    for file in sorted(output_dir.glob('*')):
        print(f"{file.name}")
    
    print("1. 'baseline_combined_line_chart.pdf' as main visualization")
    print("2. 'baseline_improvement_analysis.pdf' to highlight the Set-3 peak pattern")
    print("3. 'baseline_summary_statistics.csv' for cross-year consistency")
    
    return output_dir

if __name__ == "__main__":
    # Example usage
    print("Baseline Results Visualization")
    
    # Need to replace this with actual file path
    # csv_file = input("Enter path to publication CSV file: ").strip()
    csv_file = "baseline_multirun_results/baseline_transformer_publication_20250731_224103.csv"
    
    if csv_file:
        analyze_baseline_results(csv_file)
    else:
        print("Please provide the path to your publication CSV file.")
        print("Example: baseline_multirun_results/baseline_transformer_publication_20240101_120000.csv")
