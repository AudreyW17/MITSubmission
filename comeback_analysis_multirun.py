'''
Author: Audrey Wang (with assistance in debugging and refactoring from AI)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import re
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class MultiRunComebackAnalyzer:
    """Multi-run comeback analysis for tennis prediction models"""
    
    def __init__(self, base_output_dir="multirun_comeback_analysis"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.all_results = []
        
    def parse_score_detailed(self, score_str):
        """Parse ATP score string into individual sets"""
        if pd.isna(score_str) or not score_str:
            return []
        
        # Handle retirements and walkovers
        if 'RET' in score_str or 'W/O' in score_str or 'DEF' in score_str:
            if 'W/O' in score_str or 'DEF' in score_str:
                return []
            else:
                score_clean = re.sub(r'\s+(RET|W/O|DEF).*', '', score_str)
                if not score_clean.strip():
                    return []
        else:
            score_clean = score_str
        
        # Split into sets
        sets = score_clean.strip().split()
        parsed_sets = []
        
        for set_score in sets:
            if not set_score or set_score in ['RET', 'W/O', 'DEF']:
                continue
                
            # Extract games (ignore tiebreak scores)
            clean_set = re.sub(r'\([^)]*\)', '', set_score)
            
            if '-' in clean_set:
                try:
                    winner_games, loser_games = map(int, clean_set.split('-'))
                    parsed_sets.append((winner_games, loser_games))
                except ValueError:
                    continue
        
        return parsed_sets
    
    def is_comeback_match(self, sets):
        """Determine if match is a 1-2 → 3-2 comeback"""
        if len(sets) != 5:  # Must be exactly 5 sets
            return False
        
        # Calculate set wins from winner's perspective
        set_wins = []
        for winner_games, loser_games in sets:
            set_wins.append(1 if winner_games > loser_games else 0)
        
        # Check for 1-2 → 3-2 pattern
        sets_after_3 = sum(set_wins[:3])  # Winner's sets after 3 sets
        final_sets = sum(set_wins)        # Winner's final sets
        
        return sets_after_3 == 1 and final_sets == 3
    
    def run_single_comeback_experiment(self, csv_file, run_id, year):
        """Run single comeback analysis experiment"""
        
        print(f"\nComeback Run {run_id} for {year}")
        
        try:
            # Import predictor
            from progressive_tennis_predictor import ProgressiveTennisPredictor
            
            # Initialize with time-based seed for true randomization
            import time
            seed = int(time.time() * 1000) % 2**32 + run_id * 1000
            np.random.seed(seed)
            
            predictor = ProgressiveTennisPredictor()
            
            # Load and prepare data
            df = predictor.load_and_prepare_data(csv_file)
            
            # Different train/test split each run for comeback analysis
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
            
            print(f"Training: {len(train_df)} matches")
            print(f"Test: {len(test_df)} matches")
            
            # Train models
            #predictor.train_progressive_models(train_df, epochs=50, batch_size=32)
            predictor.train_progressive_models(train_df, epochs=100, batch_size=64) # same as baseline         
            
            # Find comeback matches in test set
            comeback_matches = []
            for idx, row in test_df.iterrows():
                sets = self.parse_score_detailed(row['score'])
                if self.is_comeback_match(sets):
                    comeback_matches.append({
                        'match_id': idx,
                        'winner': row['winner_name'],
                        'loser': row['loser_name'],
                        'surface': row['surface'],
                        'score': row['score'],
                        'sets': sets
                    })
            
            print(f"Found {len(comeback_matches)} comeback matches")
            
            if len(comeback_matches) == 0:
                # Return empty result if no comeback matches
                return {
                    'run_id': run_id,
                    'year': year,
                    'seed': seed,
                    'total_comeback_matches': 0,
                    'stages': {},
                    'timestamp': datetime.now().isoformat()
                }
            
            # Test predictions on comeback matches
            stage_results = {}
            stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
            
            for stage in stages:
                correct_predictions = 0
                total_predictions = 0
                
                for match in comeback_matches:
                    winner = match['winner']
                    loser = match['loser']
                    surface = match['surface']
                    sets = match['sets']
                    
                    # Determine sets for this stage
                    if stage == 'pre_match':
                        sets_so_far = []
                    elif stage == 'after_set_1':
                        sets_so_far = sets[:1]
                    elif stage == 'after_set_2':
                        sets_so_far = sets[:2]
                    elif stage == 'after_set_3':
                        sets_so_far = sets[:3]
                    elif stage == 'after_set_4':
                        sets_so_far = sets[:4]
                    
                    try:
                        # Make prediction
                        prediction = predictor.predict_progressive_match(
                            winner, loser, surface, best_of=5, sets_so_far=sets_so_far
                        )
                        
                        # Check if prediction is correct
                        predicted_winner = prediction['predicted_winner']
                        is_correct = predicted_winner == winner
                        
                        correct_predictions += is_correct
                        total_predictions += 1
                        
                    except Exception as e:
                        print(f"Warning: {stage} prediction failed - {str(e)}")
                        continue
                
                # Calculate accuracy for this stage
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                stage_results[stage] = {
                    'correct': correct_predictions,
                    'total': total_predictions,
                    'accuracy': accuracy
                }
                
                print(f"{stage}: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
            
            # Save this run's results
            run_result = {
                'run_id': run_id,
                'year': year,
                'seed': seed,
                'total_comeback_matches': len(comeback_matches),
                'stages': stage_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save individual run
            self.save_single_run(run_result, run_id, year)
            
            print(f"Comeback run {run_id} completed")
            return run_result
            
        except Exception as e:
            print(f"Comeback run {run_id} failed: {str(e)}")
            
            # Return failed run
            return {
                'run_id': run_id,
                'year': year,
                'seed': -1,
                'total_comeback_matches': 0,
                'stages': {},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_single_run(self, run_result, run_id, year):
        """Save individual run results"""

        filename = self.base_output_dir / f"comeback_run_{run_id}_{year}.json"
        with open(filename, 'w') as f:
            json.dump(run_result, f, indent=2)
    
    def run_multiple_comeback_experiments(self, data_files, n_runs=10):
        """Run multiple comeback experiments across all years"""
        
        print(f"MULTI-RUN COMEBACK ANALYSIS")
        print(f"Years: {list(data_files.keys())}")
        print(f"Runs per year: {n_runs}")
        
        for year, csv_file in data_files.items():
            print(f"\nPROCESSING YEAR: {year}")
            print(f"Data file: {csv_file}")
            
            for run_id in range(1, n_runs + 1):
                run_result = self.run_single_comeback_experiment(csv_file, run_id, year)
                self.all_results.append(run_result)
        
        print(f"\nAll comeback experiments completed")
        print(f"Total runs: {len(self.all_results)}")
    
    def calculate_comeback_statistics(self):
        """Calculate comprehensive comeback statistics"""
        
        print(f"\nCALCULATING COMEBACK STATISTICS")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.all_results)
        
        # Filter successful runs
        successful_runs = df[df['total_comeback_matches'] > 0]
        
        if len(successful_runs) == 0:
            print("No successful comeback runs found")
            return None
        
        print(f"Successful runs: {len(successful_runs)}/{len(df)}")
        
        # Calculate statistics by year and stage
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        
        summary_stats = []
        
        for year in successful_runs['year'].unique():
            year_data = successful_runs[successful_runs['year'] == year]
            
            print(f"\nStatistics for {year}:")
            print("-" * 30)
            
            year_stats = {'year': year}
            
            for stage in stages:
                # Extract accuracies for this stage
                accuracies = []
                total_matches = []
                
                for _, row in year_data.iterrows():
                    if stage in row['stages'] and row['stages'][stage]['total'] > 0:
                        accuracies.append(row['stages'][stage]['accuracy'])
                        total_matches.append(row['stages'][stage]['total'])
                
                if accuracies:
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    min_acc = np.min(accuracies)
                    max_acc = np.max(accuracies)
                    median_matches = np.median(total_matches)
                    
                    year_stats.update({
                        f'{stage}_mean': mean_acc,
                        f'{stage}_std': std_acc,
                        f'{stage}_min': min_acc,
                        f'{stage}_max': max_acc,
                        f'{stage}_count': len(accuracies),
                        f'{stage}_median_matches': median_matches
                    })
                    
                    print(f"{stage}: {mean_acc:.1%} ± {std_acc:.1%} "
                          f"(n={len(accuracies)}, range: {min_acc:.1%}-{max_acc:.1%})")
                else:
                    print(f"{stage}: No data")
            
            summary_stats.append(year_stats)
        
        return pd.DataFrame(summary_stats)
    
    def create_comeback_visualizations(self, stats_df):
        """Create comprehensive comeback visualizations"""
                
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        stage_labels = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Tennis Comeback Prediction Analysis\n(1-2 → 3-2 Comeback Scenarios)', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        # Plot 1: Mean accuracy by year and stage
        years = sorted(stats_df['year'].unique())
        x = np.arange(len(stage_labels))
        width = 0.2
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(years)))
        
        for i, year in enumerate(years):
            year_data = stats_df[stats_df['year'] == year].iloc[0]
            
            means = [year_data.get(f'{stage}_mean', 0) for stage in stages]
            stds = [year_data.get(f'{stage}_std', 0) for stage in stages]
            
            bars = ax1.bar(x + i*width, means, width, yerr=stds, 
                          capsize=3, label=f'{year}', color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar, mean_val in zip(bars, means):
                if mean_val > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{mean_val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_title('Comeback Prediction Accuracy by Stage\n(Mean ± Standard Deviation)', 
                      fontweight='bold', fontsize=14)
        ax1.set_xlabel('Match Stage', fontweight='bold')
        ax1.set_ylabel('Prediction Accuracy', fontweight='bold')
        ax1.set_xticks(x + width * (len(years) - 1) / 2)
        ax1.set_xticklabels(stage_labels, rotation=45, ha='right')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Cross-year average with emphasis on challenging stages
        cross_year_means = []
        cross_year_stds = []
        
        for stage in stages:
            stage_data = []
            for year in years:
                year_row = stats_df[stats_df['year'] == year]
                if not year_row.empty and f'{stage}_mean' in year_row.iloc[0]:
                    stage_data.append(year_row.iloc[0][f'{stage}_mean'])
            
            if stage_data:
                cross_year_means.append(np.mean(stage_data))
                cross_year_stds.append(np.std(stage_data))
            else:
                cross_year_means.append(0)
                cross_year_stds.append(0)
        
        # Color bars by performance level
        bar_colors = []
        for mean_val in cross_year_means:
            if mean_val >= 0.8:
                bar_colors.append('green')
            elif mean_val >= 0.6:
                bar_colors.append('orange')
            else:
                bar_colors.append('red')
        
        bars2 = ax2.bar(range(len(stage_labels)), cross_year_means, 
                       yerr=cross_year_stds, capsize=5, 
                       color=bar_colors, alpha=0.8, edgecolor='black')
        
        # Highlight the challenging after_set_3 stage
        bars2[3].set_edgecolor('red')
        bars2[3].set_linewidth(3)
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars2, cross_year_means, cross_year_stds):
            if mean_val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.05,
                       f'{mean_val:.0%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_title('Cross-Year Average Comeback Prediction\n(The After-Set-3 Challenge)', 
                      fontweight='bold', fontsize=14)
        ax2.set_xlabel('Match Stage', fontweight='bold')
        ax2.set_ylabel('Average Accuracy', fontweight='bold')
        ax2.set_xticks(range(len(stage_labels)))
        ax2.set_xticklabels(stage_labels, rotation=45, ha='right')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1.1)
        
        # Add annotation for after_set_3 challenge
        ax2.annotate('MOST CHALLENGING\nStage for Comebacks', 
                    xy=(3, cross_year_means[3]), xytext=(4.5, 0.3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, ha='center', color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
        
        # Plot 3: Variability analysis (std dev by stage)
        bars3 = ax3.bar(range(len(stage_labels)), cross_year_stds, 
                       color='lightcoral', alpha=0.8, edgecolor='darkred')
        
        for bar, std_val in zip(bars3, cross_year_stds):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{std_val:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('Prediction Variability Across Years\n(Standard Deviation)', 
                      fontweight='bold', fontsize=14)
        ax3.set_xlabel('Match Stage', fontweight='bold')
        ax3.set_ylabel('Standard Deviation', fontweight='bold')
        ax3.set_xticks(range(len(stage_labels)))
        ax3.set_xticklabels(stage_labels, rotation=45, ha='right')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Performance ranking by stage
        # Calculate average ranking for each stage
        stage_rankings = []
        for i, stage in enumerate(stages):
            # Rank stages by average performance
            stage_rankings.append((stage_labels[i], cross_year_means[i], i))
        
        # Sort by performance
        stage_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Create ranking visualization
        ranks = [i+1 for i in range(len(stage_rankings))]
        stage_names = [x[0] for x in stage_rankings]
        performances = [x[1] for x in stage_rankings]
        
        bars4 = ax4.barh(ranks, performances, 
                        color=['gold' if i == 0 else 'silver' if i == 1 else 'lightblue' 
                               for i in range(len(ranks))],
                        alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, perf in zip(bars4, performances):
            ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{perf:.0%}', ha='left', va='center', fontweight='bold')
        
        ax4.set_title('Stage Performance Ranking\n(Best to Worst for Comebacks)', 
                      fontweight='bold', fontsize=14)
        ax4.set_xlabel('Average Accuracy', fontweight='bold')
        ax4.set_ylabel('Rank', fontweight='bold')
        ax4.set_yticks(ranks)
        ax4.set_yticklabels([f"#{r}: {name}" for r, name in zip(ranks, stage_names)])
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.base_output_dir / 'comeback_analysis_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.base_output_dir / 'comeback_analysis_comprehensive.pdf', 
                   bbox_inches='tight')
        plt.close()
        
        print("Comprehensive comeback visualizations saved")
    
    def create_comeback_heatmap(self, stats_df):
        """Create heatmap showing comeback prediction patterns"""
                
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        stage_labels = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
        
        # Prepare heatmap data
        heatmap_data = []
        years = sorted(stats_df['year'].unique())
        
        for year in years:
            year_row = stats_df[stats_df['year'] == year].iloc[0]
            year_data = []
            
            for stage in stages:
                mean_key = f'{stage}_mean'
                if mean_key in year_row and pd.notna(year_row[mean_key]):
                    year_data.append(year_row[mean_key])
                else:
                    year_data.append(0)
            
            heatmap_data.append(year_data)
        
        heatmap_data = np.array(heatmap_data)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(heatmap_data, 
                   annot=True, fmt='.0%', 
                   xticklabels=stage_labels,
                   yticklabels=years,
                   cmap='RdYlGn', 
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Comeback Prediction Accuracy'},
                   linewidths=2, linecolor='white')
        
        plt.title('Comeback Prediction Heatmap\n(1-2 → 3-2 Scenarios Across Years and Stages)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Match Stage', fontsize=14, fontweight='bold')
        plt.ylabel('Year', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.base_output_dir / 'comeback_prediction_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.base_output_dir / 'comeback_prediction_heatmap.pdf', 
                   bbox_inches='tight')
        plt.close()
        
        print("Comeback heatmap saved")
    
    def save_comprehensive_results(self, stats_df):
        """Save all comeback results and statistics"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        raw_df = pd.DataFrame(self.all_results)
        raw_filename = self.base_output_dir / f"comeback_raw_results_{timestamp}.csv"
        raw_df.to_csv(raw_filename, index=False)
        print(f"Raw comeback results: {raw_filename}")
        
        # Save statistics
        stats_filename = self.base_output_dir / f"comeback_statistics_{timestamp}.csv"
        stats_df.to_csv(stats_filename, index=False)
        print(f"Comeback statistics: {stats_filename}")
        
        # Create analysis ready summary
        pub_summary = self.create_publication_summary(stats_df)
        pub_filename = self.base_output_dir / f"comeback_publication_summary_{timestamp}.csv"
        pub_summary.to_csv(pub_filename, index=False)
        print(f"Publication summary: {pub_filename}")
        
        # Save configuration
        config = {
            'analysis_type': 'multirun_comeback_prediction',
            'timestamp': timestamp,
            'total_runs': len(self.all_results),
            'years_analyzed': list(stats_df['year'].unique()),
            'stages_analyzed': ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4'],
            'comeback_definition': '1-2 set deficit becoming 3-2 victory'
        }
        
        config_filename = self.base_output_dir / f"comeback_config_{timestamp}.json"
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration: {config_filename}")
        
        return {
            'raw_file': raw_filename,
            'stats_file': stats_filename,
            'publication_file': pub_filename,
            'config_file': config_filename
        }
    
    def create_publication_summary(self, stats_df):
        """Create summary of comeback results"""
        
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        stage_labels = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
        
        pub_data = []
        
        # Cross-year summary for each stage
        for i, (stage, label) in enumerate(zip(stages, stage_labels)):
            stage_means = []
            stage_stds = []
            
            for year in sorted(stats_df['year'].unique()):
                year_row = stats_df[stats_df['year'] == year]
                if not year_row.empty and f'{stage}_mean' in year_row.columns:
                    mean_val = year_row.iloc[0][f'{stage}_mean']
                    if pd.notna(mean_val):
                        stage_means.append(mean_val)
                        
                        std_val = year_row.iloc[0].get(f'{stage}_std', 0)
                        stage_stds.append(std_val)
            
            if stage_means:
                overall_mean = np.mean(stage_means)
                overall_std = np.mean(stage_stds)  # Average of individual standard deviations
                min_acc = np.min(stage_means)
                max_acc = np.max(stage_means)
                
                pub_data.append({
                    'Stage': label,
                    'Mean_Accuracy': f"{overall_mean:.1%}",
                    'Std_Accuracy': f"{overall_std:.1%}",
                    'Range': f"{min_acc:.1%}-{max_acc:.1%}",
                    'Formatted_Result': f"{overall_mean:.1%} ± {overall_std:.1%}",
                    'Years_Available': len(stage_means),
                    'Rank': None  # Will be filled later
                })
        
        pub_df = pd.DataFrame(pub_data)
        
        # Add ranking
        pub_df['Mean_Numeric'] = [float(x.strip('%'))/100 for x in pub_df['Mean_Accuracy']]
        pub_df = pub_df.sort_values('Mean_Numeric', ascending=False)
        pub_df['Rank'] = range(1, len(pub_df) + 1)
        pub_df = pub_df.drop('Mean_Numeric', axis=1)
        
        return pub_df


def run_multirun_comeback_analysis(data_files, n_runs=10):
    """Main function to run multi-run comeback analysis"""
    
    print("MULTI-RUN COMEBACK ANALYSIS")
    print("Analyzing 1-2 → 3-2 comeback prediction across multiple runs")
    print(f"Years: {list(data_files.keys())}")
    print(f"Runs per year: {n_runs}")
    
    # Initialize analyzer
    analyzer = MultiRunComebackAnalyzer()
    
    # Run experiments
    analyzer.run_multiple_comeback_experiments(data_files, n_runs)
    
    # Calculate statistics
    stats_df = analyzer.calculate_comeback_statistics()
    
    if stats_df is not None:
        # Create visualizations
        analyzer.create_comeback_visualizations(stats_df)
        analyzer.create_comeback_heatmap(stats_df)
        
        # Save results
        saved_files = analyzer.save_comprehensive_results(stats_df)
        
        print(f"\nMULTI-RUN COMEBACK ANALYSIS COMPLETE")
        print(f"Files saved: {len(saved_files)}")
        for file_type, filepath in saved_files.items():
            print(f"{file_type}: {filepath.name}")
        
        # Show preview of key results
        print(f"\nCOMEBACK ANALYSIS PREVIEW:")
        
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        stage_labels = ['Pre-Match', 'After Set 1', 'After Set 2', 'After Set 3', 'After Set 4']
        
        # Calculate cross-year averages
        for stage, label in zip(stages, stage_labels):
            stage_means = []
            for year in sorted(stats_df['year'].unique()):
                year_row = stats_df[stats_df['year'] == year]
                if not year_row.empty:
                    mean_val = year_row.iloc[0].get(f'{stage}_mean', 0)
                    if pd.notna(mean_val) and mean_val > 0:
                        stage_means.append(mean_val)
            
            if stage_means:
                avg_acc = np.mean(stage_means)
                print(f"{label:<15}: {avg_acc:.1%}")
        
        return analyzer, stats_df
    else:
        print("No comeback analysis data generated")
        return analyzer, None


def quick_comeback_test(year='2023', csv_file='tennis_data/atp_matches_2023.csv', n_runs=3):
    """Quick test with fewer runs for debugging"""
    
    print(f"Quick Comeback Test: {n_runs} runs on {year}")
    
    data_files = {year: csv_file}
    return run_multirun_comeback_analysis(data_files, n_runs)


if __name__ == "__main__":
    print("Multi-Run Comeback Analysis")
    print("1. Full analysis (10 runs × 4 years)")
    print("2. Quick test (3 runs × 1 year)")
    print("3. Custom configuration")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Full analysis
        data_files = {
            '2021': 'tennis_data/atp_matches_2021.csv',
            '2022': 'tennis_data/atp_matches_2022.csv',
            '2023': 'tennis_data/atp_matches_2023.csv',
            '2024': 'tennis_data/atp_matches_2024.csv'
        }
        
        print("Notes:")
        print(" - Each run takes 5-10 minutes (includes training)")
        print(" - Total estimated time: 4-6 hours")
        print(" - Comeback matches are rare (5-15 per test set)")
        print(" - Results saved after each run")
        
        response = input(f"\nProceed with full comeback analysis? (y/n): ").strip().lower()
        if response == 'y':
            analyzer, stats = run_multirun_comeback_analysis(data_files, n_runs=10)
        else:
            print("Analysis cancelled.")
            
    elif choice == "2":
        # Quick test
        analyzer, stats = quick_comeback_test()
        
    elif choice == "3":
        # Custom configuration
        year = input("Enter year (2021-2024): ").strip()
        csv_file = input("Enter CSV file path: ").strip()
        n_runs = int(input("Enter number of runs: ").strip())
        
        data_files = {year: csv_file}
        analyzer, stats = run_multirun_comeback_analysis(data_files, n_runs)
        
    else:
        print("Invalid choice. Exiting")
    
    print("\nComeback analysis complete")
    print("Check the output directory for results and visualizations")