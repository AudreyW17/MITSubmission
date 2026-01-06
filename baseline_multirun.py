'''
Author: Audrey Wang (with assistance in debugging and refactoring from AI)
'''

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

class BaselineMultiRunAnalyzer:
    """Multi-run analysis for baseline transformer training"""
    
    def __init__(self, base_output_dir="baseline_multirun_results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.all_results = []
        
    def run_single_baseline_experiment(self, csv_file, run_id, year):
        print(f"\n Run {run_id} for {year}")
        
        try:
            from progressive_tennis_predictor import ProgressiveTennisPredictor
            
            # Different random state each run
            import time
            import random
            seed = int(time.time() * 1000) % 2**32 + run_id
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
                        
            predictor = ProgressiveTennisPredictor()
            
            # Load and prep data
            df = predictor.load_and_prepare_data(csv_file)
            
            if len(df) < 100:
                raise ValueError(f"Insufficient data: only {len(df)} matches")
            
            # Split data for training and testing
            train_df, test_df = train_test_split(df, test_size=0.1, random_state=seed)
            
            # Train model
            predictor.train_progressive_models(train_df, epochs=100, batch_size=64)
            
            run_results = {
                'run_id': run_id,
                'year': year,
                'timestamp': datetime.now().isoformat(),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'seed_used': seed
            }
            
            # Extract accuracy for each trained stage
            stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
            
            for stage in stages:
                if stage in predictor.models:
                    run_results[f'{stage}_accuracy'] = predictor.models[stage]['test_accuracy']
                else:
                    run_results[f'{stage}_accuracy'] = 0.0
            
            # Save individual run results
            self.save_single_run(run_results, run_id, year)
            return run_results
            
        except Exception as e:            
            failed_result = {
                'run_id': run_id,
                'year': year,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'train_samples': 0,
                'test_samples': 0
            }
            
            stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
            for stage in stages:
                failed_result[f'{stage}_accuracy'] = 0.0
            
            return failed_result
    
    def save_single_run(self, run_results, run_id, year):
        """Save run results to JSON"""
        filename = self.base_output_dir / f"baseline_run_{run_id}_{year}.json"
        with open(filename, 'w') as f:
            json.dump(run_results, f, indent=2)
    
    def run_multiple_baseline_experiments(self, data_files, n_runs=10):
        """Multiple baseline across all years"""
        print(f"STARTING BASELINE ANALYSIS")
        print(f"Years: {list(data_files.keys())}")
        print(f"Runs per year: {n_runs}")
        
        total_experiments = len(data_files) * n_runs
        completed = 0
        
        for year, csv_file in data_files.items():
            print(f"\nYEAR: {year}")
            
            for run_id in range(1, n_runs + 1):

                run_results = self.run_single_baseline_experiment(csv_file, run_id, year)
                self.all_results.append(run_results)
                
                completed += 1
                progress = completed / total_experiments * 100
                print(f"Progress: {completed}/{total_experiments} ({progress:.1f}%)")
        
        print(f"\nALL RUNS COMPLETED")
        print(f"Total: {len(self.all_results)}")
    
    def calculate_baseline_statistics(self):
        """Calculate statistics for baseline results"""
        print(f"\nCALCULATING BASELINE STATISTICS")
            
        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)
        accuracy_columns = [col for col in df.columns if col.endswith('_accuracy')]
        
        # Group by year and calculate statistics
        stats_results = []
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            
            print(f"\nBaseline Statistics for {year}:")
            print("-" * 40)
            
            year_stats = {'year': year}
            
            # Overall run statistics
            total_runs = len(year_data)
            failed_runs = len(year_data.get('error', pd.Series()).isna())
            successful_runs = total_runs - failed_runs
            
            year_stats.update({
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate': successful_runs / total_runs if total_runs > 0 else 0
            })
            
            print(f"Runs: {successful_runs}/{total_runs} successful ({successful_runs/total_runs:.1%})")
            
            # Calculate statistics for each accuracy metric
            for metric in accuracy_columns:

                # Only use successful runs
                values = year_data[metric].dropna()
                values = values[values > 0]  # Remove failed runs
                    
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    min_val = values.min()
                    max_val = values.max()
                    count = len(values)
                        
                # Confidence interval (95%)
                if count > 1:
                    se = std_val / np.sqrt(count)
                    ci_95 = 1.96 * se
                else:
                    ci_95 = 0
                        
                year_stats.update({
                    f'{metric}_mean': mean_val,
                    f'{metric}_std': std_val,
                    f'{metric}_min': min_val,
                    f'{metric}_max': max_val,
                    f'{metric}_count': count,
                    f'{metric}_ci_95': ci_95
                })
                        
                # Show improvement pattern
                stage_name = metric.replace('_accuracy', '').replace('_', ' ').title()
                print(f"{stage_name:<12}: {mean_val:.1%} ± {std_val:.1%} "
                      f"(n={count}, range: {min_val:.1%}-{max_val:.1%})")
                   
            stats_results.append(year_stats)
        
        return pd.DataFrame(stats_results)
    
    def create_baseline_publication_table(self, stats_df):
        """Create table for results"""

        stage_mapping = {
            'pre_match': 'Pre-Match',
            'after_set_1': 'After Set 1', 
            'after_set_2': 'After Set 2',
            'after_set_3': 'After Set 3',
            'after_set_4': 'After Set 4'
        }
        
        pub_data = []
        
        for _, row in stats_df.iterrows():
            year = row['year']
            
            for stage_key, stage_name in stage_mapping.items():
                mean_col = f'{stage_key}_accuracy_mean'
                std_col = f'{stage_key}_accuracy_std'
                count_col = f'{stage_key}_accuracy_count'
                min_col = f'{stage_key}_accuracy_min'
                max_col = f'{stage_key}_accuracy_max'
                ci_col = f'{stage_key}_accuracy_ci_95'
                
                if mean_col in row and pd.notna(row[mean_col]):
                    pub_data.append({
                        'Year': year,
                        'Stage': stage_name,
                        'Mean_Accuracy': row[mean_col],
                        'Std_Accuracy': row[std_col] if std_col in row else 0,
                        'Min_Accuracy': row[min_col] if min_col in row else 0,
                        'Max_Accuracy': row[max_col] if max_col in row else 0,
                        'Count': int(row[count_col]) if count_col in row else 0,
                        'CI_95': row[ci_col] if ci_col in row else 0,
                        'Formatted_Result': f"{row[mean_col]:.1%} ± {row[std_col]:.1%}" if std_col in row and pd.notna(row[std_col]) else f"{row[mean_col]:.1%}",
                        'Range': f"{row[min_col]:.1%}-{row[max_col]:.1%}" if min_col in row and max_col in row else ""
                    })
        
        return pd.DataFrame(pub_data)
    
    def save_baseline_results(self, stats_df, filename_prefix="baseline_transformer"):
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Raw
        raw_df = pd.DataFrame(self.all_results)
        raw_filename = self.base_output_dir / f"{filename_prefix}_raw_{timestamp}.csv"
        raw_df.to_csv(raw_filename, index=False)
        
        # Statistics summary
        stats_filename = self.base_output_dir / f"{filename_prefix}_stats_{timestamp}.csv"
        stats_df.to_csv(stats_filename, index=False)
        
        # Table
        pub_summary = self.create_baseline_publication_table(stats_df)
        pub_filename = self.base_output_dir / f"{filename_prefix}_publication_{timestamp}.csv"
        pub_summary.to_csv(pub_filename, index=False)
        
        # Progress Summary
        progress_summary = self.create_progress_summary(pub_summary)
        progress_filename = self.base_output_dir / f"{filename_prefix}_progress_{timestamp}.csv"
        progress_summary.to_csv(progress_filename, index=False)
        
        config_info = {
            'analysis_type': 'baseline_transformer_multirun',
            'analysis_timestamp': timestamp,
            'total_runs': len(self.all_results),
            'stages_tested': ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4'],
            'training_method': 'transformer_progressive_tennis_predictor',
            'metrics_calculated': ['mean', 'std', 'min', 'max', 'count', 'ci_95']
        }
        
        config_filename = self.base_output_dir / f"{filename_prefix}_config_{timestamp}.json"
        with open(config_filename, 'w') as f:
            json.dump(config_info, f, indent=2)

        return {
            'raw_file': raw_filename,
            'stats_file': stats_filename,
            'publication_file': pub_filename,
            'progress_file': progress_filename,
            'config_file': config_filename
        }
    
    def create_progress_summary(self, pub_df):
        """Create summary showing accuracy across stages"""
        
        progress_data = []
        
        for year in pub_df['Year'].unique():
            year_data = pub_df[pub_df['Year'] == year].sort_values('Stage')
            progress_row = {'Year': year}
            
            for _, row in year_data.iterrows():
                stage = row['Stage'].replace(' ', '_').lower()
                progress_row[f'{stage}_accuracy'] = row['Formatted_Result']
                progress_row[f'{stage}_mean'] = row['Mean_Accuracy']
            
            progress_data.append(progress_row)
        
        return pd.DataFrame(progress_data)


def run_baseline_multirun_analysis(data_files, n_runs=10):
    """Main function to run analysis"""
    
    # Initialize
    analyzer = BaselineMultiRunAnalyzer()
    
    response = input(f"\nProceed with {n_runs} runs per year? (y/n): ").strip().lower()
    if response != 'y':
        print("Analysis cancelled.")
        return None, None
    
    analyzer.run_multiple_baseline_experiments(data_files, n_runs=n_runs)
    stats_df = analyzer.calculate_baseline_statistics()
    saved_files = analyzer.save_baseline_results(stats_df)
    
    print(f"\nBASELINE MULTI-RUN ANALYSIS COMPLETE")
    print(f"Total experiments: {len(analyzer.all_results)}")
    print(f"Files generated: {len(saved_files)}")
    for file_type, filepath in saved_files.items():
        print(f"{file_type}: {filepath.name}")
    
    print(f"\nPREVIEW RESULTS:")
    pub_df = analyzer.create_baseline_publication_table(stats_df)
    for year in sorted(pub_df['Year'].unique()):
        year_data = pub_df[pub_df['Year'] == year]
        print(f"\n{year}:")
        for _, row in year_data.iterrows():
            print(f"{row['Stage']:<12}: {row['Formatted_Result']}")
    
    return analyzer, stats_df


def quick_test_single_year(year='2023', csv_file='tennis_data/atp_matches_2023.csv', n_runs=3):
    """Test with single year and fewer runs for debug"""
    print(f"Quick test: {n_runs} runs on {year}")
    
    if not os.path.exists(csv_file):
        print(f"ERROR: File {csv_file} not found")
        return None, None
    
    analyzer = BaselineMultiRunAnalyzer(base_output_dir=f"test_baseline_{year}")
    data_files = {year: csv_file}
    
    try:
        analyzer.run_multiple_baseline_experiments(data_files, n_runs=n_runs)
        stats_df = analyzer.calculate_baseline_statistics()
        # saved_files = analyzer.save_baseline_results(stats_df, filename_prefix=f"test_{year}")
        return analyzer, stats_df
    except Exception as e:
        print(f"Quick test failed: {str(e)}")
        return analyzer, None


def debug_baseline_setup(data_folder="tennis_data"):
    print("DEBUGGING BASELINE SETUP")
    
    # Check imports
    try:
        import pandas as pd
        import numpy as np
        import torch
    except ImportError as e:
        print(f"Import error: {e}")
        return False, None
    
    # progressive_tennis_predictor is available
    try:
        from progressive_tennis_predictor import ProgressiveTennisPredictor
    except ImportError as e:
        print(f"Cannot import ProgressiveTennisPredictor: {e}")
        return False, None
    
    # Check for CSV files 
    csv_files_current = [f for f in os.listdir('.') if f.endswith('.csv') and 'atp_matches' in f]
    if csv_files_current:
        csv_files = csv_files_current
        data_path = ""
    else:
        # Check tennis_data folder
        if os.path.exists(data_folder):
            csv_files_data = [f for f in os.listdir(data_folder) if f.endswith('.csv') and 'atp_matches' in f]
            csv_files = csv_files_data
            data_path = data_folder + "/" if csv_files_data else ""
        else:
            csv_files = []
            data_path = ""
    
    if not csv_files:
        print("No ATP CSV files found (ex: atp_matches_2023.csv)")
        return False, None
    
    # Test predictor 
    try:
        predictor = ProgressiveTennisPredictor()
    except Exception as e:
        print(f"Predictor failed: {e}")
        return False, None
    
    # Test data loading
    try:
        test_file = data_path + csv_files[0]
        df = predictor.load_and_prepare_data(test_file)

        if len(df) < 100:
            print(f"Only {len(df)} matches (might be insufficient for training)")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return False, None
    
    # Test GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
    else:
        print("No GPU available (CPU is slower)")
    
    return True, test_file


def minimal_baseline_test(csv_file=None):
    """Just one run to see if everything works"""
    print("MINIMAL BASELINE TEST")
    
    if csv_file is None:
        setup_ok, csv_file = debug_baseline_setup()
        if not setup_ok:
            return None
        
    try:
        from progressive_tennis_predictor import ProgressiveTennisPredictor
        
        predictor = ProgressiveTennisPredictor()
        df = predictor.load_and_prepare_data(csv_file)
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        
        predictor.train_progressive_models(train_df, epochs=10, batch_size=16)  # Very minimal for testing
        
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        results = {}
        
        for stage in stages:
            if stage in predictor.models:
                results[stage] = predictor.models[stage]['test_accuracy']
            else:
                results[stage] = 0.0
        
        print("\nMinimal test done")
        return results
        
    except Exception as e:
        print(f"\nMinimal test failed: {str(e)}")
        return None


if __name__ == "__main__":
    print("Baseline Transformer Multi-Run Analysis")
    print("1. Full analysis; 3.5 hours")
    print("2. Quick test; 15 min")
    print("3. Debug")
    print("4. Minimal test; 2 min")
    print("5. Custom config")
    
    choice = input("\nEnter choice (1/2/3/4/5): ").strip()
    
    if choice == "1":
        # Define data file paths
        data_files = {
            '2021': 'tennis_data/atp_matches_2021.csv',
            '2022': 'tennis_data/atp_matches_2022.csv', 
            '2023': 'tennis_data/atp_matches_2023.csv',
            '2024': 'tennis_data/atp_matches_2024.csv'
        }
        analyzer, stats = run_baseline_multirun_analysis(data_files=data_files, n_runs=10)
    
    elif choice == "2":
        year = 2022
        csv_file = f"tennis_data/atp_matches_{year}.csv"
        analyzer, stats = quick_test_single_year(year, csv_file, n_runs=2)
    
    elif choice == "3":
        debug_baseline_setup()
    
    elif choice == "4":
        minimal_baseline_test()
    
    elif choice == "5":
        year = input("Enter year (2021-2024): ").strip()
        csv_file = f"tennis_data/atp_matches_{year}.csv"
        n_runs = int(input("Enter number of runs: ").strip())
        analyzer, stats = quick_test_single_year(year, csv_file, n_runs)
    
    else:
        print("Invalid choice")
    
    if 'analyzer' in locals() and analyzer is not None:
        print("\nBaseline analysis complete")
        print("Check output directory for results")
