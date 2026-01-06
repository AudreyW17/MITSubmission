'''
Author: Audrey Wang (with assistance in debugging and refactoring from AI)
'''

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from pathlib import Path

class MultiRunAnalyzer:
    """Multi-run analysis for tennis prediction experiments"""
    
    def __init__(self, base_output_dir="multi_run_results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.all_results = []
        
    def run_single_experiment(self, predictor, df, solutions_to_test, run_id, year):
        """Run and collect results"""
        print(f"\nRun {run_id} for {year}")
        
        run_results = {
            'run_id': run_id,
            'year': year,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Test baseline
        if 'baseline' in solutions_to_test:
            baseline_results = self.test_baseline_stages(predictor, df)
            run_results.update(baseline_results)
        
        # Test Solution 2 (Enhanced Regularization)
        if 'solution_2' in solutions_to_test:
            from solution2_fixed import test_solution_2_fixed
            s2_accuracy = test_solution_2_fixed(predictor, df)
            run_results['s2_accuracy'] = s2_accuracy if s2_accuracy else 0.0
        
        # Test Solution 3 (Transfer Learning)
        if 'solution_3' in solutions_to_test:
            from solution3_fixed import test_solution_3_fixed
            s3_accuracy = test_solution_3_fixed(predictor, df)
            run_results['s3_accuracy'] = s3_accuracy if s3_accuracy else 0.0
        
        # Test Solution 4.1 (Basic Ensemble)
        if 'solution_4_1' in solutions_to_test:
            from solution4_ensemble import test_solution_4
            s4_1_accuracy, _ = test_solution_4(predictor, df, n_models=7)
            run_results['s4_1_accuracy'] = s4_1_accuracy if s4_1_accuracy else 0.0
        
        # Test Solution 4.2 (Enhanced Ensemble)
        if 'solution_4_2' in solutions_to_test:
            from solution4_enhanced import test_enhanced_ensemble
            s4_2_accuracy, _ = test_enhanced_ensemble(predictor, df)
            run_results['s4_2_accuracy'] = s4_2_accuracy if s4_2_accuracy else 0.0
        
        self.save_single_run(run_results, run_id, year)
        return run_results
    
    def test_baseline_stages(self, predictor, df):
        """Test baseline progressive prediction across all stages"""
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        stage_results = {}
        
        # If models already exist, evaluate them
        for stage in stages:
            if stage in predictor.models:
                accuracy = predictor.models[stage]['test_accuracy']
                stage_results[f'{stage}_accuracy'] = accuracy
            else:
                # Train and evaluate this stage
                try:
                    # Use the training method from your progressive predictor
                    X, y = predictor.prepare_progressive_features(df, stage)
                    if len(X) > 50:  # Minimum samples required
                        # Training for this run
                        from sklearn.model_selection import train_test_split
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.preprocessing import StandardScaler
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=None  # Different seed each run
                        )
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Quick baseline model - using LR
                        model = LogisticRegression(max_iter=1000, random_state=None)
                        model.fit(X_train_scaled, y_train)
                        accuracy = model.score(X_test_scaled, y_test)
                        
                        stage_results[f'{stage}_accuracy'] = accuracy
                    else:
                        stage_results[f'{stage}_accuracy'] = 0.0
                        
                except Exception as e:
                    print(f"Warning: {stage} failed - {str(e)}")
                    stage_results[f'{stage}_accuracy'] = 0.0
        
        return stage_results
    
    def save_single_run(self, run_results, run_id, year):
        """Save individual run results to JSON"""
        filename = self.base_output_dir / f"run_{run_id}_{year}.json"
        with open(filename, 'w') as f:
            json.dump(run_results, f, indent=2)
    
    def run_multiple_experiments(self, predictor_class, data_files, solutions_to_test, n_runs=10):
        """Run across all years and solutions"""
        print(f"STARTING MULTI-RUN ANALYSIS")
        print(f"Files: {list(data_files.keys())}")
        print(f"Solutions: {solutions_to_test}")
        print(f"Runs per year: {n_runs}")
        
        for year, csv_file in data_files.items():
            print(f"\nPROCESSING YEAR: {year}")
            
            for run_id in range(1, n_runs + 1):
                try:
                    # New predictor for each run
                    predictor = predictor_class()
                    df = predictor.load_and_prepare_data(csv_file)
                    
                    # Run single experiment
                    run_results = self.run_single_experiment(
                        predictor, df, solutions_to_test, run_id, year
                    )
                    
                    self.all_results.append(run_results)
                                        
                except Exception as e:
                    print(f"Run {run_id} failed: {str(e)}")
                    # Store failed run with zeros
                    failed_result = {
                        'run_id': run_id,
                        'year': year,
                        'timestamp': datetime.now().isoformat(),
                        'error': str(e)
                    }
                    self.all_results.append(failed_result)
    
    def calculate_statistics(self):
        """Calculate comprehensive statistics across all runs"""
        print(f"\nCALCULATING STATISTICS")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)
        metric_columns = [col for col in df.columns if col.endswith('_accuracy')]
        
        # Group by year and calculate statistics
        stats_results = []
        
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            
            print(f"\nStatistics for {year}:")
            print("-" * 30)
            
            year_stats = {'year': year}
            
            for metric in metric_columns:
                if metric in year_data.columns:
                    values = year_data[metric].dropna()
                    
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
                        
                        print(f"{metric}: {mean_val:.1%} ± {std_val:.1%} "
                              f"(n={count}, range: {min_val:.1%}-{max_val:.1%})")
                    else:
                        print(f"{metric}: No valid data")
            
            stats_results.append(year_stats)
        
        return pd.DataFrame(stats_results)
    
    def save_comprehensive_results(self, stats_df, filename_prefix="multirun"):
        """Save comprehensive results to multiple CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Raw
        raw_df = pd.DataFrame(self.all_results)
        raw_filename = self.base_output_dir / f"{filename_prefix}_raw_{timestamp}.csv"
        raw_df.to_csv(raw_filename, index=False)

        # Statistics summary
        stats_filename = self.base_output_dir / f"{filename_prefix}_stats_{timestamp}.csv"
        stats_df.to_csv(stats_filename, index=False)
        
        # Table
        pub_summary = self.create_publication_table(stats_df)
        pub_filename = self.base_output_dir / f"{filename_prefix}_publication_{timestamp}.csv"
        pub_summary.to_csv(pub_filename, index=False)

        config_info = {
            'analysis_timestamp': timestamp,
            'total_runs': len(self.all_results),
            'years_tested': list(raw_df['year'].unique()) if 'year' in raw_df.columns else [],
            'solutions_tested': [col.replace('_accuracy', '') for col in raw_df.columns if col.endswith('_accuracy')],
            'metrics_calculated': ['mean', 'std', 'min', 'max', 'count', 'ci_95']
        }
        
        config_filename = self.base_output_dir / f"{filename_prefix}_config_{timestamp}.json"
        with open(config_filename, 'w') as f:
            json.dump(config_info, f, indent=2)
        print(f"Configuration saved: {config_filename}")
        
        return {
            'raw_file': raw_filename,
            'stats_file': stats_filename, 
            'publication_file': pub_filename,
            'config_file': config_filename
        }
    
    def create_publication_table(self, stats_df):
        """Create a clean analysis ready summary table"""
        
        # Define the solutions in order
        solution_mapping = {
            'pre_match': 'S0-Pre',
            'after_set_1': 'S0-Set1', 
            'after_set_2': 'S0-Set2',
            'after_set_3': 'S0-Set3',
            'after_set_4': 'S0-Set4',
            's2': 'S2-Regularization',
            's3': 'S3-Transfer',
            's4_1': 'S4.1-Ensemble',
            's4_2': 'S4.2-Enhanced'
        }
        
        pub_data = []
        
        for _, row in stats_df.iterrows():
            year = row['year']
            
            for solution_key, solution_name in solution_mapping.items():
                mean_col = f'{solution_key}_accuracy_mean'
                std_col = f'{solution_key}_accuracy_std'
                count_col = f'{solution_key}_accuracy_count'
                ci_col = f'{solution_key}_accuracy_ci_95'
                min_col= f'{solution_key}_accuracy_min'
                max_col = f'{solution_key}_accuracy_max'
                
                if mean_col in row and pd.notna(row[mean_col]):
                    pub_data.append({
                        'Year': year,
                        'Solution': solution_name,
                        'Mean_Accuracy': row[mean_col],
                        'Std_Accuracy': row[std_col] if std_col in row else 0,
                        'Min_Accuracy': row[min_col] if min_col in row else 0,
                        'Max_Accuracy': row[max_col] if max_col in row else 0,
                        'Count': int(row[count_col]) if count_col in row else 0,
                        'CI_95': row[ci_col] if ci_col in row else 0,
                        'Formatted_Result': f"{row[mean_col]:.1%} ± {row[std_col]:.1%}" if std_col in row and pd.notna(row[std_col]) else f"{row[mean_col]:.1%}"
                    })
        
        return pd.DataFrame(pub_data)


def run_complete_multirun_analysis():
    """Main function to run analysis"""
    
    # Import your predictor class
    from progressive_tennis_predictor import ProgressiveTennisPredictor
    
    analyzer = MultiRunAnalyzer(base_output_dir="multirun_results")
    
    # Define data files
    data_files = {
        '2021': 'tennis_data/atp_matches_2021.csv',
        '2022': 'tennis_data/atp_matches_2022.csv',
        '2023': 'tennis_data/atp_matches_2023.csv', 
        '2024': 'tennis_data/atp_matches_2024.csv'
    }
     
    # Define solutions
    solutions_to_test = [
        'baseline',      # Progressive prediction
        'solution_2',    # Enhanced regularization  
        'solution_3',    # Transfer learning
        'solution_4_1',  # Basic ensemble
        'solution_4_2'   # Enhanced ensemble
    ]
    
    # Run multi-run experiments
    analyzer.run_multiple_experiments(
        predictor_class=ProgressiveTennisPredictor,
        data_files=data_files,
        solutions_to_test=solutions_to_test,
        n_runs=1
    )
    
    # Calculate statistics
    stats_df = analyzer.calculate_statistics()
    saved_files = analyzer.save_comprehensive_results(stats_df)
    
    print(f"\nMULTI-RUN ANALYSIS COMPLETE")
    for file_type, filepath in saved_files.items():
        print(f"  {file_type}: {filepath}")
    
    return analyzer, stats_df


def quick_stage_analysis_only():
    """Simplified version focusing only on progressive prediction stages"""
    from progressive_tennis_predictor import ProgressiveTennisPredictor
    
    analyzer = MultiRunAnalyzer(base_output_dir="stage_analysis_results")
    
    data_files = {
        '2021': 'tennis_data/atp_matches_2021.csv',
        '2022': 'tennis_data/atp_matches_2022.csv',
        '2023': 'tennis_data/atp_matches_2023.csv', 
        '2024': 'tennis_data/atp_matches_2024.csv'
    }
    
    # Only test baseline stages
    solutions_to_test = ['baseline']
    
    analyzer.run_multiple_experiments(
        predictor_class=ProgressiveTennisPredictor,
        data_files=data_files,
        solutions_to_test=solutions_to_test,
        n_runs=1
    )
    
    stats_df = analyzer.calculate_statistics()
    # saved_files = analyzer.save_comprehensive_results(stats_df, filename_prefix="stage_analysis")
    
    return analyzer, stats_df


if __name__ == "__main__":
    print("Multi-Run Tennis Prediction Analysis")
    print("1. Complete analysis (all solutions)")
    print("2. Stage analysis only (faster)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        analyzer, stats = run_complete_multirun_analysis()
    else:
        analyzer, stats = quick_stage_analysis_only()
    
    print("\nAnalysis complete")
