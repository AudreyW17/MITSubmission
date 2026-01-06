'''
Author: Audrey Wang (with assistance in debugging and refactoring from AI)
'''

import pandas as pd
import numpy as np
import re
from datetime import datetime

class ComebackAnalyzer:
    """Analyze tennis comeback predictions: 1-2 sets → 3-2 final score"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.comeback_matches = []
        self.correctly_predicted = []
        
    def parse_score_detailed(self, score_str):
        if pd.isna(score_str) or not score_str:
            return []
        
        # Handle retirements and walkovers
        if 'RET' in score_str or 'W/O' in score_str or 'DEF' in score_str:
            if 'W/O' in score_str or 'DEF' in score_str:
                return []
            else:
                # Extract completed sets before retirement
                score_clean = re.sub(r'\s+(RET|W/O|DEF).*', '', score_str)
                if not score_clean.strip():
                    return []
        else:
            score_clean = score_str
        
        # Split into individual sets
        sets = score_clean.strip().split()
        parsed_sets = []
        
        for set_score in sets:
            if not set_score or set_score in ['RET', 'W/O', 'DEF']:
                continue
            
            # Remove tiebreak scores in parentheses for parsing
            clean_set = re.sub(r'\([^)]*\)', '', set_score)
            
            if '-' in clean_set:
                try:
                    winner_games, loser_games = map(int, clean_set.split('-'))
                    parsed_sets.append((winner_games, loser_games))
                except ValueError:
                    continue
        
        return parsed_sets
    
    def is_comeback_match(self, sets):
        if len(sets) != 5:  # Must be exactly 5 sets
            return False
        
        # Calculate set wins from winner's perspective
        set_wins = []
        for winner_games, loser_games in sets:
            if winner_games > loser_games:
                set_wins.append(1)  # Winner won this set
            else:
                set_wins.append(0)  # Winner lost this set
        
        # Check for 1-2 → 3-2 pattern
        # After 3 sets: winner has 1 win, loser has 2 wins
        # Final result: winner has 3 wins, loser has 2 wins
        
        sets_after_3 = sum(set_wins[:3]) # Winner's sets after 3 sets
        final_sets = sum(set_wins) # Winner's final sets
        
        # Comeback condition: 1-2 after 3 sets, 3-2 final
        return sets_after_3 == 1 and final_sets == 3
    
    def analyze_comeback_predictions(self, test_df, save_results=True):
        """
        Analyze comeback prediction performance
        """
        print("TENNIS COMEBACK PREDICTION ANALYSIS")
        print("Analyzing matches with 1-2 → 3-2 comeback pattern...")
        
        comeback_data = []
        prediction_results = []
        
        total_matches = 0
        total_comebacks = 0
        correct_predictions = 0
        
        for idx, row in test_df.iterrows():
            total_matches += 1
            
            # Parse match details
            winner = row['winner_name']
            loser = row['loser_name']
            surface = row['surface']
            score = row['score']
            tournament = row.get('tourney_name', 'Unknown')
            date = row.get('tourney_date', 'Unknown')
            
            # Parse sets
            sets = self.parse_score_detailed(score)
            
            if not sets:
                continue
            
            # Check if this is a comeback match
            if self.is_comeback_match(sets):
                total_comebacks += 1
                
                print(f"\nComeback Match #{total_comebacks}")
                print(f"{winner} defeated {loser}")
                print(f"Score: {score}")
                print(f"Surface: {surface}")
                print(f"Tournament: {tournament}")
                
                # Store comeback match data
                comeback_match_data = {
                    'match_id': idx,
                    'winner': winner,
                    'loser': loser,
                    'surface': surface,
                    'score': score,
                    'tournament': tournament,
                    'date': date,
                    'set_1': f"{sets[0][0]}-{sets[0][1]}",
                    'set_2': f"{sets[1][0]}-{sets[1][1]}",
                    'set_3': f"{sets[2][0]}-{sets[2][1]}",
                    'set_4': f"{sets[3][0]}-{sets[3][1]}",
                    'set_5': f"{sets[4][0]}-{sets[4][1]}",
                    'sets_after_3': '1-2 (trailing)',
                    'final_result': '3-2 (comeback win)'
                }
                comeback_data.append(comeback_match_data)
                
                # Test predictions at different stages
                predictions = self.test_progressive_predictions(
                    winner, loser, surface, sets, comeback_match_data
                )
                
                # Check if After-Set-3 prediction was correct
                after_set_3_correct = predictions.get('after_set_3_correct', False)
                if after_set_3_correct:
                    correct_predictions += 1
                    prediction_results.append({**comeback_match_data, **predictions})
                    print(f"Correctly predicted comeback")
                else:
                    print(f"Failed to predict comeback")
                
                # Add prediction details to comeback data
                comeback_data[-1].update(predictions)
        
        # Summary statistics
        print(f"\nCOMEBACK ANALYSIS SUMMARY")
        print(f"Total matches analyzed: {total_matches}")
        print(f"Total comeback matches (1-2 → 3-2): {total_comebacks}")
        print(f"Correctly predicted comebacks: {correct_predictions}")
        
        if total_comebacks > 0:
            accuracy = correct_predictions / total_comebacks * 100
            print(f"Comeback prediction accuracy: {accuracy:.1f}%")
            
            # Additional insights
            print(f"\nComeback match frequency: {total_comebacks/total_matches*100:.1f}% of all matches")
            print(f"Comeback prediction challenge: Predicting unlikely outcomes")
        else:
            print("No comeback matches found in test set")
            accuracy = 0
        
        # Save results to CSV files
        if save_results and comeback_data:
            self.save_comeback_results(comeback_data, prediction_results)
        
        return {
            'total_matches': total_matches,
            'total_comebacks': total_comebacks,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'comeback_data': comeback_data,
            'prediction_results': prediction_results
        }
    
    def test_progressive_predictions(self, winner, loser, surface, sets, match_data):
        """
        Test model predictions at different stages of the comeback match
        """
        predictions = {}
        
        try:
            # Test different prediction stages
            stages_to_test = [
                ('pre_match', []),
                ('after_set_1', sets[:1]),
                ('after_set_2', sets[:2]), 
                ('after_set_3', sets[:3]),  # Key stage: 1-2 deficit
                ('after_set_4', sets[:4])
            ]
            
            for stage_name, sets_so_far in stages_to_test:
                try:
                    # Make prediction using your trained model
                    prediction = self.predictor.predict_progressive_match(
                        winner, loser, surface, best_of=5, sets_so_far=sets_so_far
                    )
                    
                    predicted_winner = prediction['predicted_winner']
                    confidence = prediction['confidence']
                    winner_prob = prediction['player1_win_probability']
                    
                    # Check if prediction is correct (actual winner = predicted winner)
                    is_correct = predicted_winner == winner
                    
                    predictions[f'{stage_name}_predicted_winner'] = predicted_winner
                    predictions[f'{stage_name}_confidence'] = f"{confidence:.3f}"
                    predictions[f'{stage_name}_winner_prob'] = f"{winner_prob:.3f}"
                    predictions[f'{stage_name}_correct'] = is_correct
                    
                    print(f"{stage_name}: {predicted_winner} ({confidence:.3f}) {'Correct' if is_correct else 'Wrong'}")
                    
                except Exception as e:
                    print(f"{stage_name}: Prediction failed - {str(e)}")
                    predictions[f'{stage_name}_predicted_winner'] = 'Error'
                    predictions[f'{stage_name}_confidence'] = '0.000'
                    predictions[f'{stage_name}_correct'] = False
            
        except Exception as e:
            print(f"Error in prediction testing: {str(e)}")
        
        return predictions
    
    def save_comeback_results(self, comeback_data, prediction_results):
        """Save results to CSV files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all comeback matches
        comeback_df = pd.DataFrame(comeback_data)
        comeback_filename = f"comeback_matches_{timestamp}.csv"
        comeback_df.to_csv(comeback_filename, index=False)
        print(f"\nSaved all comeback matches to: {comeback_filename}")
        
        # Save correctly predicted comebacks
        if prediction_results:
            correct_df = pd.DataFrame(prediction_results)
            correct_filename = f"correctly_predicted_comebacks_{timestamp}.csv"
            correct_df.to_csv(correct_filename, index=False)
            print(f"Saved correctly predicted comebacks to: {correct_filename}")
        else:
            print("No correctly predicted comebacks to save")
    
    def analyze_comeback_patterns(self, comeback_data):
        """
        Analyze patterns in comeback matches and predictions
        """
        if not comeback_data:
            print("No comeback data to analyze")
            return
        
        print(f"\nCOMEBACK PATTERN ANALYSIS")
        
        df = pd.DataFrame(comeback_data)
        
        # Surface analysis
        if 'surface' in df.columns:
            surface_counts = df['surface'].value_counts()
            print(f"\nComeback matches by surface:")
            for surface, count in surface_counts.items():
                percentage = count / len(df) * 100
                print(f"  {surface}: {count} matches ({percentage:.1f}%)")
        
        # Prediction accuracy by stage
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        print(f"\nPrediction accuracy by stage:")
        
        for stage in stages:
            correct_col = f'{stage}_correct'
            if correct_col in df.columns:
                correct_count = df[correct_col].sum()
                total_predictions = df[correct_col].notna().sum()
                if total_predictions > 0:
                    accuracy = correct_count / total_predictions * 100
                    print(f"  {stage}: {correct_count}/{total_predictions} ({accuracy:.1f}%)")
        
        # Confidence analysis for After-Set-3 (key moment)
        if 'after_set_3_confidence' in df.columns:
            confidences = df['after_set_3_confidence'].apply(lambda x: float(x) if x != 'Error' else None).dropna()
            if len(confidences) > 0:
                avg_confidence = confidences.mean()
                print(f"\nAfter-Set-3 prediction confidence:")
                print(f"Average confidence: {avg_confidence:.3f}")
                print(f"High confidence (>0.8): {(confidences > 0.8).sum()} matches")
                print(f"Low confidence (<0.6): {(confidences < 0.6).sum()} matches")

def run_comeback_analysis(predictor, test_df):
    """
    Main function to run comeback analysis
    """
    print("Starting Tennis Comeback Prediction Analysis...")
    
    # Initialize analyzer
    analyzer = ComebackAnalyzer(predictor)
    
    # Run analysis
    results = analyzer.analyze_comeback_predictions(test_df, save_results=True)
    
    # Analyze patterns
    if results['comeback_data']:
        analyzer.analyze_comeback_patterns(results['comeback_data'])
    
    # Final summary
    print(f"\nFINAL RESULTS:")
    print(f"Found {results['total_comebacks']} comeback matches")
    print(f"Correctly predicted {results['correct_predictions']} comebacks")
    print(f"Comeback prediction accuracy: {results['accuracy']:.1f}%")
    
    return results

# Usage example
if __name__ == "__main__":
    print("Comeback Analysis Module Ready")
    print("Usage:")
    print("results = run_comeback_analysis(your_trained_predictor, test_df)")
    print("\nProcess:")
    print("1. Find all 1-2 → 3-2 comeback matches")
    print("2. Test predictions at each stage")
    print("3. Save results to CSV files")
    print("4. Analyze comeback patterns")
