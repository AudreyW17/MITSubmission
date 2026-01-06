"""
Author: Audrey Wang (with assistance in debugging and refactoring from AI)

- Usage
# Train the system
predictor = main()

# Predict at different stages
prediction = predictor.predict_progressive_match(
    'Novak Djokovic', 'Rafael Nadal', 'Clay', 5, 
    sets_so_far=[(6,4), (4,6)]  # After 2 sets
)

print(f"Winner: {prediction['predicted_winner']}")
print(f"Confidence: {prediction['confidence']:.3f}")
print(f"Stage: {prediction['stage']}")


"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import time
import warnings
warnings.filterwarnings('ignore')

class ProgressiveTennisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class EnhancedTennisTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, num_classes=2):
        super(EnhancedTennisTransformer, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.15,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input and add positional encoding
        x = self.input_projection(x).unsqueeze(1)
        x = x + self.pos_encoding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Classification
        x = x.squeeze(1)
        return self.classifier(x)

class ProgressiveTennisPredictor:
    def __init__(self):
        self.player_encoder = LabelEncoder()
        self.surface_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}  # Store different models for different stages
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.player_stats = {}
        
    def parse_score(self, score_str):
        """Parse ATP score string into individual sets"""
        if pd.isna(score_str) or not score_str:
            return []
        
        # Handle retirements and walkovers
        if 'RET' in score_str or 'W/O' in score_str or 'DEF' in score_str:
            # Extract completed sets before retirement
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
                
            # Extract games (ignore tiebreak scores in parentheses)
            clean_set = re.sub(r'\([^)]*\)', '', set_score)
            
            if '-' in clean_set:
                try:
                    winner_games, loser_games = map(int, clean_set.split('-'))
                    parsed_sets.append((winner_games, loser_games))
                except ValueError:
                    continue
        
        return parsed_sets
    
    def calculate_set_features(self, sets_so_far, is_winner_perspective=True):
        """Calculate momentum and match state features from completed sets"""
        if not sets_so_far:
            return [0.0] * 10  # Default features
        
        # Flip perspective if needed
        if not is_winner_perspective:
            sets_so_far = [(loser, winner) for winner, loser in sets_so_far]
        
        features = []
        
        # 1. Sets won
        sets_won = sum(1 for w,l in sets_so_far if w > l) # From winner's perspective
        features.append(sets_won)
        
        # 2. Total games won
        total_games_won = sum(winner for winner, _ in sets_so_far)
        total_games_lost = sum(loser for _, loser in sets_so_far)
        features.append(total_games_won)
        features.append(total_games_lost)
        
        # 3. Game win percentage
        total_games = total_games_won + total_games_lost
        games_win_pct = total_games_won / max(total_games, 1)
        features.append(games_win_pct)
        
        # 4. Close sets (decided by <= 2 games)
        close_sets_won = sum(1 for w, l in sets_so_far if (w > l) and (abs(w - l) <= 2))
        features.append(close_sets_won)
        
        # 5. Dominant sets (won by >= 4 games)
        dominant_sets = sum(1 for w, l in sets_so_far if w - l >= 4)
        features.append(dominant_sets)
        
        # 6. Momentum (last set performance)
        if sets_so_far:
            last_set_winner, last_set_loser = sets_so_far[-1]
            last_set_dominance = (last_set_winner - last_set_loser) / max(last_set_winner + last_set_loser, 1)
            features.append(last_set_dominance)
        else:
            features.append(0.0)
        
        # 7. Average games per set
        avg_games_won = total_games_won / len(sets_so_far)
        avg_games_lost = total_games_lost / len(sets_so_far)
        features.append(avg_games_won)
        features.append(avg_games_lost)
        
        # 8. Consistency (standard deviation of games won per set)
        if len(sets_so_far) > 1:
            games_per_set = [w for w, _ in sets_so_far]
            consistency = np.std(games_per_set)
            features.append(consistency)
        else:
            features.append(0.0)
        
        return features
    
    def load_and_prepare_data(self, csv_file):
        """Load and prepare ATP data with enhanced features"""

        df = pd.read_csv(csv_file)
        
        # Filter for complete matches with rankings
        df = df.dropna(subset=['winner_name', 'loser_name', 'surface', 'score', 'winner_rank', 'loser_rank'])
        
        print(f"Loaded {len(df)} complete matches")
        
        # Calculate player statistics
        self.calculate_player_stats(df)
        
        return df
    
    def calculate_player_stats(self, df):
        """Calculate comprehensive player statistics"""
        
        # Initialize stats for all players
        all_players = set(df['winner_name'].unique()) | set(df['loser_name'].unique())
        for player in all_players:
            self.player_stats[player] = {
                'matches': 0, 'wins': 0, 'total_ranking': 0, 'ranking_count': 0,
                'clay_wins': 0, 'clay_matches': 0,
                'grass_wins': 0, 'grass_matches': 0,
                'hard_wins': 0, 'hard_matches': 0,
                'best_of_3_wins': 0, 'best_of_3_matches': 0,
                'best_of_5_wins': 0, 'best_of_5_matches': 0,
                'sets_won': 0, 'sets_total': 0,
                'games_won': 0, 'games_total': 0
            }
        
        # Calculate statistics from all matches
        for _, row in df.iterrows():
            winner = row['winner_name']
            loser = row['loser_name']
            surface = row['surface'].lower()
            best_of = row['best_of']
            sets = self.parse_score(row['score'])
            
            # Basic match stats
            self.player_stats[winner]['matches'] += 1
            self.player_stats[winner]['wins'] += 1
            self.player_stats[loser]['matches'] += 1
            
            # Ranking tracking
            if pd.notna(row['winner_rank']):
                self.player_stats[winner]['total_ranking'] += row['winner_rank']
                self.player_stats[winner]['ranking_count'] += 1
            if pd.notna(row['loser_rank']):
                self.player_stats[loser]['total_ranking'] += row['loser_rank']
                self.player_stats[loser]['ranking_count'] += 1
            
            # Surface-specific stats
            self.player_stats[winner][f'{surface}_matches'] += 1
            self.player_stats[winner][f'{surface}_wins'] += 1
            self.player_stats[loser][f'{surface}_matches'] += 1
            
            # Match format stats
            format_key = f'best_of_{best_of}'
            self.player_stats[winner][f'{format_key}_matches'] += 1
            self.player_stats[winner][f'{format_key}_wins'] += 1
            self.player_stats[loser][f'{format_key}_matches'] += 1
            
            # Set and game statistics
            if sets:
                winner_sets = len(sets)
                loser_sets = 0  # Since winner won the match
                
                winner_games = sum(w for w, _ in sets)
                loser_games = sum(l for _, l in sets)
                
                self.player_stats[winner]['sets_won'] += winner_sets
                self.player_stats[winner]['sets_total'] += winner_sets + loser_sets
                self.player_stats[winner]['games_won'] += winner_games
                self.player_stats[winner]['games_total'] += winner_games + loser_games
                
                self.player_stats[loser]['sets_won'] += loser_sets
                self.player_stats[loser]['sets_total'] += winner_sets + loser_sets
                self.player_stats[loser]['games_won'] += loser_games
                self.player_stats[loser]['games_total'] += winner_games + loser_games
    
    def get_enhanced_player_features(self, player, surface, best_of):
        """Get comprehensive player features"""
        if player not in self.player_stats:
            return [0.0] * 15
        
        stats = self.player_stats[player]
        features = []
        
        # 1. Overall performance
        overall_winrate = stats['wins'] / max(stats['matches'], 1)
        features.append(overall_winrate)
        
        # 2. Surface-specific performance
        surface_matches = stats[f'{surface.lower()}_matches']
        surface_winrate = stats[f'{surface.lower()}_wins'] / max(surface_matches, 1)
        features.append(surface_winrate)
        
        # 3. Format-specific performance
        format_matches = stats[f'best_of_{best_of}_matches']
        format_winrate = stats[f'best_of_{best_of}_wins'] / max(format_matches, 1)
        features.append(format_winrate)
        
        # 4. Experience metrics
        total_experience = np.log(stats['matches'] + 1)
        surface_experience = np.log(surface_matches + 1)
        format_experience = np.log(format_matches + 1)
        features.extend([total_experience, surface_experience, format_experience])
        
        # 5. Average ranking
        avg_ranking = stats['total_ranking'] / max(stats['ranking_count'], 1) if stats['ranking_count'] > 0 else 100
        ranking_quality = max(0, (200 - avg_ranking) / 200)  # Normalize ranking (lower is better)
        features.append(ranking_quality)
        
        # 6. Set performance
        set_win_rate = stats['sets_won'] / max(stats['sets_total'], 1)
        features.append(set_win_rate)
        
        # 7. Game performance
        game_win_rate = stats['games_won'] / max(stats['games_total'], 1)
        features.append(game_win_rate)
        
        # 8. Consistency metrics
        match_count = stats['matches']
        surface_specialization = surface_matches / max(match_count, 1)
        features.extend([match_count, surface_specialization])
        
        # 9. Recent form proxy (using overall performance)
        recent_form = overall_winrate
        features.append(recent_form)
        
        # 10. Additional metrics
        format_specialization = format_matches / max(match_count, 1)
        versatility = min(stats['clay_matches'], stats['grass_matches'], stats['hard_matches']) / max(match_count / 3, 1)
        features.extend([format_specialization, versatility])
        
        return features
    
    def prepare_progressive_features(self, df, stage='pre_match'):
        """Prepare features for different prediction stages"""
        print(f"Preparing features for stage: {stage}")
        
        features = []
        labels = []
        
        # Fit encoders
        all_players = list(set(df['winner_name'].unique()) | set(df['loser_name'].unique()))
        all_surfaces = df['surface'].unique()
        
        self.player_encoder.fit(all_players)
        self.surface_encoder.fit(all_surfaces)
        
        for idx, row in df.iterrows():
            winner = row['winner_name']
            loser = row['loser_name']
            surface = row['surface']
            best_of = row['best_of']
            sets = self.parse_score(row['score'])
            
            if not sets:  # Skip matches without proper scores
                continue
            
            # Determine how many sets to use based on stage
            if stage == 'pre_match':
                sets_to_use = []
            elif stage == 'after_set_1':
                sets_to_use = sets[:1] if len(sets) >= 1 else []
            elif stage == 'after_set_2':
                sets_to_use = sets[:2] if len(sets) >= 2 else []
            elif stage == 'after_set_3':
                sets_to_use = sets[:3] if len(sets) >= 3 else []
            elif stage == 'after_set_4':
                sets_to_use = sets[:4] if len(sets) >= 4 else []
            else:
                continue
            
            # Skip if we don't have enough sets for this stage
            if stage != 'pre_match' and len(sets) < int(stage.split('_')[-1]):
                continue
            
            # Get player features
            winner_features = self.get_enhanced_player_features(winner, surface, best_of)
            loser_features = self.get_enhanced_player_features(loser, surface, best_of)
            
            # Get match context features
            surface_encoded = self.surface_encoder.transform([surface])[0]
            best_of_encoded = 1 if best_of == 5 else 0
            
            # Get set-based features
            winner_set_features = self.calculate_set_features(sets_to_use, True)
            loser_set_features = self.calculate_set_features(sets_to_use, False)
            
            # Create feature vector
            feature_vector = (winner_features + loser_features + 
                            [surface_encoded, best_of_encoded] + 
                            winner_set_features + loser_set_features)
            
            features.append(feature_vector)
            labels.append(1)  # Winner wins
            
            # Create reverse match
            reverse_feature_vector = (loser_features + winner_features + 
                                    [surface_encoded, best_of_encoded] + 
                                    loser_set_features + winner_set_features)
            features.append(reverse_feature_vector)
            labels.append(0)  # Loser loses
        
        return np.array(features), np.array(labels)
    
    def train_progressive_models(self, df, epochs=150, batch_size=64):
        """Train separate models for different match stages"""
        
        stages = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
        
        for stage in stages:
            print(f"\n{'='*50}")
            print(f"TRAINING MODEL FOR: {stage.upper()}")
            print(f"{'='*50}")
            
            # Prepare data for this stage
            X, y = self.prepare_progressive_features(df, stage)
            
            if len(X) < 100:  # Skip if not enough data
                print(f"Insufficient data for {stage}: {len(X)} samples")
                continue
            
            print(f"Dataset size: {len(X)} samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Create datasets
            train_dataset = ProgressiveTennisDataset(X_train_scaled, y_train)
            val_dataset = ProgressiveTennisDataset(X_val_scaled, y_val)
            test_dataset = ProgressiveTennisDataset(X_test_scaled, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Initialize model
            input_dim = X_train.shape[1]
            model = EnhancedTennisTransformer(input_dim=input_dim).to(self.device)
            print(f"input_dim=,{input_dim}")
          
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
            
            best_val_acc = 0
            patience_counter = 0
            
            # Training loop
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()
                
                val_acc = val_correct / val_total
                scheduler.step(val_loss)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'scaler': scaler,
                        'input_dim': input_dim
                    }, f'best_tennis_model_{stage}.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 20 == 0:
                    print(f'Epoch {epoch}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, '
                          f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
                
                if patience_counter >= 25:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Load best model and test
            checkpoint = torch.load(f'best_tennis_model_{stage}.pth', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Test evaluation
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    test_total += batch_labels.size(0)
                    test_correct += (predicted == batch_labels).sum().item()
            
            test_accuracy = test_correct / test_total
            print(f'\n{stage.upper()} - Final Results:')
            print(f'Best Validation Accuracy: {best_val_acc:.4f}')
            print(f'Test Accuracy: {test_accuracy:.4f}')
            
            # Store model info
            self.models[stage] = {
                'model': model,
                'scaler': checkpoint['scaler'],
                'input_dim': checkpoint['input_dim'],
                'test_accuracy': test_accuracy
            }    
    
    def predict_progressive_match(self, player1, player2, surface, best_of=3, sets_so_far=None):
        """Predict match outcome at different stages"""
        if sets_so_far is None:
            sets_so_far = []
        
        # Determine stage
        num_sets = len(sets_so_far)
        if num_sets == 0:
            stage = 'pre_match'
        elif num_sets <= 4:
            stage = f'after_set_{num_sets}'
        else:
            stage = 'after_set_4'  # Use latest available model
        
        if stage not in self.models:
            raise ValueError(f"Model for stage '{stage}' not trained!")
        
        model_info = self.models[stage]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Prepare features
        player1_features = self.get_enhanced_player_features(player1, surface, best_of)
        player2_features = self.get_enhanced_player_features(player2, surface, best_of)
        
        surface_encoded = self.surface_encoder.transform([surface])[0]
        best_of_encoded = 1 if best_of == 5 else 0
        
        player1_set_features = self.calculate_set_features(sets_so_far, True)
        player2_set_features = self.calculate_set_features(sets_so_far, False)
        
        feature_vector = np.array([player1_features + player2_features + 
                                 [surface_encoded, best_of_encoded] + 
                                 player1_set_features + player2_set_features])
        
        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(feature_vector_scaled).to(self.device)
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            player1_win_prob = probabilities[0][1].item()
            player2_win_prob = probabilities[0][0].item()
        
        return {
            'stage': stage,
            'sets_completed': num_sets,
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'best_of': best_of,
            'current_sets': sets_so_far,
            'player1_win_probability': player1_win_prob,
            'player2_win_probability': player2_win_prob,
            'predicted_winner': player1 if player1_win_prob > player2_win_prob else player2,
            'confidence': max(player1_win_prob, player2_win_prob),
            'model_test_accuracy': model_info['test_accuracy']
        }

def demonstrate_progressive_prediction(predictor):
    """Demonstrate how prediction accuracy improves with more set information"""
    
    print("\nPROGRESSIVE PREDICTION DEMONSTRATION")
    
    # Example match scenarios
    scenarios = [
        {
            'player1': 'Novak Djokovic',
            'player2': 'Rafael Nadal', 
            'surface': 'Clay',
            'best_of': 5,
            'actual_sets': [(4, 6), (6, 2), (6, 3), (6, 3)]  # Hypothetical
        },
        {
            'player1': 'Carlos Alcaraz',
            'player2': 'Daniil Medvedev',
            'surface': 'Hard', 
            'best_of': 3,
            'actual_sets': [(7, 6), (3, 6), (6, 4)]  # Hypothetical
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"MATCH: {scenario['player1']} vs {scenario['player2']}")
        print(f"Surface: {scenario['surface']}, Best of {scenario['best_of']}")
        print(f"{'='*60}")
        
        # Progressive predictions
        sets_so_far = []
        
        for i in range(len(scenario['actual_sets']) + 1):
            if i == 0:
                stage_name = "PRE-MATCH"
                sets_so_far = []
            else:
                stage_name = f"AFTER SET {i}"
                sets_so_far = scenario['actual_sets'][:i]
            
            try:
                prediction = predictor.predict_progressive_match(
                    scenario['player1'], 
                    scenario['player2'], 
                    scenario['surface'],
                    scenario['best_of'],
                    sets_so_far
                )
                
                print(f"\n{stage_name}:")
                if sets_so_far:
                    sets_display = " | ".join([f"{w}-{l}" for w, l in sets_so_far])
                    print(f"  Current score: {sets_display}")
                
                print(f"  Predicted winner: {prediction['predicted_winner']}")
                print(f"  Confidence: {prediction['confidence']:.3f}")
                print(f"  {prediction['player1']}: {prediction['player1_win_probability']:.3f}")
                print(f"  {prediction['player2']}: {prediction['player2_win_probability']:.3f}")
                print(f"  Model accuracy: {prediction['model_test_accuracy']:.3f}")
                
            except Exception as e:
                print(f"  {stage_name}: Model not available - {str(e)}")

def analyze_prediction_accuracy_improvement(predictor, test_df):
    """Analyze how prediction accuracy improves across different stages"""
    
    print("\nPREDICTION ACCURACY ANALYSIS ACROSS MATCH STAGES")
    
    # Get accuracy for each stage
    stage_accuracies = {}
    for stage, model_info in predictor.models.items():
        stage_accuracies[stage] = model_info['test_accuracy']
    
    # Sort by logical progression
    stage_order = ['pre_match', 'after_set_1', 'after_set_2', 'after_set_3', 'after_set_4']
    
    print("\nPrediction Accuracy by Stage:")
    print("-" * 40)
    
    baseline_accuracy = None
    for stage in stage_order:
        if stage in stage_accuracies:
            accuracy = stage_accuracies[stage]
            if baseline_accuracy is None:
                baseline_accuracy = accuracy
                improvement = 0.0
            else:
                improvement = accuracy - baseline_accuracy
            
            print(f"{stage.replace('_', ' ').title():<15}: {accuracy:.4f} "
                  f"(+{improvement:+.4f} from baseline)")
    
    # Create accuracy improvement visualization data
    return stage_accuracies

def test_real_match_prediction(predictor, test_df, num_matches=5):
    """Test progressive prediction on real matches from the dataset"""
    
    print("\nREAL MATCH PROGRESSIVE PREDICTION TEST")
    
    # Sample some interesting matches
    sample_matches = test_df.sample(n=num_matches, random_state=42)
    
    for idx, match in sample_matches.iterrows():
        winner = match['winner_name']
        loser = match['loser_name']
        surface = match['surface']
        best_of = match['best_of']
        score = match['score']
        
        print(f"\n{'='*60}")
        print(f"REAL MATCH: {winner} defeated {loser}")
        print(f"Surface: {surface}, Best of {best_of}")
        print(f"Actual score: {score}")
        print(f"{'='*60}")
        
        # Parse actual sets
        actual_sets = predictor.parse_score(score)
        if not actual_sets:
            print("Could not parse score - skipping")
            continue
        
        # Progressive predictions
        print(f"\nProgressive Predictions:")
        print("-" * 30)
        
        sets_so_far = []
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(actual_sets) + 1):
            stage_name = "Pre-match" if i == 0 else f"After set {i}"
            
            try:
                prediction = predictor.predict_progressive_match(
                    winner, loser, surface, best_of, sets_so_far
                )
                
                predicted_winner = prediction['predicted_winner']
                confidence = prediction['confidence']
                
                # Check if prediction is correct
                is_correct = predicted_winner == winner
                correct_predictions += is_correct
                total_predictions += 1
                
                status = "Correct" if is_correct else "Wrong"
                
                print(f"{stage_name:<12}: {predicted_winner:<20} "
                      f"({confidence:.3f}) {status}")
                
                # Add next set for next iteration
                if i < len(actual_sets):
                    sets_so_far.append(actual_sets[i])
                    
            except Exception as e:
                print(f"{stage_name:<12}: Model not available")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\nMatch prediction accuracy: {accuracy:.3f} "
              f"({correct_predictions}/{total_predictions})")

def main():
    """Main execution function"""
    
    start_time = time.time()

    # Initialize predictor
    predictor = ProgressiveTennisPredictor()
    
    # Load data
    print("Starting Progressive Tennis Match Prediction System")
    
    df = predictor.load_and_prepare_data('tennis_data/atp_matches_2024.csv')
    
    # Analyze match lengths in your data:
    match_lengths = df.apply(lambda row: len(predictor.parse_score(row['score'])), axis=1)
    print(f"3-set matches: {sum(match_lengths == 3)}")
    print(f"4-set matches: {sum(match_lengths == 4)}")  
    print(f"5-set matches: {sum(match_lengths == 5)}")
    
    # Split data for training and testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(train_df)} matches")
    print(f"Test set: {len(test_df)} matches")
    
    # Train progressive models
    predictor.train_progressive_models(train_df)
    
    # Analyze accuracy improvements
    stage_accuracies = analyze_prediction_accuracy_improvement(predictor, test_df)
    
    # Comeback analysis
    #from comeback_analysis import run_comeback_analysis        
    #results = run_comeback_analysis(predictor, test_df)

    if 0:
        # Demonstrate progressive prediction
        demonstrate_progressive_prediction(predictor)
        
        # Test on real matches
        test_real_match_prediction(predictor, test_df)
        
        print("\nSUMMARY OF RESULTS")
        
        print("\nTrained Models:")
        for stage, model_info in predictor.models.items():
            print(f"  {stage.replace('_', ' ').title()}: {model_info['test_accuracy']:.4f} accuracy")
        
        print(f"\nTotal stages trained: {len(predictor.models)}")
        print("Progressive prediction system ready")
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
    
    return predictor

if __name__ == "__main__":
    predictor = main()