'''
Author: Audrey Wang (with assistance in debugging and refactoring from AI)
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import itertools
from collections import defaultdict
import time

class EnsembleTennisTransformer(nn.Module):
    """Individual transformer for ensemble with configurable architecture"""
    
    def __init__(self, input_dim, d_model=96, nhead=6, num_layers=2, dropout_rate=0.15, 
                 activation='relu', use_layer_norm=True):
        super(EnsembleTennisTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Configurable transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Configurable classifier
        if use_layer_norm:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model // 2, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.8),
                nn.Linear(d_model // 4, 2)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.BatchNorm1d(d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model // 2, 2)
            )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x).unsqueeze(1)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)

class TennisEnsemble:
    """Advanced ensemble system for tennis prediction"""
    
    def __init__(self, enhanced_predictor):
        self.enhanced_predictor = enhanced_predictor
        self.device = enhanced_predictor.device
        self.models = []
        self.scalers = []
        self.model_configs = []
        self.model_weights = []
        self.training_accuracies = []
        
    def create_diverse_model_configs(self, n_models=7):
        """Create diverse model configurations for ensemble"""
        
        configs = []
        
        # Strategy 1: Different architectures
        arch_configs = [
            {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'name': 'Small_Deep'},
            {'d_model': 96, 'nhead': 6, 'num_layers': 2, 'name': 'Medium_Balanced'},  
            {'d_model': 128, 'nhead': 8, 'num_layers': 1, 'name': 'Large_Shallow'},
            {'d_model': 80, 'nhead': 4, 'num_layers': 3, 'name': 'Medium_Deep'},
            {'d_model': 112, 'nhead': 7, 'num_layers': 2, 'name': 'Large_Balanced'}
        ]
        
        # Strategy 2: Different regularization approaches
        reg_configs = [
            {'dropout_rate': 0.1, 'use_layer_norm': False, 'name': 'Low_Dropout_BN'},
            {'dropout_rate': 0.25, 'use_layer_norm': True, 'name': 'High_Dropout_LN'},
            {'dropout_rate': 0.2, 'use_layer_norm': True, 'name': 'Med_Dropout_LN'}
        ]
        
        # Strategy 3: Different training approaches
        train_configs = [
            {'lr': 0.001, 'weight_decay': 0.01, 'batch_size': 32, 'name': 'Standard'},
            {'lr': 0.0005, 'weight_decay': 0.02, 'batch_size': 16, 'name': 'Conservative'},
            {'lr': 0.0015, 'weight_decay': 0.005, 'batch_size': 48, 'name': 'Aggressive'}
        ]
        
        # Combine strategies
        config_combinations = []
        for arch, reg, train in itertools.product(arch_configs[:3], reg_configs[:2], train_configs[:2]):
            combined_config = {**arch, **reg, **train}
            combined_config['name'] = f"{arch['name']}_{reg['name']}_{train['name']}"
            config_combinations.append(combined_config)
        
        # Select diverse subset
        selected_configs = config_combinations[:n_models]
        
        # Add some specific high-performing configs
        if len(selected_configs) < n_models:
            # Add regularization-inspired config
            selected_configs.append({
                'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout_rate': 0.3,
                'use_layer_norm': False, 'lr': 0.0005, 'weight_decay': 0.01, 
                'batch_size': 16, 'name': 'Regularized_Like_Solution2'
            })
            
            # Add transfer-learning-inspired config  
            selected_configs.append({
                'd_model': 96, 'nhead': 6, 'num_layers': 2, 'dropout_rate': 0.15,
                'use_layer_norm': True, 'lr': 0.0008, 'weight_decay': 0.01,
                'batch_size': 24, 'name': 'Transfer_Like_Solution3'
            })
        
        return selected_configs[:n_models]
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val, config, model_id):
        """Train a single model in the ensemble"""
        
        print(f"  Training Model {model_id}: {config['name']}")
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create datasets
        class TennisDataset(Dataset):
            def __init__(self, features, labels):
                self.features = torch.FloatTensor(features)
                self.labels = torch.LongTensor(labels)
            
            def __len__(self):
                return len(self.features)
            
            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]
        
        train_dataset = TennisDataset(X_train_scaled, y_train)
        val_dataset = TennisDataset(X_val_scaled, y_val)
        
        batch_size = config.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model
        model = EnsembleTennisTransformer(
            input_dim=X_train.shape[1],
            d_model=config.get('d_model', 96),
            nhead=config.get('nhead', 6),
            num_layers=config.get('num_layers', 2),
            dropout_rate=config.get('dropout_rate', 0.15),
            use_layer_norm=config.get('use_layer_norm', True)
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5, min_lr=1e-6
        )
        
        # Training
        best_val_acc = 0
        best_model_state = None
        patience = 0
        max_patience = 25
        
        epochs = config.get('epochs', 100)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
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
            train_acc = train_correct / train_total
            avg_val_loss = val_loss / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
            
            # Early stopping
            if patience >= max_patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return model, scaler, best_val_acc
    
    def train_ensemble(self, df, stage='after_set_4', n_models=7, use_bagging=True):
        """Train the complete ensemble"""
        
        print(f"\nSOLUTION 4: ENSEMBLE LEARNING FOR {stage.upper()}")
        
        # Prepare data
        X, y = self.enhanced_predictor.prepare_progressive_features(df, stage)
        print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
        
        if len(X) < 50:
            print("Dataset too small for ensemble")
            return None
        
        # Create model configurations
        configs = self.create_diverse_model_configs(n_models)
        print(f"Created {len(configs)} diverse model configurations")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # Train ensemble models
        print(f"\nTraining {len(configs)} ensemble models...")
        
        self.models = []
        self.scalers = []
        self.model_configs = []
        self.training_accuracies = []
        
        for i, config in enumerate(configs):
            # Set seed for reproducibility but vary across models - seed fixed!!!!
            #torch.manual_seed(42 + i * 7)
            #np.random.seed(42 + i * 7)
            # randomize seed for multirun
            seed = int(time.time() * 1000) % 10000 + i * 40
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Optional: Use bagging (different data subsets)
            if use_bagging and len(X_train) > 200:
                # Sample 80% of training data for this model
                bag_size = int(0.8 * len(X_train))
                bag_indices = np.random.choice(len(X_train), bag_size, replace=False)
                X_train_bag = X_train[bag_indices]
                y_train_bag = y_train[bag_indices]
            else:
                X_train_bag = X_train
                y_train_bag = y_train
            
            # Train this model
            model, scaler, val_acc = self.train_ensemble_model(
                X_train_bag, y_train_bag, X_val, y_val, config, i+1
            )
            
            self.models.append(model)
            self.scalers.append(scaler)
            self.model_configs.append(config)
            self.training_accuracies.append(val_acc)
        
        # Calculate model weights based on validation performance
        self.calculate_model_weights()
        
        # Evaluate ensemble
        ensemble_accuracy = self.evaluate_ensemble(X_test, y_test)
        
        print(f"\nENSEMBLE RESULTS:")
        print(f"Individual model accuracies:")
        for i, (config, acc) in enumerate(zip(self.model_configs, self.training_accuracies)):
            print(f"  Model {i+1} ({config['name'][:20]}): {acc:.4f}")
        
        print(f"\nEnsemble statistics:")
        print(f"Mean individual accuracy: {np.mean(self.training_accuracies):.4f}")
        print(f"Std individual accuracy: {np.std(self.training_accuracies):.4f}")
        print(f"Best individual accuracy: {np.max(self.training_accuracies):.4f}")
        print(f"Ensemble test accuracy: {ensemble_accuracy:.4f}")
        
        improvement_vs_baseline = ensemble_accuracy - 0.915
        improvement_vs_solution2 = ensemble_accuracy - 0.939
        
        print(f"\nComparison:")
        print(f"Improvement vs baseline (91.5%): {improvement_vs_baseline:+.4f}")
        print(f"Improvement vs Solution 2 (93.9%): {improvement_vs_solution2:+.4f}")
        
        if ensemble_accuracy > 0.95:
            print("Excellent, Ensemble achieves 95%+ accuracy")
        elif ensemble_accuracy > 0.94:
            print("Great, Strong ensemble performance")
        elif ensemble_accuracy > np.max(self.training_accuracies):
            print("Good, Ensemble beats best individual model")
        else:
            print("Ensemble provides stability but not accuracy gain")
        
        # Store results
        self.enhanced_predictor.models[f'{stage}_ensemble'] = {
            'ensemble': self,
            'test_accuracy': ensemble_accuracy,
            'individual_accuracies': self.training_accuracies,
            'model_configs': self.model_configs,
            'n_models': len(self.models)
        }
        
        return ensemble_accuracy
    
    def calculate_model_weights(self):
        """Calculate weights for ensemble based on validation performance"""
        
        # Strategy 1: Performance-based weighting
        accuracies = np.array(self.training_accuracies)
        
        # Softmax weighting (emphasizes better models)
        exp_accs = np.exp((accuracies - np.mean(accuracies)) * 10)
        performance_weights = exp_accs / np.sum(exp_accs)
        
        # Strategy 2: Diversity bonus
        # Models that make different predictions get bonus weight
        diversity_weights = np.ones(len(self.models)) / len(self.models)
        
        # Combine strategies
        self.model_weights = 0.7 * performance_weights + 0.3 * diversity_weights
        
        print(f"\nModel weights calculated:")
        for i, (config, weight) in enumerate(zip(self.model_configs, self.model_weights)):
            print(f"  Model {i+1}: {weight:.3f} ({config['name'][:15]})")
    
    def predict_ensemble(self, X):
        """Make ensemble prediction"""
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        all_predictions = []
        
        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            # Scale input
            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Get model prediction
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.append(probabilities.cpu().numpy())
        
        # Weighted ensemble prediction
        weighted_predictions = np.zeros_like(all_predictions[0])
        for i, pred in enumerate(all_predictions):
            weighted_predictions += self.model_weights[i] * pred
        
        return weighted_predictions
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble on test data"""
        
        ensemble_probs = self.predict_ensemble(X_test)
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        
        accuracy = np.mean(ensemble_predictions == y_test)
        return accuracy
    
    def predict_match_ensemble(self, player1, player2, surface, best_of=3, sets_so_far=None):
        """Predict match using ensemble"""
        
        if sets_so_far is None:
            sets_so_far = []
        
        # Determine stage
        num_sets = len(sets_so_far)
        if num_sets == 0:
            stage = 'pre_match'
        elif num_sets <= 4:
            stage = f'after_set_{num_sets}'
        else:
            stage = 'after_set_4'
        
        # Check if we have ensemble for this stage
        ensemble_key = f'{stage}_ensemble'
        if ensemble_key not in self.enhanced_predictor.models:
            raise ValueError(f"No ensemble trained for stage '{stage}'")
        
        # Prepare features (similar to original predictor)
        player1_features = self.enhanced_predictor.get_enhanced_player_features(player1, surface, best_of)
        player2_features = self.enhanced_predictor.get_enhanced_player_features(player2, surface, best_of)
        
        surface_encoded = self.enhanced_predictor.surface_encoder.transform([surface])[0]
        best_of_encoded = 1 if best_of == 5 else 0
        
        player1_set_features = self.enhanced_predictor.calculate_set_features(sets_so_far, True)
        player2_set_features = self.enhanced_predictor.calculate_set_features(sets_so_far, False)
        
        feature_vector = np.array([player1_features + player2_features + 
                                 [surface_encoded, best_of_encoded] + 
                                 player1_set_features + player2_set_features])
        
        # Make ensemble prediction
        ensemble_probs = self.predict_ensemble(feature_vector)
        
        player1_win_prob = ensemble_probs[0][1]
        player2_win_prob = ensemble_probs[0][0]
        
        return {
            'stage': stage,
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'ensemble_size': len(self.models),
            'player1_win_probability': player1_win_prob,
            'player2_win_probability': player2_win_prob,
            'predicted_winner': player1 if player1_win_prob > player2_win_prob else player2,
            'confidence': max(player1_win_prob, player2_win_prob),
            'prediction_method': 'weighted_ensemble'
        }

def test_solution_4(enhanced_predictor, df, n_models=7):
    """Test Solution 4: Ensemble Learning"""
    
    print("TESTING SOLUTION 4: ENSEMBLE LEARNING")
    
    # Create ensemble
    ensemble = TennisEnsemble(enhanced_predictor)
    
    # Train ensemble
    ensemble_accuracy = ensemble.train_ensemble(
        df, 
        stage='after_set_4', 
        n_models=n_models,
        use_bagging=True
    )
    
    if ensemble_accuracy:
        print(f"\nSOLUTION 4 SUMMARY:")
        print(f"Ensemble accuracy: {ensemble_accuracy:.1%}")
        print(f"Number of models: {len(ensemble.models)}")
        print(f"Improvement vs baseline: {ensemble_accuracy - 0.915:+.1%}")
        
        # Compare with other solutions
        print(f"\nSOLUTION COMPARISON:")
        print(f"Solution 2 (Regularization): 93.9%")
        print(f"Solution 3 (Transfer best): 95.6%")
        print(f"Solution 3 (Transfer avg): ~93.0%")
        print(f"Solution 4 (Ensemble): {ensemble_accuracy:.1%}")
        
        if ensemble_accuracy > 0.956:
            print("Ensemble beats all previous solutions")
        elif ensemble_accuracy > 0.939:
            print("Ensemble shows strong performance")
        
        return ensemble_accuracy, ensemble
    
    return None, None

# if __name__ == "__main__":
#     print("Solution 4: Ensemble Learning ready")
#     print("Usage: accuracy, ensemble = test_solution_4(enhanced_predictor, df_2023, n_models=7)")
