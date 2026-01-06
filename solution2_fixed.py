'''
Author: Audrey Wang (with assistance in debugging and refactoring from AI)
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

class FixedImprovedTennisTransformer(nn.Module):
    """Fixed transformer with consistent architecture for saving/loading"""
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_classes=2, 
                 dropout_rate=0.3, l2_strength=0.001):
        super(FixedImprovedTennisTransformer, self).__init__()
        
        # Store architecture parameters for consistent loading
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.l2_strength = l2_strength
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Enhanced transformer with configurable dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced classification head with more regularization
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.5),  # Higher dropout for final layers
            nn.Linear(d_model // 2, d_model // 4),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 4, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x).unsqueeze(1)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)
    
    def get_l2_loss(self):
        """Calculate L2 regularization loss"""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param) ** 2
        return self.l2_strength * l2_loss

def train_enhanced_regularization_fixed(enhanced_predictor, df, stage='after_set_4', epochs=200):
    """
    Fixed version of enhanced regularization training
    """
    print(f"\nSOLUTION 2: ENHANCED REGULARIZATION FOR {stage.upper()}")
    
    # Initialize predictor with different random state for each run
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    # Prepare data
    X, y = enhanced_predictor.prepare_progressive_features(df, stage)
    print(f"Dataset size: {len(X)} samples")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    if len(X) < 50:
        print("Dataset too small for reliable training")
        return None
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")  
    print(f"Test samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
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
    test_dataset = TennisDataset(X_test_scaled, y_test)
    
    # Smaller batch size for small datasets
    batch_size = min(16, max(4, len(X_train) // 8))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Enhanced regularization parameters - consistent architecture
    input_dim = X_train.shape[1]
    
    # Fixed architecture parameters
    model_config = {
        'd_model': 64,  # Smaller model for small dataset
        'nhead': 4,     # Fewer attention heads
        'num_layers': 2,  # Fewer layers
        'dropout_rate': 0.3,  # Higher dropout
        'l2_strength': 0.001  # L2 regularization
    }
    
    model = FixedImprovedTennisTransformer(
        input_dim=input_dim,
        **model_config
    ).to(enhanced_predictor.device)
    
    print(f"Model architecture:")
    print(f" - d_model: {model_config['d_model']}")
    print(f" - nhead: {model_config['nhead']}")
    print(f" - num_layers: {model_config['num_layers']}")
    print(f" - dropout_rate: {model_config['dropout_rate']}")
    print(f" - Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training with enhanced regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0005,  # Lower learning rate
        weight_decay=0.01  # Weight decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6
    )
    
    # Early stopping with more patience
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    max_patience = 40
    
    print(f"Training with enhanced regularization...")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")
    print(f"Early stopping patience: {max_patience}")
    
    training_history = {
        'train_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(enhanced_predictor.device)
            batch_labels = batch_labels.to(enhanced_predictor.device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # Loss with L2 regularization
            ce_loss = criterion(outputs, batch_labels)
            l2_loss = model.get_l2_loss()
            total_loss = ce_loss + l2_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(enhanced_predictor.device)
                batch_labels = batch_labels.to(enhanced_predictor.device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save training history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(current_lr)
        
        # Save best model state in memory (avoid file I/O issues)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model_state_dict': model.state_dict().copy(),
                'model_config': model_config,
                'scaler': scaler,
                'input_dim': input_dim,
                'epoch': epoch,
                'val_acc': val_acc
            }
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 25 == 0 or epoch < 5:
            print(f'Epoch {epoch:3d}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'LR: {current_lr:.6f}')
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f'\nTraining completed')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f"Loaded best model from epoch {best_model_state['epoch']}")
    else:
        print("Warning: No best model state saved")
        return None
    
    # Final test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(enhanced_predictor.device)
            batch_labels = batch_labels.to(enhanced_predictor.device)
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    test_accuracy = test_correct / test_total
    
    print(f'\nFINAL RESULTS - SOLUTION 2 (Enhanced Regularization):')
    print(f'Training samples used: {len(X_train):,}')
    print(f'Model architecture: d_model={model_config["d_model"]}, layers={model_config["num_layers"]}')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Final test accuracy: {test_accuracy:.4f}')
    print(f'Improvement over baseline (91.5%): {test_accuracy - 0.915:+.4f}')
    
    # Store in enhanced predictor
    enhanced_predictor.models[f'{stage}_regularized'] = {
        'model': model,
        'model_state': best_model_state,
        'scaler': scaler,
        'input_dim': input_dim,
        'test_accuracy': test_accuracy,
        'training_samples': len(X_train),
        'config': model_config
    }
    
    # Training analysis
    print(f'\nTraining Analysis:')
    print(f'Epochs completed: {len(training_history["train_loss"])}')
    print(f'Final learning rate: {training_history["learning_rates"][-1]:.6f}')
    print(f'Training loss: {training_history["train_loss"][0]:.4f} → {training_history["train_loss"][-1]:.4f}')
    print(f'Validation accuracy: {training_history["val_acc"][0]:.4f} → {training_history["val_acc"][-1]:.4f}')
    
    return test_accuracy

def test_solution_2_fixed(enhanced_predictor, df):
    """Test the fixed Solution 2"""
    
    print("TESTING FIXED SOLUTION 2")
    
    # Test enhanced regularization
    regularized_accuracy = train_enhanced_regularization_fixed(
        enhanced_predictor, 
        df, 
        stage='after_set_4', 
        epochs=200
    )
    
    if regularized_accuracy:
        baseline = 0.915
        improvement = regularized_accuracy - baseline
        
        print(f"\nSOLUTION 2 RESULTS:")
        print(f"Baseline (original): 91.5%")
        print(f"Enhanced regularization: {regularized_accuracy:.1%}")
        print(f"Improvement: {improvement:+.1%}")
        
        if improvement > 0.01:
            print("Good improvement")
        elif improvement > 0:
            print("Small improvement")
        else:
            print("No improvement - try other solutions")
    else:
        print("Training failed")
    
    return regularized_accuracy

# # Usage example
# if __name__ == "__main__":
#     print("Fixed Solution 2 ready")
#     print("Usage:")
#     print("accuracy = test_solution_2_fixed(enhanced_predictor, df_2023)")
