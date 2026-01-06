'''
Author: Audrey Wang (with assistance in debugging and refactoring from AI)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def safe_torch_load(filepath):
    """Load PyTorch models safely across versions"""
    try:
        return torch.load(filepath, weights_only=False)
    except TypeError:
        return torch.load(filepath)

class AlignedTennisTransformer(nn.Module):
    """Transformer designed for better transfer learning between stages"""
    
    def __init__(self, input_dim, d_model=96, nhead=6, num_layers=2, num_classes=2):
        super(AlignedTennisTransformer, self).__init__()
        
        # Store dimensions for transfer compatibility
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Feature decomposition for better transfer
        self.player_processor = nn.Sequential(
            nn.Linear(30, d_model // 3), # Player features (15+15)
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.context_processor = nn.Sequential(
            nn.Linear(2, d_model // 6), # Surface + format
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Set features processor (variable size handling)
        remaining_features = input_dim - 32 # Subtract player + context features
        self.set_processor = nn.Sequential(
            nn.Linear(remaining_features, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combine all features
        self.feature_combiner = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.15,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Decompose features
        player_features = x[:, :30]     # First 30 features (15+15 player stats)
        context_features = x[:, 30:32]  # Next 2 features (surface + format)  
        set_features = x[:, 32:]        # Remaining features (set information)
        
        # Process each component
        player_processed = self.player_processor(player_features)
        context_processed = self.context_processor(context_features)
        set_processed = self.set_processor(set_features)
        
        # Combine features
        combined = torch.cat([player_processed, context_processed, set_processed], dim=1)
        combined = self.feature_combiner(combined)
        
        # Add positional encoding and process through transformer
        combined = combined.unsqueeze(1) + self.pos_encoding
        transformed = self.transformer(combined)
        
        # Classify
        output = self.classifier(transformed.squeeze(1))
        return output
    
    def get_transferable_state_dict(self):
        """Get state dict with only transferable components"""
        state_dict = self.state_dict()
        transferable = {}
        
        # Transfer these components (architecture-independent)
        transferable_keys = [
            'player_processor',
            'context_processor', 
            'feature_combiner',
            'pos_encoding',
            'transformer',
            'classifier'
        ]
        
        for key, value in state_dict.items():
            for transferable_key in transferable_keys:
                if key.startswith(transferable_key):
                    transferable[key] = value
                    break
        
        return transferable

def train_aligned_source_model(enhanced_predictor, df, source_stage='after_set_3'):
    """Train source model with aligned architecture"""
    
    print(f"Training aligned source model ({source_stage})...")
    
    # Prepare source data
    X_source, y_source = enhanced_predictor.prepare_progressive_features(df, source_stage)
    
    if len(X_source) < 100:
        print(f"Insufficient data for source model: {len(X_source)} samples")
        return None
    
    print(f"Source dataset: {len(X_source)} samples, {X_source.shape[1]} features")
    
    # Split and scale
    X_train, X_val, y_train, y_val = train_test_split(X_source, y_source, test_size=0.2, random_state=42)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create aligned model
    source_model = AlignedTennisTransformer(
        input_dim=X_source.shape[1],
        d_model=96,
        nhead=6,
        num_layers=2
    ).to(enhanced_predictor.device)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(source_model.parameters(), lr=0.001, weight_decay=0.01)
    
    best_val_acc = 0
    best_state = None
    
    print(f"Training source model for 100 epochs...")
    
    for epoch in range(100):
        # Training
        source_model.train()
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(enhanced_predictor.device)
            batch_labels = batch_labels.to(enhanced_predictor.device)
            
            optimizer.zero_grad()
            outputs = source_model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        # Validation every 20 epochs
        if epoch % 20 == 0:
            source_model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(enhanced_predictor.device)
                    batch_labels = batch_labels.to(enhanced_predictor.device)
                    outputs = source_model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            val_acc = val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    'model_state_dict': source_model.state_dict().copy(),
                    'transferable_state_dict': source_model.get_transferable_state_dict(),
                    'scaler': scaler,
                    'input_dim': X_source.shape[1]
                }
            
            print(f"Epoch {epoch}: Validation Accuracy = {val_acc:.4f}")
    
    if best_state is not None:
        source_model.load_state_dict(best_state['model_state_dict'])
        print(f"Source model trained with {best_val_acc:.4f} accuracy")
        return source_model, best_state
    else:
        print("Source model training failed")
        return None, None

def improved_transfer_learning(enhanced_predictor, df, source_stage='after_set_3', target_stage='after_set_4'):
    """Improved transfer learning with better architecture alignment"""
    
    print(f"\nSOLUTION 3.2: IMPROVED TRANSFER LEARNING")
    print(f"Source: {source_stage.upper()} → Target: {target_stage.upper()}")
    
    # Step 1: Train or load source model
    source_model, source_state = train_aligned_source_model(enhanced_predictor, df, source_stage)
    
    if source_model is None:
        print("Failed to train source model")
        return None
    
    # Step 2: Prepare target data
    X_target, y_target = enhanced_predictor.prepare_progressive_features(df, target_stage)
    print(f"Target dataset: {len(X_target)} samples, {X_target.shape[1]} features")
    
    if len(X_target) < 30:
        print("Insufficient target data")
        return None
    
    # Step 3: Create target model with same architecture concepts
    target_model = AlignedTennisTransformer(
        input_dim=X_target.shape[1],
        d_model=96,
        nhead=6, 
        num_layers=2
    ).to(enhanced_predictor.device)
    
    # Step 4: Smart transfer learning
    print("\nTransferring knowledge...")
    
    source_transferable = source_state['transferable_state_dict']
    target_state_dict = target_model.state_dict()
    
    transferred_count = 0
    for key, value in source_transferable.items():
        if key in target_state_dict and value.shape == target_state_dict[key].shape:
            target_state_dict[key] = value.clone()
            transferred_count += 1
            print(f"Transferred: {key}")
        else:
            print(f"Skipped: {key} (shape mismatch or missing)")
    
    target_model.load_state_dict(target_state_dict)
    print(f"Successfully transferred {transferred_count} parameter tensors")
    
    # Step 5: Prepare target training
    X_train, X_test, y_train, y_test = train_test_split(X_target, y_target, test_size=0.2, random_state=42)
    
    # Use same scaler approach for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
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
    test_dataset = TennisDataset(X_test_scaled, y_test)
    
    # Small batch size for fine-tuning
    batch_size = min(8, len(X_train) // 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Step 6: Fine-tuning with very conservative parameters
    print(f"\nFine-tuning on target data...")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Batch size: {batch_size}")
    
    # Conservative fine-tuning
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for different parts
    transferred_params = []
    new_params = []
    
    for name, param in target_model.named_parameters():
        if name in source_transferable:
            transferred_params.append(param)
        else:
            new_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': transferred_params, 'lr': 0.00005},  # Very low LR for transferred
        {'params': new_params, 'lr': 0.0002}            # Higher LR for new params
    ], weight_decay=0.01)
    
    # Fine-tuning loop
    target_model.train()
    
    print("Fine-tuning epochs:")
    for epoch in range(50): # Fewer epochs for fine-tuning
        epoch_loss = 0
        num_batches = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(enhanced_predictor.device)
            batch_labels = batch_labels.to(enhanced_predictor.device)
            
            optimizer.zero_grad()
            outputs = target_model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gentle gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if epoch % 10 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    # Step 7: Final evaluation
    target_model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(enhanced_predictor.device)
            batch_labels = batch_labels.to(enhanced_predictor.device)
            
            outputs = target_model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
    
    test_accuracy = test_correct / test_total
    
    print(f'\nSOLUTION 3.2 RESULTS (Improved Transfer Learning):')
    print(f'Source model accuracy: {source_state["scaler"] if "scaler" in source_state else "N/A"}')
    print(f'Transferred parameters: {transferred_count}')
    print(f'Target test accuracy: {test_accuracy:.4f}')
    print(f'Improvement over baseline (91.5%): {test_accuracy - 0.915:+.4f}')
    
    baseline_comparison = test_accuracy - 0.915
    solution2_comparison = test_accuracy - 0.939
    
    print(f'Improvement over Solution 2 (93.9%): {solution2_comparison:+.4f}')
    
    if test_accuracy > 0.939:
        print('Transfer learning beats enhanced regularization')
    elif test_accuracy > 0.920:
        print('Good improvement over baseline')
    else:
        print('Modest improvement - ensemble might help more')
    
    # Store results
    enhanced_predictor.models[f'{target_stage}_transfer_improved'] = {
        'model': target_model,
        'scaler': scaler,
        'input_dim': X_target.shape[1],
        'test_accuracy': test_accuracy,
        'transferred_params': transferred_count,
        'source_stage': source_stage
    }
    
    return test_accuracy

def compare_transfer_approaches(enhanced_predictor, df):
    """Compare original vs improved transfer learning"""
    
    print("COMPARING TRANSFER LEARNING APPROACHES")
    
    # Test improved approach
    improved_accuracy = improved_transfer_learning(
        enhanced_predictor, df, 'after_set_3', 'after_set_4'
    )
    
    # Get original approach result (you already ran this)
    original_accuracy = 0.9123  # Your result
    
    print(f"\nTRANSFER LEARNING COMPARISON:")
    print(f"Original approach: {original_accuracy:.1%}")
    print(f"Improved approach: {improved_accuracy:.1%}")
    print(f"Improvement: {improved_accuracy - original_accuracy:+.1%}")
    
    print(f"\nOVERALL SOLUTION RANKING:")
    print(f"Solution 2 (Regularization): 93.9%")
    print(f"Solution 3 (Original Transfer): {original_accuracy:.1%}")
    print(f"Solution 3.2 (Improved Transfer): {improved_accuracy:.1%}")
    
    return improved_accuracy

# Usage function
def test_solution_3_fixed(enhanced_predictor, df):
    """Test the fixed Solution 3"""
    
    print("TESTING IMPROVED SOLUTION 3: TRANSFER LEARNING")
    
    accuracy = compare_transfer_approaches(enhanced_predictor, df)
    
    return accuracy

# if __name__ == "__main__":
#     print("Improved Solution 3 ready")
#     print("Usage: accuracy = test_solution_3_fixed(enhanced_predictor, df_2023)")

    