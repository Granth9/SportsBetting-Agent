"""Deep neural network model for betting predictions."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class BettingNN(nn.Module):
    """Neural network architecture for betting predictions."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.3):
        """Initialize the neural network.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super(BettingNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (2 classes: home_win, away_win)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """Forward pass."""
        logits = self.network(x)
        probs = self.softmax(logits)
        return probs


class DeepPredictor(BaseModel):
    """Deep neural network predictor."""
    
    def __init__(
        self,
        name: str = "Neural Analyst",
        hidden_layers: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001
    ):
        """Initialize the deep predictor.
        
        Args:
            name: Model name
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Number of epochs to wait before stopping
            early_stopping_min_delta: Minimum change to qualify as improvement
        """
        super().__init__(name, "neural_network")
        
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = None
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train the neural network.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        logger.info(f"Training {self.name} on {len(X)} samples")
        
        # Initialize model
        self.input_dim = X.shape[1]
        self.model = BettingNN(self.input_dim, self.hidden_layers, self.dropout).to(self.device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Validation setup
        val_dataloader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Validation and early stopping
            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                val_loss = val_loss / len(val_dataloader)
                self.model.train()
                
                # Check for improvement
                if val_loss < best_val_loss - self.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"{self.name} training completed")
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make a prediction.
        
        Args:
            X: Input features
            
        Returns:
            ModelPrediction object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probs = self.model(X_tensor).cpu().numpy()[0]
        
        # Get prediction and confidence
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        
        # Map to outcome (0 = AWAY_WIN, 1 = HOME_WIN)
        prediction = Outcome.HOME_WIN if pred_idx == 1 else Outcome.AWAY_WIN
        
        # Get key features (top contributors)
        key_features = self._get_key_features(X)
        
        reasoning = f"Neural network predicts {prediction.value} with {confidence:.1%} confidence based on deep pattern analysis."
        
        return ModelPrediction(
            model_name=self.name,
            prediction=prediction,
            confidence=confidence,
            probability=confidence,
            key_features=key_features,
            reasoning=reasoning
        )
    
    def predict_proba(self, X: np.ndarray) -> Dict[Outcome, float]:
        """Get probability distribution.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of outcome probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probs = self.model(X_tensor).cpu().numpy()[0]
        
        # probs has shape [2] for binary classification
        return {
            Outcome.AWAY_WIN: float(probs[0]),
            Outcome.HOME_WIN: float(probs[1])
        }
    
    def _get_key_features(self, X: np.ndarray) -> Dict[str, Any]:
        """Extract key features that influenced the prediction.
        
        Uses gradient-based feature importance (integrated gradients).
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of key features with importance scores
        """
        if not self.is_trained or not self.feature_names:
            return {
                'feature_count': X.shape[1],
                'analysis_method': 'deep_learning'
            }
        
        # Use gradient-based feature importance
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(X_tensor)
        
        # Get gradients
        output[0, 1].backward()  # Gradient w.r.t. HOME_WIN probability
        gradients = X_tensor.grad[0].cpu().numpy()
        
        # Get top features by absolute gradient
        if len(self.feature_names) == len(gradients):
            feature_importance = dict(zip(self.feature_names, np.abs(gradients)))
            # Normalize
            total = sum(feature_importance.values())
            if total > 0:
                feature_importance = {k: v/total for k, v in feature_importance.items()}
            
            # Return top 5 features
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
            return top_features
        
        return {
            'feature_count': X.shape[1],
            'analysis_method': 'deep_learning'
        }
    
    def save(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Save path
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state': self.model.state_dict() if self.model else None,
            'name': self.name,
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'is_trained': self.is_trained
        }, save_path)
        
        logger.info(f"Saved {self.name} to {save_path}")
    
    def load(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.name = checkpoint['name']
        self.input_dim = checkpoint['input_dim']
        self.hidden_layers = checkpoint['hidden_layers']
        self.dropout = checkpoint['dropout']
        self.is_trained = checkpoint['is_trained']
        
        if checkpoint['model_state']:
            self.model = BettingNN(self.input_dim, self.hidden_layers, self.dropout).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
        
        logger.info(f"Loaded {self.name} from {path}")

