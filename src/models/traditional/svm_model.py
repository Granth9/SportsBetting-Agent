"""Support Vector Machine model for betting predictions."""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, Any, Optional

from src.models.base_model import BaseModel
from src.utils.data_types import ModelPrediction, Outcome
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class SVMModel(BaseModel):
    """Support Vector Machine predictor."""
    
    def __init__(
        self,
        name: str = "SVM Strategist",
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        **kwargs
    ):
        """Initialize the SVM model.
        
        Args:
            name: Model name
            C: Regularization parameter
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient ('scale', 'auto', or float)
            **kwargs: Additional sklearn parameters
        """
        super().__init__(name, "svm")
        
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            random_state=42,
            **kwargs
        )
    
        # SVM requires feature scaling
        self.scaler = StandardScaler()
        self._scaler_fitted = False
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs) -> None:
        """Train the SVM model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional, not used by this model)
            y_val: Validation targets (optional, not used by this model)
            **kwargs: Additional training parameters
        """
        logger.info(f"Training {self.name} on {len(X)} samples")
        
        # Scale features (required for SVM)
        X_scaled = self.scaler.fit_transform(X)
        self._scaler_fitted = True
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
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
        
        # Scale features
        if not self._scaler_fitted:
            raise ValueError("Scaler must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Predict probability
        probs = self.model.predict_proba(X_scaled)[0]
        pred_class = int(self.model.predict(X_scaled)[0])
        
        # Determine outcome
        prediction = Outcome.HOME_WIN if pred_class == 1 else Outcome.AWAY_WIN
        confidence = float(probs[pred_class])
        
        # Get support vectors info (SVM doesn't have traditional feature importance)
        key_features = self._get_support_info()
        
        reasoning = f"SVM model predicts {prediction.value} with {confidence:.1%} confidence. "
        reasoning += f"Based on support vector classification with {self.kernel} kernel."
        
        return ModelPrediction(
            model_name=self.name,
            prediction=prediction,
            confidence=confidence,
            probability=float(probs[pred_class]),
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
        
        if not self._scaler_fitted:
            raise ValueError("Scaler must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]
        
        return {
            Outcome.AWAY_WIN: float(probs[0]),
            Outcome.HOME_WIN: float(probs[1])
        }
    
    def _get_support_info(self) -> Dict[str, Any]:
        """Get information about support vectors.
        
        Returns:
            Dictionary with support vector information
        """
        if not self.is_trained:
            return {}
        
        return {
            'n_support_vectors': len(self.model.support_vectors_),
            'kernel': self.kernel,
            'C': self.C
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.
        
        Note: SVM doesn't provide traditional feature importance.
        For linear kernel, we can use coefficients.
        
        Returns:
            Dictionary of feature importances (only for linear kernel)
        """
        if not self.is_trained or self.kernel != 'linear':
            return None
        
        if not hasattr(self.model, 'coef_'):
            return None
        
            coef = np.abs(self.model.coef_[0])
            
                # Normalize
                total = np.sum(coef)
                if total > 0:
                    coef = coef / total
        
        if self.feature_names and len(self.feature_names) == len(coef):
                return dict(zip(self.feature_names, coef))
        
        return {f'feature_{i}': float(c) for i, c in enumerate(coef)}
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        from pathlib import Path
        import joblib
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'scaler_fitted': self._scaler_fitted,
            'name': self.name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'performance_history': self.performance_history,
            'feature_names': self.feature_names,
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma
        }
        
        joblib.dump(model_data, save_path)
    
    def load(self, path: str) -> None:
        """Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        import joblib
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self._scaler_fitted = model_data.get('scaler_fitted', False)
        self.name = model_data['name']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.performance_history = model_data.get('performance_history', [])
        self.feature_names = model_data.get('feature_names', [])
        self.C = model_data.get('C', 1.0)
        self.kernel = model_data.get('kernel', 'rbf')
        self.gamma = model_data.get('gamma', 'scale')
