"""Data preprocessing and normalization."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Any, Optional, Tuple

from src.utils.logger import setup_logger


logger = setup_logger(__name__)


class DataPreprocessor:
    """Preprocess and normalize data for model training."""
    
    def __init__(self, scaler_type: str = "standard"):
        """Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names: List[str] = []
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        """Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            target_col: Name of target column to exclude
        """
        logger.info("Fitting preprocessor on training data")
        
        # Separate features and target
        if target_col:
            features_df = df.drop(columns=[target_col])
        else:
            features_df = df
        
        # Identify feature types
        self.numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Found {len(self.numeric_features)} numeric and {len(self.categorical_features)} categorical features")
        
        # Fit scaler on numeric features
        if self.numeric_features:
            self.scaler.fit(features_df[self.numeric_features])
        
        self.feature_names = features_df.columns.tolist()
    
    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            target_col: Name of target column to exclude
            
        Returns:
            Transformed DataFrame
        """
        # Separate features and target
        if target_col and target_col in df.columns:
            target = df[target_col]
            features_df = df.drop(columns=[target_col])
        else:
            target = None
            features_df = df.copy()
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Scale numeric features
        if self.numeric_features:
            features_df[self.numeric_features] = self.scaler.transform(
                features_df[self.numeric_features]
            )
        
        # Encode categorical features
        if self.categorical_features:
            features_df = self._encode_categorical(features_df)
        
        # Add target back if it was separated
        if target is not None:
            features_df[target_col] = target
        
        return features_df
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Fit and transform data in one step.
        
        Args:
            df: DataFrame to fit and transform
            target_col: Name of target column to exclude
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, target_col)
        return self.transform(df, target_col)
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Fill numeric missing values with median
        for col in self.numeric_features:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Fill categorical missing values with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding.
        
        Args:
            df: DataFrame with categorical features
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        for col in self.categorical_features:
            if col in df.columns:
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
        
        return df
    
    def prepare_features_dict(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare a feature dictionary for prediction.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            NumPy array ready for model input
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([features])
        
        # Transform
        df_transformed = self.transform(df)
        
        # Return as array
        return df_transformed.values
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation.
        
        Returns:
            List of feature names
        """
        return self.feature_names


class DataSplitter:
    """Split data into train/validation/test sets."""
    
    @staticmethod
    def split_by_date(
        df: pd.DataFrame,
        date_col: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by date to avoid leakage.
        
        Args:
            df: DataFrame to split
            date_col: Name of date column
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
        
        # Sort by date
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def split_by_season(
        df: pd.DataFrame,
        season_col: str,
        train_seasons: List[int],
        val_seasons: List[int],
        test_seasons: List[int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by season.
        
        Args:
            df: DataFrame to split
            season_col: Name of season column
            train_seasons: List of seasons for training
            val_seasons: List of seasons for validation
            test_seasons: List of seasons for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = df[df[season_col].isin(train_seasons)]
        val_df = df[df[season_col].isin(val_seasons)]
        test_df = df[df[season_col].isin(test_seasons)]
        
        logger.info(f"Split by season: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df

