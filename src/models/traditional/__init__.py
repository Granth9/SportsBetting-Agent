"""Traditional ML models (XGBoost, Random Forest, etc.)."""

from src.models.traditional.gradient_boost_model import GradientBoostModel
from src.models.traditional.random_forest_model import RandomForestModel
from src.models.traditional.statistical_model import StatisticalModel
from src.models.traditional.lightgbm_model import LightGBMModel
from src.models.traditional.svm_model import SVMModel

# CatBoost is optional
try:
    from src.models.traditional.catboost_model import CatBoostModel
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

__all__ = [
    'GradientBoostModel',
    'RandomForestModel',
    'StatisticalModel',
    'LightGBMModel',
    'SVMModel'
]

if CATBOOST_AVAILABLE:
    __all__.append('CatBoostModel')
