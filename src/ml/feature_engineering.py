"""
Feature Engineering Module

Provides utilities for feature extraction, normalization, and importance analysis.
Complements the clustering pipeline with additional feature processing capabilities.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import structlog

logger = structlog.get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering utilities for psychographic profiling

    Provides:
    - Feature extraction from UserMetrics
    - Multiple normalization strategies
    - Missing value imputation
    - Feature importance analysis
    - PCA dimensionality reduction
    """

    # Core features from UserMetrics (matches metrics_pipeline)
    CORE_FEATURES = [
        # Emotion balance (2)
        'avg_valence', 'avg_arousal',

        # Volatility (4)
        'valence_stddev', 'arousal_stddev', 'emotion_stability', 'topic_volatility',

        # Toxicity/Hostility (2)
        'mean_toxicity', 'mean_hostility',

        # P0: Safety & Hostility (3)
        'avg_safety_concern', 'hostility_incident_count', 'breakthrough_count',

        # P1: Empathy & Meta-Emotional (3)
        'avg_empathic_concern', 'avg_personal_distress', 'avg_meta_emotional_score',

        # Engagement (2)
        'avg_engagement_score', 'reaction_bias',

        # Temporal patterns (1)
        'peak_activity_hour',

        # Social influence (2)
        'influence_score', 'follower_log',

        # Cognitive complexity (3)
        'topic_diversity', 'entropy_score', 'avg_content_length',

        # Trust/Alignment (2)
        'trust_level', 'agreement_rate',

        # Post statistics (1)
        'post_count'
    ]

    # Plutchik 8 emotions (backward compat)
    EMOTION_FEATURES = [
        'joy', 'trust', 'fear', 'surprise',
        'sadness', 'disgust', 'anger', 'anticipation'
    ]

    # Full 27 GoEmotions from SDK active_labels (P1)
    GOEMOTION_FEATURES = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse',
        'sadness', 'surprise'
    ]

    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize feature engineer

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler(scaler_type)
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False

    def _create_scaler(self, scaler_type: str):
        """Create scaler based on type"""
        scalers = {
            'standard': StandardScaler(),  # Mean=0, Std=1
            'minmax': MinMaxScaler(),      # Scale to [0, 1]
            'robust': RobustScaler()       # Robust to outliers
        }

        if scaler_type not in scalers:
            logger.warning(
                "invalid_scaler_type",
                scaler_type=scaler_type,
                default='standard'
            )
            return StandardScaler()

        return scalers[scaler_type]

    def extract_features_dict(
        self,
        metrics: List[Any]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract features from UserMetrics objects

        Args:
            metrics: List of UserMetrics objects

        Returns:
            (feature_matrix, user_ids, feature_names)
            - feature_matrix: (n_users, 35) array
            - user_ids: List of user IDs
            - feature_names: List of feature names
        """
        user_ids = []
        features = []

        for metric in metrics:
            user_ids.append(metric.user_id)
            feature_row = []

            # Extract core features
            for feat_name in self.CORE_FEATURES:
                value = getattr(metric, feat_name, None)

                # Handle None values
                if value is None:
                    if feat_name == 'peak_activity_hour':
                        value = 12  # Noon default
                    elif feat_name == 'post_count':
                        value = 0
                    else:
                        value = 0.0

                feature_row.append(float(value))

            # Extract Plutchik emotion distribution (backward compat)
            emotion_dist = metric.emotion_distribution or {}
            for emotion in self.EMOTION_FEATURES:
                feature_row.append(float(emotion_dist.get(emotion, 0.0)))

            # Extract GoEmotions distribution from active_labels (P1)
            labels_dist = getattr(metric, 'active_labels_distribution', None) or {}
            for emotion in self.GOEMOTION_FEATURES:
                feature_row.append(float(labels_dist.get(emotion, 0.0)))

            features.append(feature_row)

        feature_names = self.CORE_FEATURES + self.EMOTION_FEATURES + self.GOEMOTION_FEATURES

        return np.array(features), user_ids, feature_names

    def fit_transform(
        self,
        features: np.ndarray,
        impute_missing: bool = True
    ) -> np.ndarray:
        """
        Fit scaler and transform features

        Args:
            features: Raw feature matrix
            impute_missing: Whether to impute missing values

        Returns:
            Transformed feature matrix
        """
        # Handle missing values
        if impute_missing:
            features = self.imputer.fit_transform(features)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        self.is_fitted = True

        logger.info(
            "features_fit_transformed",
            n_samples=features.shape[0],
            n_features=features.shape[1],
            scaler_type=self.scaler_type
        )

        return features_scaled

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler

        Args:
            features: Raw feature matrix

        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer not fitted. Call fit_transform first.")

        return self.scaler.transform(features)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale

        Args:
            features: Scaled feature matrix

        Returns:
            Original scale features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer not fitted.")

        return self.scaler.inverse_transform(features)

    def calculate_feature_importance(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculate feature importance using variance analysis

        Args:
            features: Feature matrix
            labels: Cluster labels
            feature_names: Names of features

        Returns:
            DataFrame with feature importances
        """
        from sklearn.ensemble import RandomForestClassifier

        # Train random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, labels)

        # Get feature importances
        importances = rf.feature_importances_

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        logger.info(
            "feature_importance_calculated",
            n_features=len(feature_names),
            top_feature=importance_df.iloc[0]['feature'],
            top_importance=float(importance_df.iloc[0]['importance'])
        )

        return importance_df

    def apply_pca(
        self,
        features: np.ndarray,
        n_components: int = 64,
        explained_variance_threshold: float = 0.95
    ) -> Tuple[np.ndarray, PCA, Dict[str, Any]]:
        """
        Apply PCA dimensionality reduction

        Args:
            features: Feature matrix
            n_components: Target number of components
            explained_variance_threshold: Min variance to explain

        Returns:
            (transformed_features, pca_model, metadata)
        """
        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42)
        features_pca = pca.fit_transform(features)

        # Calculate metadata
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_for_threshold = np.argmax(
            explained_variance >= explained_variance_threshold
        ) + 1

        metadata = {
            'n_components': n_components,
            'explained_variance_total': float(np.sum(pca.explained_variance_ratio_)),
            'explained_variance_per_component': pca.explained_variance_ratio_.tolist(),
            'n_components_for_95_variance': int(n_components_for_threshold)
        }

        logger.info(
            "pca_applied",
            original_dim=features.shape[1],
            reduced_dim=n_components,
            explained_variance=metadata['explained_variance_total']
        )

        return features_pca, pca, metadata

    def detect_outliers(
        self,
        features: np.ndarray,
        contamination: float = 0.1
    ) -> np.ndarray:
        """
        Detect outliers using Isolation Forest

        Args:
            features: Feature matrix
            contamination: Expected proportion of outliers

        Returns:
            Boolean array (True = outlier)
        """
        from sklearn.ensemble import IsolationForest

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        predictions = iso_forest.fit_predict(features)

        # -1 = outlier, 1 = inlier
        outliers = predictions == -1

        logger.info(
            "outliers_detected",
            n_outliers=int(np.sum(outliers)),
            outlier_rate=float(np.mean(outliers))
        )

        return outliers

    def create_feature_statistics(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculate feature statistics

        Args:
            features: Feature matrix
            feature_names: Names of features

        Returns:
            DataFrame with statistics
        """
        stats = pd.DataFrame({
            'feature': feature_names,
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0),
            'q25': np.percentile(features, 25, axis=0),
            'q75': np.percentile(features, 75, axis=0)
        })

        return stats

    def select_top_features(
        self,
        importance_df: pd.DataFrame,
        top_k: int = 20
    ) -> List[str]:
        """
        Select top K most important features

        Args:
            importance_df: DataFrame from calculate_feature_importance
            top_k: Number of features to select

        Returns:
            List of top feature names
        """
        top_features = importance_df.head(top_k)['feature'].tolist()

        logger.info(
            "top_features_selected",
            k=top_k,
            features=top_features[:5]  # Log first 5
        )

        return top_features

    def create_feature_correlation_matrix(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Create feature correlation matrix

        Args:
            features: Feature matrix
            feature_names: Names of features

        Returns:
            Correlation matrix DataFrame
        """
        df = pd.DataFrame(features, columns=feature_names)
        corr_matrix = df.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        feature_names[i],
                        feature_names[j],
                        corr_matrix.iloc[i, j]
                    ))

        if high_corr_pairs:
            logger.warning(
                "high_correlations_detected",
                n_pairs=len(high_corr_pairs),
                examples=high_corr_pairs[:3]
            )

        return corr_matrix


# Utility functions

def normalize_features(
    features: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """
    Standalone function to normalize features

    Args:
        features: Feature matrix
        method: Normalization method

    Returns:
        Normalized features
    """
    engineer = FeatureEngineer(scaler_type=method)
    return engineer.fit_transform(features)


def calculate_feature_importance(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Standalone function to calculate feature importance

    Args:
        features: Feature matrix
        labels: Cluster labels
        feature_names: Feature names

    Returns:
        Importance DataFrame
    """
    engineer = FeatureEngineer()
    return engineer.calculate_feature_importance(features, labels, feature_names)


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print(f"Core features: {len(FeatureEngineer.CORE_FEATURES)}")
    print(f"Emotion features: {len(FeatureEngineer.EMOTION_FEATURES)}")
    print(f"Total features: {len(FeatureEngineer.CORE_FEATURES) + len(FeatureEngineer.EMOTION_FEATURES)}")
