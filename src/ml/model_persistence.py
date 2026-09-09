"""
Model Persistence and Versioning Module

Provides utilities for saving, loading, and versioning ML models.
Supports model registry, performance tracking, and A/B testing framework.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import joblib
import json
import hashlib
import shutil
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from src.database.models import ModelRegistry

logger = structlog.get_logger(__name__)


class ModelPersistence:
    """
    Model persistence and versioning system

    Provides:
    - Model serialization (joblib)
    - Model versioning with git-like hashing
    - Model registry in database
    - Performance tracking
    - A/B testing support
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize model persistence

        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info("model_persistence_initialized", models_dir=str(self.models_dir))

    def _generate_model_hash(self, model: Any) -> str:
        """
        Generate unique hash for model

        Uses joblib serialization + SHA256 hashing

        Args:
            model: Model object to hash

        Returns:
            SHA256 hash string
        """
        # Serialize model to bytes
        model_bytes = joblib.dumps(model)

        # Generate hash
        hash_obj = hashlib.sha256(model_bytes)
        model_hash = hash_obj.hexdigest()[:16]  # First 16 chars

        return model_hash

    def save_model(
        self,
        model: Any,
        model_type: str,
        topic_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model to disk

        Args:
            model: Model object (sklearn model, scaler, etc.)
            model_type: Type of model ('kmeans', 'scaler', 'pca', 'umap')
            topic_id: Optional topic ID for topic-specific models
            metadata: Optional metadata to save with model

        Returns:
            Model file path
        """
        # Generate unique hash
        model_hash = self._generate_model_hash(model)

        # Create filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if topic_id:
            filename = f"{model_type}_topic{topic_id}_{timestamp}_{model_hash}.joblib"
        else:
            filename = f"{model_type}_{timestamp}_{model_hash}.joblib"

        filepath = self.models_dir / filename

        # Save model
        joblib.dump(model, filepath)

        # Save metadata if provided
        if metadata:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(
            "model_saved",
            model_type=model_type,
            topic_id=topic_id,
            filepath=str(filepath),
            model_hash=model_hash
        )

        return str(filepath)

    def load_model(self, filepath: str) -> Any:
        """
        Load model from disk

        Args:
            filepath: Path to model file

        Returns:
            Loaded model object
        """
        model = joblib.load(filepath)

        logger.info("model_loaded", filepath=filepath)

        return model

    def load_metadata(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load model metadata

        Args:
            filepath: Path to model file

        Returns:
            Metadata dict or None
        """
        metadata_path = Path(filepath).with_suffix('.json')

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata

    async def register_model(
        self,
        db: AsyncSession,
        model_type: str,
        version: str,
        filepath: str,
        topic_id: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = False
    ) -> ModelRegistry:
        """
        Register model in database

        Args:
            db: Database session
            model_type: Type of model
            version: Version string (e.g., "v1.0.0")
            filepath: Path to model file
            topic_id: Optional topic ID
            metrics: Performance metrics
            metadata: Additional metadata
            is_active: Whether this is the active model

        Returns:
            ModelRegistry record
        """
        # If marking as active, deactivate other models of same type/topic
        if is_active:
            await self._deactivate_models(db, model_type, topic_id)

        # Create registry entry
        registry = ModelRegistry(
            model_type=model_type,
            version=version,
            filepath=filepath,
            topic_id=topic_id,
            metrics=metrics or {},
            metadata=metadata or {},
            is_active=is_active,
            created_at=datetime.utcnow()
        )

        db.add(registry)
        await db.commit()
        await db.refresh(registry)

        logger.info(
            "model_registered",
            model_id=registry.id,
            model_type=model_type,
            version=version,
            is_active=is_active
        )

        return registry

    async def _deactivate_models(
        self,
        db: AsyncSession,
        model_type: str,
        topic_id: Optional[int]
    ):
        """Deactivate all models of given type and topic"""
        query = select(ModelRegistry).where(
            ModelRegistry.model_type == model_type,
            ModelRegistry.is_active == True
        )

        if topic_id:
            query = query.where(ModelRegistry.topic_id == topic_id)

        result = await db.execute(query)
        models = result.scalars().all()

        for model in models:
            model.is_active = False

        await db.commit()

        logger.info(
            "models_deactivated",
            model_type=model_type,
            topic_id=topic_id,
            count=len(models)
        )

    async def get_active_model(
        self,
        db: AsyncSession,
        model_type: str,
        topic_id: Optional[int] = None
    ) -> Optional[ModelRegistry]:
        """
        Get active model from registry

        Args:
            db: Database session
            model_type: Type of model
            topic_id: Optional topic ID

        Returns:
            ModelRegistry record or None
        """
        query = select(ModelRegistry).where(
            ModelRegistry.model_type == model_type,
            ModelRegistry.is_active == True
        )

        if topic_id:
            query = query.where(ModelRegistry.topic_id == topic_id)

        result = await db.execute(query)
        model = result.scalar_one_or_none()

        return model

    async def get_model_history(
        self,
        db: AsyncSession,
        model_type: str,
        topic_id: Optional[int] = None,
        limit: int = 10
    ) -> List[ModelRegistry]:
        """
        Get model version history

        Args:
            db: Database session
            model_type: Type of model
            topic_id: Optional topic ID
            limit: Number of versions to return

        Returns:
            List of ModelRegistry records
        """
        query = select(ModelRegistry).where(
            ModelRegistry.model_type == model_type
        )

        if topic_id:
            query = query.where(ModelRegistry.topic_id == topic_id)

        query = query.order_by(desc(ModelRegistry.created_at)).limit(limit)

        result = await db.execute(query)
        models = result.scalars().all()

        return list(models)

    async def activate_model(
        self,
        db: AsyncSession,
        model_id: int
    ) -> ModelRegistry:
        """
        Activate a specific model version

        Args:
            db: Database session
            model_id: Model registry ID

        Returns:
            Activated ModelRegistry record
        """
        # Get model
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalar_one()

        # Deactivate others
        await self._deactivate_models(db, model.model_type, model.topic_id)

        # Activate this one
        model.is_active = True
        await db.commit()
        await db.refresh(model)

        logger.info(
            "model_activated",
            model_id=model_id,
            model_type=model.model_type,
            version=model.version
        )

        return model

    async def update_model_metrics(
        self,
        db: AsyncSession,
        model_id: int,
        metrics: Dict[str, float]
    ) -> ModelRegistry:
        """
        Update model performance metrics

        Args:
            db: Database session
            model_id: Model registry ID
            metrics: Performance metrics to update

        Returns:
            Updated ModelRegistry record
        """
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalar_one()

        # Merge metrics
        model.metrics.update(metrics)

        await db.commit()
        await db.refresh(model)

        logger.info(
            "model_metrics_updated",
            model_id=model_id,
            metrics=metrics
        )

        return model

    async def compare_models(
        self,
        db: AsyncSession,
        model_id_a: int,
        model_id_b: int,
        metric_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare two model versions

        Args:
            db: Database session
            model_id_a: First model ID
            model_id_b: Second model ID
            metric_keys: Optional list of specific metrics to compare

        Returns:
            Comparison results
        """
        # Get models
        result_a = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id_a)
        )
        model_a = result_a.scalar_one()

        result_b = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id_b)
        )
        model_b = result_b.scalar_one()

        # Extract metrics
        metrics_a = model_a.metrics
        metrics_b = model_b.metrics

        # Compare
        if metric_keys:
            keys_to_compare = metric_keys
        else:
            # All common keys
            keys_to_compare = set(metrics_a.keys()) & set(metrics_b.keys())

        comparison = {
            'model_a': {
                'id': model_a.id,
                'version': model_a.version,
                'created_at': model_a.created_at.isoformat()
            },
            'model_b': {
                'id': model_b.id,
                'version': model_b.version,
                'created_at': model_b.created_at.isoformat()
            },
            'metrics': {}
        }

        for key in keys_to_compare:
            val_a = metrics_a.get(key, 0.0)
            val_b = metrics_b.get(key, 0.0)
            difference = val_b - val_a
            percent_change = (difference / val_a * 100) if val_a != 0 else 0.0

            comparison['metrics'][key] = {
                'model_a': val_a,
                'model_b': val_b,
                'difference': difference,
                'percent_change': percent_change
            }

        logger.info(
            "models_compared",
            model_a_id=model_id_a,
            model_b_id=model_id_b,
            metrics_compared=len(comparison['metrics'])
        )

        return comparison

    async def delete_model(
        self,
        db: AsyncSession,
        model_id: int,
        delete_file: bool = True
    ):
        """
        Delete model from registry and optionally from disk

        Args:
            db: Database session
            model_id: Model registry ID
            delete_file: Whether to delete model file
        """
        # Get model
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalar_one()

        filepath = model.filepath

        # Delete from database
        await db.delete(model)
        await db.commit()

        # Delete files
        if delete_file:
            model_path = Path(filepath)
            if model_path.exists():
                model_path.unlink()

            # Delete metadata
            metadata_path = model_path.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()

        logger.info(
            "model_deleted",
            model_id=model_id,
            filepath=filepath,
            file_deleted=delete_file
        )

    def cleanup_old_models(self, keep_versions: int = 5):
        """
        Clean up old model files (keep only N most recent per type)

        Args:
            keep_versions: Number of versions to keep per model type
        """
        # Group model files by type
        model_files = {}

        for filepath in self.models_dir.glob("*.joblib"):
            # Extract model type from filename
            parts = filepath.stem.split('_')
            model_type = parts[0]

            if model_type not in model_files:
                model_files[model_type] = []

            model_files[model_type].append(filepath)

        # Sort by modification time and delete old ones
        deleted_count = 0

        for model_type, files in model_files.items():
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Delete old versions
            for filepath in files[keep_versions:]:
                filepath.unlink()

                # Delete metadata
                metadata_path = filepath.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()

                deleted_count += 1

        logger.info(
            "old_models_cleaned",
            keep_versions=keep_versions,
            deleted_count=deleted_count
        )


class ModelVersioning:
    """
    Model versioning utilities

    Provides semantic versioning and automatic version increments
    """

    @staticmethod
    def parse_version(version: str) -> Tuple[int, int, int]:
        """
        Parse semantic version string

        Args:
            version: Version string (e.g., "v1.2.3")

        Returns:
            (major, minor, patch) tuple
        """
        # Remove 'v' prefix if present
        version = version.lstrip('v')

        parts = version.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        return (major, minor, patch)

    @staticmethod
    def format_version(major: int, minor: int, patch: int) -> str:
        """
        Format version tuple as string

        Args:
            major: Major version
            minor: Minor version
            patch: Patch version

        Returns:
            Version string (e.g., "v1.2.3")
        """
        return f"v{major}.{minor}.{patch}"

    @staticmethod
    def increment_version(
        version: str,
        level: str = 'patch'
    ) -> str:
        """
        Increment version number

        Args:
            version: Current version string
            level: Level to increment ('major', 'minor', 'patch')

        Returns:
            New version string
        """
        major, minor, patch = ModelVersioning.parse_version(version)

        if level == 'major':
            major += 1
            minor = 0
            patch = 0
        elif level == 'minor':
            minor += 1
            patch = 0
        elif level == 'patch':
            patch += 1
        else:
            raise ValueError(f"Invalid level: {level}")

        return ModelVersioning.format_version(major, minor, patch)

    @staticmethod
    async def get_next_version(
        db: AsyncSession,
        model_type: str,
        topic_id: Optional[int] = None,
        level: str = 'patch'
    ) -> str:
        """
        Get next version number for model type

        Args:
            db: Database session
            model_type: Type of model
            topic_id: Optional topic ID
            level: Version level to increment

        Returns:
            Next version string
        """
        # Get latest version
        query = select(ModelRegistry).where(
            ModelRegistry.model_type == model_type
        )

        if topic_id:
            query = query.where(ModelRegistry.topic_id == topic_id)

        query = query.order_by(desc(ModelRegistry.created_at)).limit(1)

        result = await db.execute(query)
        latest_model = result.scalar_one_or_none()

        if latest_model:
            current_version = latest_model.version
            next_version = ModelVersioning.increment_version(current_version, level)
        else:
            # No existing versions, start at v1.0.0
            next_version = "v1.0.0"

        return next_version


# Utility functions

def save_clustering_models(
    persistence: ModelPersistence,
    topic_id: int,
    kmeans_model: Any,
    scaler: Any,
    pca_model: Optional[Any] = None,
    umap_model: Optional[Any] = None
) -> Dict[str, str]:
    """
    Save all clustering models for a topic

    Args:
        persistence: ModelPersistence instance
        topic_id: Topic ID
        kmeans_model: KMeans model
        scaler: StandardScaler
        pca_model: Optional PCA model
        umap_model: Optional UMAP model

    Returns:
        Dict mapping model type to filepath
    """
    filepaths = {}

    # Save KMeans
    filepaths['kmeans'] = persistence.save_model(
        kmeans_model,
        model_type='kmeans',
        topic_id=topic_id,
        metadata={'n_clusters': kmeans_model.n_clusters}
    )

    # Save scaler
    filepaths['scaler'] = persistence.save_model(
        scaler,
        model_type='scaler',
        topic_id=topic_id
    )

    # Save PCA if provided
    if pca_model:
        filepaths['pca'] = persistence.save_model(
            pca_model,
            model_type='pca',
            topic_id=topic_id,
            metadata={'n_components': pca_model.n_components}
        )

    # Save UMAP if provided
    if umap_model:
        filepaths['umap'] = persistence.save_model(
            umap_model,
            model_type='umap',
            topic_id=topic_id
        )

    logger.info(
        "clustering_models_saved",
        topic_id=topic_id,
        models_saved=list(filepaths.keys())
    )

    return filepaths


def load_clustering_models(
    persistence: ModelPersistence,
    filepaths: Dict[str, str]
) -> Dict[str, Any]:
    """
    Load all clustering models

    Args:
        persistence: ModelPersistence instance
        filepaths: Dict mapping model type to filepath

    Returns:
        Dict mapping model type to loaded model
    """
    models = {}

    for model_type, filepath in filepaths.items():
        models[model_type] = persistence.load_model(filepath)

    logger.info(
        "clustering_models_loaded",
        models_loaded=list(models.keys())
    )

    return models


if __name__ == "__main__":
    # Example usage
    print("Model Persistence Module")

    persistence = ModelPersistence(models_dir="models")
    print(f"Models directory: {persistence.models_dir}")

    versioning = ModelVersioning()
    print(f"Next version after v1.2.3 (patch): {versioning.increment_version('v1.2.3', 'patch')}")
    print(f"Next version after v1.2.3 (minor): {versioning.increment_version('v1.2.3', 'minor')}")
    print(f"Next version after v1.2.3 (major): {versioning.increment_version('v1.2.3', 'major')}")
