"""
KMeans Clustering Pipeline

Assigns users to 10 psychographic archetypes per topic using KMeans clustering.
Generates 64-dimensional embeddings from 27 psychological features.
Tracks cluster shifts over time.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
import structlog

from src.database.connection import get_db_session
from src.database.models import (
    UserMetrics, UserPsychProfile, ArchetypeMetadata, ClusterShift,
    ProcessingJob, Topic
)
from src.database.utils import (
    create_processing_job,
    update_job_status,
    get_topic_archetypes,
    update_archetype_stats
)

logger = structlog.get_logger(__name__)


class ClusteringPipeline:
    """
    Pipeline for clustering users into psychographic archetypes

    Workflow:
    1. Extract 27 features from UserMetrics
    2. Standardize features (z-score normalization)
    3. Reduce to 64 dimensions (PCA or selection)
    4. Run KMeans clustering (k=10)
    5. Assign cluster labels (P1-P10, C1-C10, etc.)
    6. Generate embeddings for similarity search
    7. Track cluster shifts for existing users
    8. Update archetype metadata with centroids
    """

    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        """
        Initialize clustering pipeline

        Args:
            n_clusters: Number of archetypes per topic (default: 10)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()

    def _extract_features(self, metrics: List[UserMetrics]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract 27 numerical features from UserMetrics

        Args:
            metrics: List of UserMetrics objects

        Returns:
            (feature_matrix, user_ids)
            - feature_matrix: (n_users, 27) numpy array
            - user_ids: List of user IDs in same order
        """
        feature_names = [
            # Emotion balance (2)
            'avg_valence', 'avg_arousal',

            # Volatility (4)
            'valence_stddev', 'arousal_stddev', 'emotion_stability', 'topic_volatility',

            # Toxicity/Hostility (2)
            'mean_toxicity', 'mean_hostility',

            # Engagement (2)
            'avg_engagement_score', 'reaction_bias',

            # Temporal patterns (1 - just peak hour)
            'peak_activity_hour',

            # Social influence (2)
            'influence_score', 'follower_log',

            # Cognitive complexity (3)
            'topic_diversity', 'entropy_score', 'avg_content_length',

            # Trust/Alignment (2)
            'trust_level', 'agreement_rate',

            # Post statistics (1)
            'post_count'

            # Emotion distribution (8) - extract from JSONB
            # Will be added as: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
        ]

        user_ids = []
        features = []

        for metric in metrics:
            user_ids.append(metric.user_id)

            # Extract scalar features
            feature_row = []

            for feat_name in feature_names:
                value = getattr(metric, feat_name, None)

                # Handle None values
                if value is None:
                    # Use sensible defaults
                    if feat_name == 'peak_activity_hour':
                        value = 12  # Noon
                    elif feat_name == 'post_count':
                        value = 0
                    else:
                        value = 0.0

                feature_row.append(float(value))

            # Extract emotion distribution (8 features from JSONB)
            emotion_dist = metric.emotion_distribution or {}
            for emotion in ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']:
                feature_row.append(float(emotion_dist.get(emotion, 0.0)))

            features.append(feature_row)

        return np.array(features), user_ids

    def _standardize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Standardize features using z-score normalization

        Args:
            features: (n_users, n_features) array

        Returns:
            Standardized features with mean=0, std=1
        """
        # Fit and transform
        standardized = self.scaler.fit_transform(features)

        logger.info(
            "features_standardized",
            n_samples=features.shape[0],
            n_features=features.shape[1],
            mean=float(np.mean(standardized)),
            std=float(np.std(standardized))
        )

        return standardized

    def _extract_sdk_vectors(self, metrics: List[Any]) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Extract SDK-native 9D emotional vectors from PostEnriched data.

        These vectors are ML-derived from DeBERTa in the Kaiko SDK, and may
        provide higher-quality clustering than the custom feature pipeline.

        Returns:
            (vector_matrix, user_ids) or (None, []) if insufficient data
        """
        user_ids = []
        vectors = []

        for metric in metrics:
            # Check if the user has posts with emotional_vector populated
            # For now, we work with what's available via UserMetrics
            user_ids.append(metric.user_id)

            # Build a composite vector from P0/P1 fields when SDK vector isn't directly available
            composite = [
                float(getattr(metric, 'avg_valence', 0) or 0),
                float(getattr(metric, 'avg_arousal', 0) or 0),
                float(getattr(metric, 'mean_toxicity', 0) or 0),
                float(getattr(metric, 'avg_safety_concern', 0) or 0),
                float(getattr(metric, 'avg_empathic_concern', 0) or 0),
                float(getattr(metric, 'avg_personal_distress', 0) or 0),
                float(getattr(metric, 'avg_meta_emotional_score', 0) or 0),
                float(getattr(metric, 'emotion_stability', 0) or 0),
                float(getattr(metric, 'entropy_score', 0) or 0),
            ]
            vectors.append(composite)

        if not vectors:
            return None, []

        return np.array(vectors), user_ids

    def _reduce_to_embedding_dim(
        self,
        features: np.ndarray,
        target_dim: int = 64,
        use_sdk_vectors: bool = False
    ) -> np.ndarray:
        """
        Reduce features to 64 dimensions for embedding storage.

        When use_sdk_vectors=True, uses the 9D SDK-native emotional vector
        as a compact representation, padded to target_dim. This is an
        alternative to PCA that leverages the SDK's DeBERTa-derived vectors.

        Args:
            features: (n_users, n_features) standardized features
            target_dim: Target dimension (default: 64 for pgvector)
            use_sdk_vectors: If True, prefer SDK 9D vectors over PCA

        Returns:
            (n_users, target_dim) embedding matrix
        """
        n_samples, n_features = features.shape

        if n_features == target_dim:
            return features

        elif n_features < target_dim:
            # Pad with zeros
            padding = np.zeros((n_samples, target_dim - n_features))
            embeddings = np.hstack([features, padding])

            logger.info(
                "features_padded",
                original_dim=n_features,
                target_dim=target_dim,
                sdk_vectors=use_sdk_vectors
            )

        else:
            # Use PCA to reduce
            from sklearn.decomposition import PCA

            pca = PCA(n_components=target_dim, random_state=self.random_state)
            embeddings = pca.fit_transform(features)

            logger.info(
                "features_reduced_pca",
                original_dim=n_features,
                target_dim=target_dim,
                explained_variance=float(np.sum(pca.explained_variance_ratio_))
            )

        return embeddings

    def _assign_cluster_labels(
        self,
        topic_name: str,
        cluster_ids: np.ndarray
    ) -> List[str]:
        """
        Convert cluster IDs (0-9) to archetype labels (P1-P10, C1-C10, etc.)

        Args:
            topic_name: Topic name (e.g., 'politics', 'crypto')
            cluster_ids: Array of cluster IDs (0-9)

        Returns:
            List of archetype labels (e.g., ['P1', 'P2', ...])
        """
        # Map topic to prefix
        topic_prefixes = {
            'politics': 'P',
            'crypto': 'C',
            'sports': 'S',
            'entertainment': 'E',
            'technology': 'T',
            'health': 'H',
            'education': 'ED',
            'economy': 'EC',
            'environment': 'EN',
            'social_issues': 'SI'
        }

        prefix = topic_prefixes.get(topic_name.lower(), 'G')  # Generic if not found

        # Convert cluster IDs to labels (1-indexed)
        labels = [f"{prefix}{cid + 1}" for cid in cluster_ids]

        return labels

    def _calculate_cluster_quality(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate clustering quality metrics

        Args:
            features: Standardized feature matrix
            labels: Cluster assignments

        Returns:
            Dict with quality metrics
        """
        quality = {}

        try:
            # Silhouette score (-1 to 1, higher is better)
            # Measures how similar an object is to its own cluster vs other clusters
            silhouette = silhouette_score(features, labels)
            quality['silhouette_score'] = float(silhouette)

        except Exception as e:
            logger.warning("silhouette_calculation_failed", error=str(e))
            quality['silhouette_score'] = 0.0

        try:
            # Calinski-Harabasz score (higher is better)
            # Ratio of between-cluster to within-cluster variance
            ch_score = calinski_harabasz_score(features, labels)
            quality['calinski_harabasz_score'] = float(ch_score)

        except Exception as e:
            logger.warning("calinski_harabasz_calculation_failed", error=str(e))
            quality['calinski_harabasz_score'] = 0.0

        # Inertia (lower is better)
        # Sum of squared distances to nearest cluster center
        # Already available from KMeans model

        return quality

    async def cluster_topic_users(
        self,
        topic_id: int,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """
        Cluster all users for a topic into 10 archetypes

        Args:
            topic_id: Topic to cluster
            period_start: Start of time window
            period_end: End of time window

        Returns:
            Dict with clustering results and stats
        """
        async with get_db_session() as db:
            # Create processing job
            job = await create_processing_job(
                db,
                job_type="cluster",
                status="running",
                topic_id=topic_id,
                period_start=period_start,
                period_end=period_end
            )
            job_id = job.id
            await db.commit()

            try:
                # Get topic info
                topic_query = select(Topic).where(Topic.id == topic_id)
                topic_result = await db.execute(topic_query)
                topic = topic_result.scalar_one_or_none()

                if not topic:
                    raise ValueError(f"Topic {topic_id} not found")

                topic_name = topic.topic_name

                # Get all user metrics for this topic in time window
                metrics_query = (
                    select(UserMetrics)
                    .where(
                        and_(
                            UserMetrics.topic_id == topic_id,
                            UserMetrics.period_start >= period_start,
                            UserMetrics.period_end <= period_end
                        )
                    )
                )

                result = await db.execute(metrics_query)
                metrics_list = result.scalars().all()

                if len(metrics_list) < self.n_clusters:
                    logger.warning(
                        "insufficient_users_for_clustering",
                        topic_id=topic_id,
                        user_count=len(metrics_list),
                        min_required=self.n_clusters
                    )
                    raise ValueError(f"Need at least {self.n_clusters} users, got {len(metrics_list)}")

                logger.info(
                    "clustering_topic",
                    topic_id=topic_id,
                    topic_name=topic_name,
                    user_count=len(metrics_list)
                )

                # Extract features
                features, user_ids = self._extract_features(metrics_list)

                # Standardize
                features_std = self._standardize_features(features)

                # Generate embeddings (64-dim)
                embeddings = self._reduce_to_embedding_dim(features_std, target_dim=64)

                # Run KMeans clustering
                kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300
                )

                cluster_ids = kmeans.fit_predict(features_std)

                # Convert to archetype labels
                archetype_labels = self._assign_cluster_labels(topic_name, cluster_ids)

                # Calculate quality metrics
                quality = self._calculate_cluster_quality(features_std, cluster_ids)
                quality['inertia'] = float(kmeans.inertia_)

                logger.info(
                    "clustering_complete",
                    topic_id=topic_id,
                    quality=quality
                )

                # Track cluster shifts and create/update profiles
                shifts_created = 0
                profiles_created = 0
                profiles_updated = 0

                for i, (user_id, cluster_id, archetype_label) in enumerate(
                    zip(user_ids, cluster_ids, archetype_labels)
                ):
                    # Get or create user psych profile
                    profile_query = (
                        select(UserPsychProfile)
                        .where(
                            and_(
                                UserPsychProfile.user_id == user_id,
                                UserPsychProfile.topic_id == topic_id
                            )
                        )
                    )

                    profile_result = await db.execute(profile_query)
                    existing_profile = profile_result.scalar_one_or_none()

                    # Check for cluster shift
                    if existing_profile and existing_profile.cluster_id != archetype_label:
                        # Track shift
                        shift = ClusterShift(
                            user_id=user_id,
                            topic_id=topic_id,
                            from_cluster_id=existing_profile.cluster_id,
                            to_cluster_id=archetype_label,
                            shift_timestamp=datetime.utcnow(),
                            confidence_delta=None  # Could calculate from cluster distances
                        )
                        db.add(shift)
                        shifts_created += 1

                    # Create or update profile
                    if existing_profile:
                        # Update
                        existing_profile.cluster_id = archetype_label
                        existing_profile.archetype_name = self._get_archetype_name(archetype_label)
                        existing_profile.embedding = embeddings[i].tolist()
                        existing_profile.cluster_confidence = None  # Could calculate from distances
                        existing_profile.last_updated = datetime.utcnow()
                        existing_profile.previous_cluster_id = existing_profile.cluster_id

                        profiles_updated += 1

                    else:
                        # Create new
                        profile = UserPsychProfile(
                            user_id=user_id,
                            topic_id=topic_id,
                            cluster_id=archetype_label,
                            archetype_name=self._get_archetype_name(archetype_label),
                            embedding=embeddings[i].tolist(),
                            cluster_confidence=None,
                            metrics=None,  # Could store aggregated metrics
                            assigned_at=datetime.utcnow(),
                            last_updated=datetime.utcnow()
                        )
                        db.add(profile)
                        profiles_created += 1

                # Commit profiles and shifts
                await db.commit()

                # Update archetype metadata with centroids
                for cluster_id in range(self.n_clusters):
                    archetype_label = f"{self._get_topic_prefix(topic_name)}{cluster_id + 1}"

                    # Get archetype metadata
                    arch_query = (
                        select(ArchetypeMetadata)
                        .where(
                            and_(
                                ArchetypeMetadata.topic_id == topic_id,
                                ArchetypeMetadata.cluster_id == archetype_label
                            )
                        )
                    )

                    arch_result = await db.execute(arch_query)
                    archetype = arch_result.scalar_one_or_none()

                    if archetype:
                        # Update centroid
                        centroid = kmeans.cluster_centers_[cluster_id]

                        # Pad or truncate to 64 dimensions
                        if len(centroid) < 64:
                            centroid_64 = np.pad(centroid, (0, 64 - len(centroid)))
                        else:
                            centroid_64 = centroid[:64]

                        archetype.centroid = centroid_64.tolist()

                        # Count members
                        member_count = int(np.sum(cluster_ids == cluster_id))
                        archetype.member_count = member_count

                        archetype.last_updated = datetime.utcnow()

                await db.commit()

                # Update job status
                await update_job_status(
                    db,
                    job_id=job_id,
                    status="completed",
                    records_processed=len(user_ids)
                )
                await db.commit()

                results = {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "n_users": len(user_ids),
                    "n_clusters": self.n_clusters,
                    "profiles_created": profiles_created,
                    "profiles_updated": profiles_updated,
                    "shifts_tracked": shifts_created,
                    "quality_metrics": quality
                }

                logger.info(
                    "topic_clustering_complete",
                    results=results
                )

                return results

            except Exception as e:
                logger.error(
                    "topic_clustering_failed",
                    topic_id=topic_id,
                    error=str(e)
                )

                await update_job_status(
                    db,
                    job_id=job_id,
                    status="failed",
                    error_message=str(e)
                )
                await db.commit()

                raise

    def _get_topic_prefix(self, topic_name: str) -> str:
        """Get topic prefix for archetype labels"""
        topic_prefixes = {
            'politics': 'P',
            'crypto': 'C',
            'sports': 'S',
            'entertainment': 'E',
            'technology': 'T',
            'health': 'H',
            'education': 'ED',
            'economy': 'EC',
            'environment': 'EN',
            'social_issues': 'SI'
        }
        return topic_prefixes.get(topic_name.lower(), 'G')

    def _get_archetype_name(self, cluster_label: str) -> str:
        """
        Get archetype name from cluster label

        This is a placeholder. In production, archetype names would be
        in the archetype_metadata table.

        Args:
            cluster_label: e.g., 'P1', 'C5'

        Returns:
            Archetype name
        """
        # Placeholder - would query database in production
        return f"Archetype {cluster_label}"

    async def cluster_all_topics(
        self,
        hours_back: int = 24
    ) -> Dict[int, Dict[str, Any]]:
        """
        Cluster users for all active topics

        Args:
            hours_back: Time window in hours

        Returns:
            Dict mapping topic_id -> clustering results
        """
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(hours=hours_back)

        results = {}

        async with get_db_session() as db:
            # Get all active topics
            query = select(Topic).where(Topic.is_active == True)
            result = await db.execute(query)
            topics = result.scalars().all()

            logger.info(
                "clustering_all_topics",
                topic_count=len(topics),
                period_start=period_start,
                period_end=period_end
            )

            # Cluster each topic
            for topic in topics:
                try:
                    topic_results = await self.cluster_topic_users(
                        topic_id=topic.id,
                        period_start=period_start,
                        period_end=period_end
                    )

                    results[topic.id] = topic_results

                except Exception as e:
                    logger.error(
                        "topic_clustering_error",
                        topic_id=topic.id,
                        error=str(e)
                    )
                    results[topic.id] = {"error": str(e)}

        logger.info(
            "all_topics_clustering_complete",
            topics_processed=len(results)
        )

        return results

    async def generate_umap_embeddings(
        self,
        topic_id: int,
        n_components: int = 2
    ) -> Dict[str, Any]:
        """
        Generate 2D UMAP embeddings for visualization

        Args:
            topic_id: Topic to generate embeddings for
            n_components: Number of dimensions (default: 2 for visualization)

        Returns:
            Dict with UMAP coordinates and metadata
        """
        try:
            import umap
        except ImportError:
            logger.error("umap_not_installed")
            raise ImportError("umap-learn not installed. Install with: pip install umap-learn")

        async with get_db_session() as db:
            # Get all user profiles for this topic
            query = (
                select(UserPsychProfile)
                .where(UserPsychProfile.topic_id == topic_id)
            )

            result = await db.execute(query)
            profiles = result.scalars().all()

            if len(profiles) < 10:
                logger.warning(
                    "insufficient_profiles_for_umap",
                    topic_id=topic_id,
                    profile_count=len(profiles)
                )
                return {"error": "Need at least 10 profiles for UMAP"}

            # Extract embeddings
            user_ids = [p.user_id for p in profiles]
            cluster_ids = [p.cluster_id for p in profiles]
            embeddings = np.array([p.embedding for p in profiles])

            logger.info(
                "generating_umap",
                topic_id=topic_id,
                n_profiles=len(profiles),
                embedding_dim=embeddings.shape[1]
            )

            # Generate UMAP projection
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.random_state,
                n_neighbors=min(15, len(profiles) - 1),
                min_dist=0.1,
                metric='cosine'
            )

            umap_coords = reducer.fit_transform(embeddings)

            logger.info(
                "umap_complete",
                topic_id=topic_id,
                explained_variance=None  # UMAP doesn't provide this
            )

            # Store UMAP coordinates back to database
            for profile, coords in zip(profiles, umap_coords):
                profile.umap_x = float(coords[0])
                profile.umap_y = float(coords[1])

            await db.commit()

            logger.info(
                "umap_coordinates_stored",
                topic_id=topic_id,
                profiles_updated=len(profiles),
            )

            results = {
                "topic_id": topic_id,
                "n_profiles": len(profiles),
                "umap_coordinates": umap_coords.tolist(),
                "user_ids": user_ids,
                "cluster_ids": cluster_ids,
                "n_components": n_components
            }

            return results


# Convenience function for testing
async def test_pipeline():
    """Test the clustering pipeline"""
    pipeline = ClusteringPipeline(n_clusters=10)

    # Cluster all topics for last 7 days
    results = await pipeline.cluster_all_topics(hours_back=168)

    for topic_id, stats in results.items():
        print(f"Topic {topic_id}: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pipeline())
