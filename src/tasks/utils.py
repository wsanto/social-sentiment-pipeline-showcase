"""
Task Utilities for Celery

Provides common utilities for Celery tasks including:
- Structured logging with task context
- Execution time tracking
- Error handling with context
"""

import functools
import time
from typing import Callable, Any
from datetime import datetime

from src.config.logging import get_logger

logger = get_logger(__name__)


def task_logger(task_name: str):
    """
    Decorator that adds structured logging to Celery tasks.

    Logs task start, completion, duration, and any errors.

    Usage:
        @app.task
        @task_logger("my_task")
        def my_task():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            task_id = kwargs.get('task_id', 'unknown')

            logger.info(
                f"task_started",
                extra={
                    "task_name": task_name,
                    "task_id": task_id,
                    "started_at": datetime.utcnow().isoformat()
                }
            )

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.info(
                    f"task_completed",
                    extra={
                        "task_name": task_name,
                        "task_id": task_id,
                        "duration_seconds": round(duration, 2),
                        "completed_at": datetime.utcnow().isoformat()
                    }
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                logger.error(
                    f"task_failed",
                    extra={
                        "task_name": task_name,
                        "task_id": task_id,
                        "duration_seconds": round(duration, 2),
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "failed_at": datetime.utcnow().isoformat()
                    },
                    exc_info=True
                )

                raise

        return wrapper
    return decorator


def log_task_progress(task_name: str, current: int, total: int, message: str = ""):
    """
    Log progress during long-running tasks.

    Args:
        task_name: Name of the task
        current: Current progress count
        total: Total items to process
        message: Optional progress message
    """
    percentage = (current / total * 100) if total > 0 else 0

    logger.info(
        f"task_progress",
        extra={
            "task_name": task_name,
            "current": current,
            "total": total,
            "percentage": round(percentage, 1),
            "message": message
        }
    )


class TaskMetrics:
    """
    Simple in-memory metrics collector for tasks.

    Tracks execution counts, durations, and error rates.
    Can be extended to push to Prometheus/DataDog later.
    """

    _metrics: dict = {}

    @classmethod
    def record_execution(cls, task_name: str, duration: float, success: bool):
        """Record a task execution."""
        if task_name not in cls._metrics:
            cls._metrics[task_name] = {
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "total_duration": 0.0,
                "last_execution": None,
                "last_status": None
            }

        metrics = cls._metrics[task_name]
        metrics["total_executions"] += 1
        metrics["total_duration"] += duration
        metrics["last_execution"] = datetime.utcnow().isoformat()

        if success:
            metrics["successful"] += 1
            metrics["last_status"] = "success"
        else:
            metrics["failed"] += 1
            metrics["last_status"] = "failed"

    @classmethod
    def get_metrics(cls, task_name: str = None) -> dict:
        """
        Get metrics for a task or all tasks.

        Args:
            task_name: Optional task name. If None, returns all metrics.

        Returns:
            Dict of metrics
        """
        if task_name:
            metrics = cls._metrics.get(task_name, {})
            if metrics:
                # Calculate averages
                avg_duration = metrics["total_duration"] / metrics["total_executions"]
                success_rate = metrics["successful"] / metrics["total_executions"] * 100
                return {
                    **metrics,
                    "average_duration": round(avg_duration, 2),
                    "success_rate": round(success_rate, 1)
                }
            return {}

        # Return all metrics with calculated fields
        result = {}
        for name, metrics in cls._metrics.items():
            if metrics["total_executions"] > 0:
                avg_duration = metrics["total_duration"] / metrics["total_executions"]
                success_rate = metrics["successful"] / metrics["total_executions"] * 100
                result[name] = {
                    **metrics,
                    "average_duration": round(avg_duration, 2),
                    "success_rate": round(success_rate, 1)
                }
        return result

    @classmethod
    def reset(cls):
        """Reset all metrics (useful for testing)."""
        cls._metrics = {}


def with_metrics(task_name: str):
    """
    Decorator that records task metrics.

    Usage:
        @app.task
        @with_metrics("my_task")
        def my_task():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            success = False

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            finally:
                duration = time.time() - start_time
                TaskMetrics.record_execution(task_name, duration, success)

        return wrapper
    return decorator
