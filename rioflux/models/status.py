from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举类，参考 Airflow 的状态定义"""

    # 任务尚未运行
    PENDING = "pending"

    # 任务已调度，等待运行
    SCHEDULED = "scheduled"

    # 任务正在排队
    QUEUED = "queued"

    # 任务正在运行
    RUNNING = "running"

    # 任务成功完成
    SUCCESS = "success"

    # 任务执行失败
    FAILED = "failed"

    # 任务被上游失败影响
    UPSTREAM_FAILED = "upstream_failed"

    # 任务被跳过
    SKIPPED = "skipped"

    # 任务被重试
    UP_FOR_RETRY = "up_for_retry"

    # 任务被重试但最终失败
    UP_FOR_RESCHEDULE = "up_for_reschedule"

    # 任务被删除
    REMOVED = "removed"

    @staticmethod
    def is_finished(status):
        """
        判断任务是否已结束
        """
        return status in [
            TaskStatus.SUCCESS,
            TaskStatus.FAILED,
            TaskStatus.UPSTREAM_FAILED,
            TaskStatus.SKIPPED,
            TaskStatus.REMOVED,
        ]
