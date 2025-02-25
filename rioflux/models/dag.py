import threading
from typing import Any, Dict, List, Optional, Set

from .status import TaskStatus


class DAG:
    _thread_local = threading.local()

    def __init__(
        self,
        dag_id: str,
        validate_single_end: bool = True,
        enable_auto_run: bool = True,
    ):
        self.dag_id = dag_id
        self.tasks: Dict[str, "BaseOperator"] = {}
        self.edges: Set[tuple] = set()
        self.context = DAGContext()
        self.dag_status: TaskStatus = TaskStatus.PENDING
        self.task_status: Dict[str, TaskStatus] = {}

        self.validate_single_end = validate_single_end
        self.enable_auto_run = enable_auto_run

    def _validate_single_end_node(self) -> None:
        """验证 DAG 是否只有一个终止节点"""
        if not self.validate_single_end:
            return

        end_nodes = [
            task_id
            for task_id in self.tasks
            if not self.tasks[task_id].downstream_task_ids
        ]

        if len(end_nodes) == 0:
            raise ValueError(f"DAG {self.dag_id} 没有终止节点")
        elif len(end_nodes) > 1:
            raise ValueError(
                f"DAG {self.dag_id} 有多个终止节点: {end_nodes}，"
                f"请确保所有分支最终都指向同一个终止节点"
            )

    def add_task(self, task: "BaseOperator"):
        if task.task_id in self.tasks:
            raise ValueError(f"任务 {task.task_id} 已存在于 DAG {self.dag_id} 中")
        self.tasks[task.task_id] = task
        self.task_status[task.task_id] = TaskStatus.PENDING
        task.dag = self

    def add_edge(self, upstream_task: "BaseOperator", downstream_task: "BaseOperator"):
        self.edges.add((upstream_task.task_id, downstream_task.task_id))
        upstream_task.downstream_task_ids.add(downstream_task.task_id)
        downstream_task.upstream_task_ids.add(upstream_task.task_id)

    def _get_ready_tasks(self) -> List[str]:
        """获取所有依赖已满足的任务"""
        ready_tasks = []
        for task_id in self.tasks:
            if self.task_status[task_id] != TaskStatus.PENDING:
                continue

            task = self.tasks[task_id]
            if all(
                self.task_status[upstream_id] == TaskStatus.SUCCESS
                or self.task_status[upstream_id] == TaskStatus.SKIPPED
                for upstream_id in task.upstream_task_ids
            ):
                ready_tasks.append(task_id)
        return ready_tasks

    def run(self, initial_context: Dict[str, Any] = None) -> None:
        """执行 DAG"""

        if TaskStatus.is_finished(self.dag_status):
            print(f"DAG {self.dag_id} 已完成，跳过执行")
            return
        else:
            self.dag_status = TaskStatus.PENDING

        # 验证终止节点
        self._validate_single_end_node()
        # 重置所有任务状态
        self.task_status = {task_id: TaskStatus.PENDING for task_id in self.tasks}
        self.context.clear()

        # 初始化上下文
        if initial_context:
            for key, value in initial_context.items():
                self.context.set_var(key, value)

        while any(status == TaskStatus.PENDING for status in self.task_status.values()):
            ready_tasks = self._get_ready_tasks()
            if not ready_tasks:
                # 检查是否所有未完成的任务都是 TODO 状态
                unfinished_tasks = [
                    task_id
                    for task_id, status in self.task_status.items()
                    if not TaskStatus.is_finished(status)
                ]
                if unfinished_tasks:
                    raise RuntimeError(
                        f"检测到循环依赖或死锁，未完成的任务: {unfinished_tasks}"
                    )
                break

            for task_id in ready_tasks:
                task = self.tasks[task_id]
                try:
                    print(f"执行任务: {task_id}")
                    # 执行任务并传递上下文
                    result = task.execute(context=self.context)
                    # 存储任务结果
                    self.context.set_task_result(task_id, result)
                    self.task_status[task_id] = TaskStatus.SUCCESS
                except Exception as e:
                    self.task_status[task_id] = TaskStatus.FAILED
                    self.dag_status = TaskStatus.FAILED
                    print(f"任务 {task_id} 失败: {e}")
                    raise
        self.dag_status = TaskStatus.SUCCESS

    def __enter__(self):
        if not hasattr(self._thread_local, "active_dag_stack"):
            self._thread_local.active_dag_stack = []
        self._thread_local.active_dag_stack.append(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dag_status == TaskStatus.PENDING and self.enable_auto_run:
            self.run()
        self._thread_local.active_dag_stack.pop()

    @classmethod
    def get_current_dag(cls) -> Optional["DAG"]:
        if (
            hasattr(cls._thread_local, "active_dag_stack")
            and cls._thread_local.active_dag_stack
        ):
            return cls._thread_local.active_dag_stack[-1]
        return None


class DAGContext:
    """DAG 上下文：用于在任务之间共享数据"""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._task_results: Dict[str, Any] = {}

    def set_var(self, key: str, value: Any) -> None:
        """设置上下文变量"""
        self._data[key] = value

    def get_var(self, key: str, default: Any = None) -> Any:
        """获取上下文变量"""
        return self._data.get(key, default)

    def set_task_result(self, task_id: str, result: Any) -> None:
        """存储任务执行结果"""
        self._task_results[task_id] = result

    def get_task_result(self, task_id: str, default: Any = None) -> Any:
        """获取任务执行结果"""
        return self._task_results.get(task_id, default)

    def clear(self) -> None:
        """清空上下文数据"""
        self._data.clear()
        self._task_results.clear()
