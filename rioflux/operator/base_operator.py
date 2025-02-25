from abc import ABC, abstractmethod
from typing import Optional, Set


class BaseOperator(ABC):
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.upstream_task_ids: Set[str] = set()
        self.downstream_task_ids: Set[str] = set()
        self.dag: Optional["DAG"] = None
        self._init_in_dag()

    def _init_in_dag(self):
        from ..models.dag import DAG

        self.dag = DAG.get_current_dag()
        self.dag.add_task(self)

    def __rshift__(self, other: "BaseOperator") -> "BaseOperator":
        """实现 task1 >> task2 语法"""
        if isinstance(other, (list, tuple)):
            for task in other:
                self.dag.add_edge(self, task)
            return other
        else:
            self.dag.add_edge(self, other)
            return other

    def __rrshift__(self, other) -> "BaseOperator":
        """实现 [task1, task2] >> task3 语法"""
        if isinstance(other, (list, tuple)):
            for task in other:
                self.dag.add_edge(task, self)
            return self
        return NotImplemented

    def __lshift__(self, other: "BaseOperator") -> "BaseOperator":
        """实现 task1 << task2 语法"""
        if isinstance(other, (list, tuple)):
            for task in other:
                self.dag.add_edge(task, self)
            return other
        else:
            self.dag.add_edge(other, self)
            return other

    def __rlshift__(self, other) -> "BaseOperator":
        """实现 [task1, task2] << task3 语法"""
        if isinstance(other, (list, tuple)):
            for task in other:
                self.dag.add_edge(self, task)
            return self
        return NotImplemented

    @abstractmethod
    def execute(self, context: dict):
        """子类必须实现的具体任务逻辑"""
        pass
