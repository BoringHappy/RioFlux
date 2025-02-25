from typing import Any, Callable, Dict, Optional

from ..models.status import TaskStatus
from .base_operator import BaseOperator


class PythonOperator(BaseOperator):
    """
    Python 操作符：执行一个 Python 可调用对象

    用法示例：
    ```python
    def my_task_function(context):
        value = context.get_var('input_value')
        result = value * 2
        return result

    python_task = PythonOperator(
        task_id='process_task',
        python_callable=my_task_function
    )
    ```
    """

    def __init__(
        self,
        task_id: str,
        python_callable: Callable[[Dict[str, Any]], Any],
    ):
        """
        初始化 Python 操作符

        Args:
            task_id: 任务ID
            python_callable: Python 可调用对象，接收 context 参数
            dag: 可选的 DAG 实例
        """
        super().__init__(task_id=task_id)
        self.python_callable = python_callable

    def execute(self, context: Dict[str, Any]) -> Any:
        """
        执行 Python 可调用对象

        Args:
            context: DAG 上下文

        Returns:
            Any: Python 可调用对象的返回值
        """
        return self.python_callable(context)


class BranchPythonOperator(PythonOperator):
    """
    分支操作符：根据条件函数的返回值决定执行哪个下游任务

    用法示例：
    ```python
    def branch_func(context):
        value = context.get_var('some_value')
        return 'path_a' if value > 10 else 'path_b'

    branch_op = BranchOperator(
        task_id='branch_task',
        python_callable=branch_func
    )
    ```
    """

    def __init__(
        self,
        task_id: str,
        python_callable: Callable[[Dict[str, Any]], str],
    ):
        """
        初始化分支操作符

        Args:
            task_id: 任务ID
            python_callable: 分支条件函数，接收context参数，返回下游任务的task_id
            dag: 可选的DAG实例
        """
        super().__init__(task_id=task_id, python_callable=python_callable)

    def execute(self, context: Dict[str, Any]) -> str:
        """
        执行分支条件函数，返回选中的下游任务ID

        Args:
            context: DAG上下文

        Returns:
            str: 选中的下游任务ID
        """
        # 执行条件函数获取下游任务ID
        chosen_task_id = self.python_callable(context)

        # 验证返回的task_id是否是有效的下游任务
        if chosen_task_id not in self.downstream_task_ids:
            raise ValueError(
                f"分支条件函数返回的task_id '{chosen_task_id}'不是有效的下游任务ID。"
                f"有效的下游任务ID: {self.downstream_task_ids}"
            )

        # 将未选中的下游任务标记为跳过
        for task_id in self.downstream_task_ids:
            if task_id != chosen_task_id:
                self.dag.task_status[task_id] = TaskStatus.SKIPPED

        return chosen_task_id
