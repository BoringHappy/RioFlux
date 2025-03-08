import pytest
from typing import Dict, Any

from rioflux import DAG, BaseOperator


# 创建一个简单的 BaseOperator 子类用于测试
class SimpleOperator(BaseOperator):
    def __init__(self, task_id: str):
        super().__init__(task_id=task_id)
        self.executed = False
        self.context_data = None

    def execute(self, context: Dict[str, Any]) -> str:
        self.executed = True
        self.context_data = context
        return f"执行了 {self.task_id}"


def test_base_operator_init(dag):
    """测试 BaseOperator 初始化和自动加入 DAG"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        task = SimpleOperator(task_id="test_task")

        # 验证任务已添加到 DAG
        assert "test_task" in dag.tasks
        assert dag.tasks["test_task"] == task
        assert task.dag == dag
        assert task.upstream_task_ids == set()
        assert task.downstream_task_ids == set()


def test_rshift_operator_single(dag):
    """测试 >> 操作符（单个任务）"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        task1 = SimpleOperator(task_id="task1")
        task2 = SimpleOperator(task_id="task2")

        # 使用操作符设置依赖关系
        result = task1 >> task2

        # 验证依赖关系
        assert "task1" in task2.upstream_task_ids
        assert "task2" in task1.downstream_task_ids
        assert (task1.task_id, task2.task_id) in dag.edges

        # 验证操作符返回值
        assert result == task2


def test_rshift_operator_multiple(dag):
    """测试 >> 操作符（多个任务）"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        task1 = SimpleOperator(task_id="task1")
        task2 = SimpleOperator(task_id="task2")
        task3 = SimpleOperator(task_id="task3")
        task4 = SimpleOperator(task_id="task4")

        # 使用操作符设置依赖关系，确保所有分支最终指向同一个终止节点
        result = task1 >> [task2, task3] >> task4

        # 验证依赖关系
        assert "task1" in task2.upstream_task_ids
        assert "task1" in task3.upstream_task_ids
        assert "task2" in task1.downstream_task_ids
        assert "task3" in task1.downstream_task_ids
        assert "task4" in task2.downstream_task_ids
        assert "task4" in task3.downstream_task_ids
        assert (task1.task_id, task2.task_id) in dag.edges
        assert (task1.task_id, task3.task_id) in dag.edges
        assert (task2.task_id, task4.task_id) in dag.edges
        assert (task3.task_id, task4.task_id) in dag.edges

        # 验证操作符返回值
        assert result == task4


def test_rrshift_operator(dag):
    """测试 [task1, task2] >> task3 语法"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        task1 = SimpleOperator(task_id="task1")
        task2 = SimpleOperator(task_id="task2")
        task3 = SimpleOperator(task_id="task3")

        # 使用操作符设置依赖关系
        result = [task1, task2] >> task3

        # 验证依赖关系
        assert "task1" in task3.upstream_task_ids
        assert "task2" in task3.upstream_task_ids
        assert "task3" in task1.downstream_task_ids
        assert "task3" in task2.downstream_task_ids
        assert (task1.task_id, task3.task_id) in dag.edges
        assert (task2.task_id, task3.task_id) in dag.edges

        # 验证操作符返回值
        assert result == task3


def test_lshift_operator_single(dag):
    """测试 << 操作符（单个任务）"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        task1 = SimpleOperator(task_id="task1")
        task2 = SimpleOperator(task_id="task2")

        # 使用操作符设置依赖关系
        result = task1 << task2

        # 验证依赖关系
        assert "task2" in task1.upstream_task_ids
        assert "task1" in task2.downstream_task_ids
        assert (task2.task_id, task1.task_id) in dag.edges

        # 验证操作符返回值
        assert result == task2


def test_lshift_operator_multiple(dag):
    """测试 << 操作符（多个任务）"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        task1 = SimpleOperator(task_id="task1")
        task2 = SimpleOperator(task_id="task2")
        task3 = SimpleOperator(task_id="task3")

        # 使用操作符设置依赖关系
        result = task1 << [task2, task3]

        # 验证依赖关系
        assert "task2" in task1.upstream_task_ids
        assert "task3" in task1.upstream_task_ids
        assert "task1" in task2.downstream_task_ids
        assert "task1" in task3.downstream_task_ids
        assert (task2.task_id, task1.task_id) in dag.edges
        assert (task3.task_id, task1.task_id) in dag.edges

        # 验证操作符返回值
        assert result == [task2, task3]


def test_rlshift_operator(dag):
    """测试 [task1, task2] << task3 语法"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        task1 = SimpleOperator(task_id="task1")
        task2 = SimpleOperator(task_id="task2")
        task3 = SimpleOperator(task_id="task3")
        task4 = SimpleOperator(task_id="task4")

        # 设置依赖关系，确保有单一终止节点
        [task1, task2] << task3 >> task4

        # 验证依赖关系
        assert "task3" in task1.upstream_task_ids
        assert "task3" in task2.upstream_task_ids
        assert "task1" in task3.downstream_task_ids
        assert "task2" in task3.downstream_task_ids
        assert "task4" in task3.downstream_task_ids
        assert (task3.task_id, task1.task_id) in dag.edges
        assert (task3.task_id, task2.task_id) in dag.edges
        assert (task3.task_id, task4.task_id) in dag.edges

        # 验证操作符返回值为task3
        # 注意：这里无法直接验证[task1, task2] << task3的返回值，
        # 因为我们在测试中添加了额外的task3 >> task4操作
