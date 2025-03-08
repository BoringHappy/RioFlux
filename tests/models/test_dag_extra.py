import pytest
from typing import Dict, Any

from rioflux import DAG, BaseOperator, PythonOperator
from rioflux.models.status import TaskStatus


class SimpleOperator(BaseOperator):
    def __init__(self, task_id: str):
        super().__init__(task_id=task_id)
        self.executed = False
        self.context_data = None

    def execute(self, context: Dict[str, Any]) -> str:
        self.executed = True
        self.context_data = context
        return f"执行了 {self.task_id}"


def test_dag_validate_single_end_success():
    """测试DAG单一终止节点验证（成功情况）"""
    with DAG(dag_id="test_dag", validate_single_end=True, enable_auto_run=False) as dag:
        start = SimpleOperator(task_id="start")
        middle1 = SimpleOperator(task_id="middle1")
        middle2 = SimpleOperator(task_id="middle2")
        end = SimpleOperator(task_id="end")

        # 所有分支最终都指向同一个终止节点
        start >> [middle1, middle2] >> end

        # 验证不抛出异常
        dag._validate_single_end_node()


def test_dag_validate_single_end_multiple_ends():
    """测试DAG单一终止节点验证（多个终止节点）"""
    with DAG(dag_id="test_dag", validate_single_end=True, enable_auto_run=False) as dag:
        start = SimpleOperator(task_id="start")
        end1 = SimpleOperator(task_id="end1")
        end2 = SimpleOperator(task_id="end2")

        # 创建多个终止节点
        start >> end1
        start >> end2

        # 验证抛出预期的异常
        with pytest.raises(ValueError, match="有多个终止节点"):
            dag._validate_single_end_node()


def test_dag_validate_single_end_no_end():
    """测试DAG单一终止节点验证（无终止节点 - 循环依赖）"""
    with DAG(dag_id="test_dag", validate_single_end=True, enable_auto_run=False) as dag:
        # 创建任务但不设置依赖，所有任务都有上游和下游
        task1 = SimpleOperator(task_id="task1")
        task2 = SimpleOperator(task_id="task2")
        task3 = SimpleOperator(task_id="task3")

        # 创建循环依赖
        task1 >> task2 >> task3 >> task1

        # 验证抛出预期的异常
        with pytest.raises(ValueError, match="没有终止节点"):
            dag._validate_single_end_node()


def test_dag_validate_single_end_disabled():
    """测试禁用DAG单一终止节点验证"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        start = SimpleOperator(task_id="start")
        end1 = SimpleOperator(task_id="end1")
        end2 = SimpleOperator(task_id="end2")

        # 创建多个终止节点
        start >> end1
        start >> end2

        # 验证不抛出异常
        dag._validate_single_end_node()  # 应该没有异常，因为验证被禁用


def test_dag_auto_run():
    """测试DAG自动运行功能"""
    executed_tasks = []

    class TrackOperator(BaseOperator):
        def execute(self, context):
            executed_tasks.append(self.task_id)
            return self.task_id

    # 启用自动运行
    with DAG(dag_id="auto_run_dag", enable_auto_run=True) as dag:
        task1 = TrackOperator(task_id="task1")
        task2 = TrackOperator(task_id="task2")
        task1 >> task2

    # 退出with块后应自动运行
    assert executed_tasks == ["task1", "task2"]
    assert dag.task_status["task1"] == TaskStatus.SUCCESS
    assert dag.task_status["task2"] == TaskStatus.SUCCESS

    # 禁用自动运行
    executed_tasks.clear()
    with DAG(dag_id="no_auto_run_dag", enable_auto_run=False) as dag:
        task1 = TrackOperator(task_id="task1")
        task2 = TrackOperator(task_id="task2")
        task1 >> task2

    # 退出with块后不应运行
    assert executed_tasks == []
    assert dag.task_status["task1"] == TaskStatus.PENDING
    assert dag.task_status["task2"] == TaskStatus.PENDING


def test_dag_deadlock_detection():
    """测试DAG死锁检测"""
    with DAG(
        dag_id="deadlock_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:

        def task_func(context):
            return "执行成功"

        # 创建任务
        task1 = PythonOperator(task_id="task1", python_callable=task_func)
        task2 = PythonOperator(task_id="task2", python_callable=task_func)
        task3 = PythonOperator(task_id="task3", python_callable=task_func)

        # 设置依赖，确保有死锁
        # task2和task3互相依赖，形成死锁
        task1 >> task2
        task2 >> task3
        task3 >> task2  # 循环依赖

        # 运行DAG，应检测到死锁
        with pytest.raises(RuntimeError, match="检测到循环依赖或死锁"):
            dag.run()


def test_dag_context():
    """测试DAG上下文数据共享"""
    with DAG(dag_id="context_dag", enable_auto_run=False) as dag:

        def set_var_task(context):
            context.set_var("test_key", "测试值")
            return "设置变量"

        def use_var_task(context):
            value = context.get_var("test_key")
            return f"获取到的值: {value}"

        def use_result_task(context):
            result = context.get_task_result("set_var")
            value = context.get_var("test_key")
            return f"上个任务结果: {result}, 变量值: {value}"

        # 创建任务
        task1 = PythonOperator(task_id="set_var", python_callable=set_var_task)
        task2 = PythonOperator(task_id="use_var", python_callable=use_var_task)
        task3 = PythonOperator(task_id="use_result", python_callable=use_result_task)

        # 设置依赖
        task1 >> task2 >> task3

        # 运行DAG
        dag.run()

        # 验证任务结果和上下文
        assert dag.context.get_task_result("set_var") == "设置变量"
        assert dag.context.get_task_result("use_var") == "获取到的值: 测试值"
        assert (
            dag.context.get_task_result("use_result")
            == "上个任务结果: 设置变量, 变量值: 测试值"
        )
        assert dag.context.get_var("test_key") == "测试值"


def test_dag_already_finished():
    """测试已完成的DAG跳过执行"""
    execute_count = 0

    def counting_task(context):
        nonlocal execute_count
        execute_count += 1
        return execute_count

    # 创建DAG
    with DAG(dag_id="finished_dag", enable_auto_run=False) as dag:
        task = PythonOperator(task_id="task", python_callable=counting_task)

    # 第一次运行
    dag.run()
    assert execute_count == 1
    assert dag.dag_status == TaskStatus.SUCCESS

    # 第二次运行应该跳过
    dag.run()
    assert execute_count == 1  # 计数器没有增加

    # 重置状态后再次运行
    dag.dag_status = TaskStatus.PENDING
    dag.run()
    assert execute_count == 2  # 计数器增加
