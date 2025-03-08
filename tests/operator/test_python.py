import pytest

from rioflux import DAG, PythonOperator
from rioflux.models.status import TaskStatus


def test_python_operator_basic(dag):
    """测试基本的 Python 操作符功能"""
    with DAG(dag_id="test_dag") as dag:

        def task_function(context):
            return "测试成功"

        task = PythonOperator(
            task_id="test_task",
            python_callable=task_function,
        )

        # 运行 DAG
        dag.run()

        # 验证任务状态和结果
        assert dag.task_status[task.task_id] == TaskStatus.SUCCESS
        assert dag.context.get_task_result(task.task_id) == "测试成功"


def test_python_operator_with_context(dag):
    """测试 Python 操作符与上下文交互"""
    with DAG(dag_id="test_dag") as dag:

        def task_function(context):
            # 从上下文获取数据
            input_value = context.get_var("input_value")
            # 处理数据
            result = input_value * 2
            # 将结果存入上下文
            context.set_var("output_value", result)
            return result

        task = PythonOperator(
            task_id="test_task",
            python_callable=task_function,
        )

        # 设置初始上下文
        initial_context = {"input_value": 5}

        # 运行 DAG
        dag.run(initial_context=initial_context)

        # 验证任务状态和结果
        assert dag.task_status[task.task_id] == TaskStatus.SUCCESS
        assert dag.context.get_task_result(task.task_id) == 10
        assert dag.context.get_var("output_value") == 10


def test_python_operator_error_handling(dag):
    """测试 Python 操作符的错误处理"""
    with DAG(dag_id="test_dag") as dag:

        def task_function(context):
            raise ValueError("测试错误")

        task = PythonOperator(
            task_id="test_task",
            python_callable=task_function,
        )

        # 运行 DAG 并验证是否抛出预期的异常
        with pytest.raises(ValueError, match="测试错误"):
            dag.run()

        # 验证任务状态
        assert dag.task_status[task.task_id] == TaskStatus.FAILED
        assert dag.dag_status == TaskStatus.FAILED


def test_python_operator_multiple_tasks(dag):
    """测试多个 Python 操作符的依赖关系"""
    with DAG(dag_id="test_dag") as dag:

        def task1_function(context):
            return 5

        def task2_function(context):
            task1_result = context.get_task_result("task1")
            return task1_result * 2

        def task3_function(context):
            task2_result = context.get_task_result("task2")
            return task2_result + 1

        # 创建任务
        task1 = PythonOperator(
            task_id="task1",
            python_callable=task1_function,
        )
        task2 = PythonOperator(
            task_id="task2",
            python_callable=task2_function,
        )
        task3 = PythonOperator(
            task_id="task3",
            python_callable=task3_function,
        )

        # 设置依赖关系
        task1 >> task2 >> task3

        # 运行 DAG
        dag.run()

        # 验证任务状态和结果
        assert dag.task_status["task1"] == TaskStatus.SUCCESS
        assert dag.task_status["task2"] == TaskStatus.SUCCESS
        assert dag.task_status["task3"] == TaskStatus.SUCCESS
        assert dag.context.get_task_result("task1") == 5
        assert dag.context.get_task_result("task2") == 10
        assert dag.context.get_task_result("task3") == 11
