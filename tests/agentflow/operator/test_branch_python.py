import pytest

from rioflux import DAG, BranchPythonOperator, PythonOperator
from rioflux.models.status import TaskStatus


def test_branch_python_operator_basic(dag):
    """测试基本的分支操作符功能"""

    def branch_func(context):
        return "path_a"

    def task_a(context):
        return "A 路径"

    def task_b(context):
        return "B 路径"

    with DAG(dag_id="test_dag", validate_single_end=False) as dag:
        # 创建任务
        branch_task = BranchPythonOperator(
            task_id="branch_task",
            python_callable=branch_func,
        )
        task_a = PythonOperator(
            task_id="path_a",
            python_callable=task_a,
        )
        task_b = PythonOperator(
            task_id="path_b",
            python_callable=task_b,
        )

        # 设置依赖关系
        branch_task >> [task_a, task_b]

        dag.run()
        # 验证任务状态和结果
        assert dag.task_status["branch_task"] == TaskStatus.SUCCESS
        assert dag.task_status["path_a"] == TaskStatus.SUCCESS
        assert dag.task_status["path_b"] == TaskStatus.SKIPPED
        assert dag.context.get_task_result("path_a") == "A 路径"


def test_branch_python_operator_with_condition_a(dag):
    """测试基于条件的分支操作符"""
    with DAG(dag_id="test_dag", validate_single_end=False) as dag:

        def branch_func(context):
            value = context.get_var("input_value")
            return "high_value" if value > 10 else "low_value"

        def process_high(context):
            return "高值处理"

        def process_low(context):
            return "低值处理"

        # 创建任务
        branch_task = BranchPythonOperator(
            task_id="branch_task",
            python_callable=branch_func,
        )
        high_task = PythonOperator(
            task_id="high_value",
            python_callable=process_high,
        )
        low_task = PythonOperator(
            task_id="low_value",
            python_callable=process_low,
        )

        # 设置依赖关系
        branch_task >> [high_task, low_task]

        # 测试高值路径
        dag.run(initial_context={"input_value": 15})
        assert dag.task_status["high_value"] == TaskStatus.SUCCESS
        assert dag.task_status["low_value"] == TaskStatus.SKIPPED
        assert dag.context.get_task_result("high_value") == "高值处理"


def test_branch_python_operator_with_condition_b(dag):
    """测试基于条件的分支操作符"""
    with DAG(dag_id="test_dag", validate_single_end=False) as dag:

        def branch_func(context):
            value = context.get_var("input_value")
            return "high_value" if value > 10 else "low_value"

        def process_high(context):
            return "高值处理"

        def process_low(context):
            return "低值处理"

        # 创建任务
        branch_task = BranchPythonOperator(
            task_id="branch_task",
            python_callable=branch_func,
        )
        high_task = PythonOperator(
            task_id="high_value",
            python_callable=process_high,
        )
        low_task = PythonOperator(
            task_id="low_value",
            python_callable=process_low,
        )

        # 设置依赖关系
        branch_task >> [high_task, low_task]

        # 测试高值路径
        dag.run(initial_context={"input_value": 5})
        assert dag.task_status["high_value"] == TaskStatus.SKIPPED
        assert dag.task_status["low_value"] == TaskStatus.SUCCESS
        assert dag.context.get_task_result("low_value") == "低值处理"


def test_branch_python_operator_invalid_path(dag):
    """测试无效分支路径的错误处理"""
    with DAG(dag_id="test_dag", validate_single_end=False) as dag:

        def branch_func(context):
            return "invalid_path"

        def task_a(context):
            return "A 路径"

        # 创建任务
        branch_task = BranchPythonOperator(
            task_id="branch_task",
            python_callable=branch_func,
        )
        task_a = PythonOperator(
            task_id="path_a",
            python_callable=task_a,
        )

        # 设置依赖关系
        branch_task >> task_a

        # 验证是否抛出预期的异常
        with pytest.raises(ValueError, match="不是有效的下游任务ID"):
            dag.run()

        # 验证任务状态
        assert dag.task_status["branch_task"] == TaskStatus.FAILED
        assert dag.dag_status == TaskStatus.FAILED


def test_branch_python_operator_complex_flow(dag):
    """测试复杂流程中的分支操作符"""
    with DAG(dag_id="test_dag", validate_single_end=False) as dag:

        def prepare_data(context):
            return {"value": 20}

        def branch_func(context):
            task_result = context.get_task_result("prepare")
            return "process_high" if task_result["value"] > 10 else "process_low"

        def process_high(context):
            return "高值处理"

        def process_low(context):
            return "低值处理"

        def finalize(context):
            if context.get_task_result("process_high"):
                result = context.get_task_result("process_high")
            else:
                result = context.get_task_result("process_low")
            return f"最终结果: {result}"

        # 创建任务
        prepare_task = PythonOperator(
            task_id="prepare",
            python_callable=prepare_data,
        )
        branch_task = BranchPythonOperator(
            task_id="branch_task",
            python_callable=branch_func,
        )
        high_task = PythonOperator(
            task_id="process_high",
            python_callable=process_high,
        )
        low_task = PythonOperator(
            task_id="process_low",
            python_callable=process_low,
        )
        final_task = PythonOperator(
            task_id="finalize",
            python_callable=finalize,
        )

        # 设置依赖关系
        prepare_task >> branch_task >> [high_task, low_task] >> final_task

        # 运行 DAG
        dag.run()

        # 验证任务状态和结果
        assert dag.task_status["prepare"] == TaskStatus.SUCCESS
        assert dag.task_status["branch_task"] == TaskStatus.SUCCESS
        assert dag.task_status["process_high"] == TaskStatus.SUCCESS
        assert dag.task_status["process_low"] == TaskStatus.SKIPPED
        assert dag.task_status["finalize"] == TaskStatus.SUCCESS
        assert dag.context.get_task_result("finalize") == "最终结果: 高值处理"
