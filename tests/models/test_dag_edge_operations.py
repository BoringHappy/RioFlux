import pytest
from rioflux import DAG, PythonOperator
from rioflux.models.status import TaskStatus
from unittest.mock import patch


@pytest.fixture
def simple_dag():
    """创建一个简单的测试 DAG"""
    with DAG(
        dag_id="test_edge_ops", validate_single_end=False, enable_auto_run=False
    ) as dag:
        task1 = PythonOperator(task_id="task1", python_callable=lambda context: "task1")
        task2 = PythonOperator(task_id="task2", python_callable=lambda context: "task2")
        task3 = PythonOperator(task_id="task3", python_callable=lambda context: "task3")

        # 创建初始依赖关系: task1 -> task2 -> task3
        task1 >> task2 >> task3

    return dag, task1, task2, task3


class TestEdgeOperations:
    """测试边操作相关功能"""

    def test_delete_edge_basic(self, simple_dag):
        """测试基本的边删除功能"""
        dag, task1, task2, task3 = simple_dag

        # 验证初始图结构
        assert (task1.task_id, task2.task_id) in dag.edges
        assert (task2.task_id, task3.task_id) in dag.edges
        assert task2.task_id in task1.downstream_task_ids
        assert task1.task_id in task2.upstream_task_ids

        # 删除边 task1 -> task2
        result = dag.delete_edge(task1, task2)

        # 验证删除成功
        assert result is True
        assert (task1.task_id, task2.task_id) not in dag.edges
        assert task2.task_id not in task1.downstream_task_ids
        assert task1.task_id not in task2.upstream_task_ids

        # 验证其他边没有受到影响
        assert (task2.task_id, task3.task_id) in dag.edges
        assert task3.task_id in task2.downstream_task_ids
        assert task2.task_id in task3.upstream_task_ids

    def test_delete_nonexistent_edge(self, simple_dag):
        """测试删除不存在的边"""
        dag, task1, task2, task3 = simple_dag

        # 尝试删除不存在的边 task1 -> task3
        result = dag.delete_edge(task1, task3)

        # 验证删除失败
        assert result is False

        # 验证原有的边没有受到影响
        assert (task1.task_id, task2.task_id) in dag.edges
        assert (task2.task_id, task3.task_id) in dag.edges

    def test_add_dynamic_task_with_edge_replacement(self, simple_dag):
        """测试带有边替换的动态任务添加"""
        dag, task1, task2, task3 = simple_dag

        # 初始确认直接边存在
        assert (task1.task_id, task3.task_id) not in dag.edges  # 初始没有直接连接

        # 添加一个直接边 task1 -> task3
        dag.add_edge(task1, task3)
        assert (task1.task_id, task3.task_id) in dag.edges  # 现在有直接连接

        # 创建一个新任务（不自动添加到DAG）
        with patch("rioflux.operator.base_operator.BaseOperator._init_in_dag"):
            task_middle = PythonOperator(
                task_id="task_middle", python_callable=lambda context: "middle"
            )
            task_middle.dag = None  # 确保不与任何DAG相关联

        # 使用 add_dynamic_task 添加任务，并替换冲突的边
        # 这应该会删除 task1 -> task3 的直接连接
        dag.add_dynamic_task(
            task=task_middle,
            upstream_tasks=[task1],
            downstream_tasks=[task3],
            replace_conflicting_edges=True,
        )

        # 验证新的任务已添加
        assert task_middle.task_id in dag.tasks

        # 验证新的边已添加
        assert (task1.task_id, task_middle.task_id) in dag.edges
        assert (task_middle.task_id, task3.task_id) in dag.edges

        # 验证冲突的边已被删除
        assert (task1.task_id, task3.task_id) not in dag.edges

        # 原来的 task1 -> task2 -> task3 路径应该保持不变
        assert (task1.task_id, task2.task_id) in dag.edges
        assert (task2.task_id, task3.task_id) in dag.edges

    def test_add_dynamic_task_without_edge_replacement(self, simple_dag):
        """测试不替换边的动态任务添加"""
        dag, task1, task2, task3 = simple_dag

        # 创建一个新任务（不自动添加到DAG）
        with patch("rioflux.operator.base_operator.BaseOperator._init_in_dag"):
            task_parallel = PythonOperator(
                task_id="task_parallel", python_callable=lambda context: "parallel"
            )
            task_parallel.dag = None  # 确保不与任何DAG相关联

        # 使用 add_dynamic_task 添加任务，但不替换冲突的边
        dag.add_dynamic_task(
            task=task_parallel,
            upstream_tasks=[task1],
            downstream_tasks=[task3],
            replace_conflicting_edges=False,
        )

        # 验证新的任务已添加
        assert task_parallel.task_id in dag.tasks

        # 验证新的边已添加
        assert (task1.task_id, task_parallel.task_id) in dag.edges
        assert (task_parallel.task_id, task3.task_id) in dag.edges

        # 验证原有的边没有被删除
        assert (task1.task_id, task2.task_id) in dag.edges
        assert (task2.task_id, task3.task_id) in dag.edges

    def test_delete_edge_during_execution(self):
        """测试在 DAG 执行过程中删除边"""
        # 创建一个用于测试的 DAG
        with DAG(
            dag_id="test_runtime_edge_ops",
            validate_single_end=False,
            enable_auto_run=False,
        ) as dag:
            # 使用计数器跟踪执行
            execution_count = {"task1": 0, "task2": 0, "task3": 0}

            def task1_fn(context):
                execution_count["task1"] += 1
                # 动态调整 DAG 结构：移除 task2 -> task3 的边
                nonlocal task2, task3
                dag.delete_edge(task2, task3)
                return "task1 done"

            def task2_fn(context):
                execution_count["task2"] += 1
                return "task2 done"

            def task3_fn(context):
                execution_count["task3"] += 1
                return "task3 done"

            task1 = PythonOperator(task_id="task1", python_callable=task1_fn)
            task2 = PythonOperator(task_id="task2", python_callable=task2_fn)
            task3 = PythonOperator(task_id="task3", python_callable=task3_fn)

            # 设置初始依赖: task1 -> task2 -> task3 和 task1 -> task3
            task1 >> task2 >> task3
            task1 >> task3

        # 运行 DAG
        dag.run()

        # 验证所有任务都执行了
        assert execution_count["task1"] == 1
        assert execution_count["task2"] == 1
        assert execution_count["task3"] == 1

        # 验证边已被删除
        assert (task2.task_id, task3.task_id) not in dag.edges
        assert task3.task_id not in task2.downstream_task_ids
        assert task2.task_id not in task3.upstream_task_ids

    def test_replace_edges_during_execution(self):
        """测试在 DAG 执行过程中替换边"""
        # 创建一个用于测试的 DAG
        with DAG(
            dag_id="test_runtime_replace_edges",
            validate_single_end=False,
            enable_auto_run=False,
        ) as dag:
            # 使用计数器跟踪执行
            execution_count = {"task1": 0, "task2": 0, "task3": 0, "task4": 0}

            def task1_fn(context):
                execution_count["task1"] += 1
                # 动态添加新任务 task4，替换 task2 -> task3 的边
                nonlocal task2, task3

                # 创建新任务（使用模拟阻止自动添加到DAG）
                with patch("rioflux.operator.base_operator.BaseOperator._init_in_dag"):
                    task4 = PythonOperator(
                        task_id="task4",
                        python_callable=lambda ctx: execution_count.update(
                            {"task4": execution_count["task4"] + 1}
                        )
                        or "task4 done",
                    )
                    task4.dag = None  # 确保不与任何DAG相关联

                # 添加任务，并替换 task2 -> task3 的边
                dag.add_dynamic_task(
                    task=task4,
                    upstream_tasks=[task2],
                    downstream_tasks=[task3],
                    replace_conflicting_edges=True,
                )

                return "task1 done"

            def task2_fn(context):
                execution_count["task2"] += 1
                return "task2 done"

            def task3_fn(context):
                execution_count["task3"] += 1
                return "task3 done"

            task1 = PythonOperator(task_id="task1", python_callable=task1_fn)
            task2 = PythonOperator(task_id="task2", python_callable=task2_fn)
            task3 = PythonOperator(task_id="task3", python_callable=task3_fn)

            # 设置初始依赖: task1 -> task2 -> task3
            task1 >> task2 >> task3

        # 运行 DAG
        dag.run()

        # 验证所有任务都执行了
        assert execution_count["task1"] == 1
        assert execution_count["task2"] == 1
        assert execution_count["task3"] == 1
        assert execution_count["task4"] == 1

        # 验证新的边结构
        assert (task2.task_id, "task4") in dag.edges
        assert ("task4", task3.task_id) in dag.edges
        assert (task2.task_id, task3.task_id) not in dag.edges
