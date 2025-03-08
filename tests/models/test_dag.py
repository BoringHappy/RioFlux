import pytest
from rioflux import DAG, PythonOperator, BranchPythonOperator
from rioflux.models.status import TaskStatus


class TestDAGMermaid:
    """测试 DAG 的 mermaid 图表生成功能"""

    def test_empty_dag_mermaid(self, dag):
        """测试空 DAG 的 mermaid 生成"""
        # 使用传入的 dag fixture
        mermaid = dag.to_mermaid()
        assert "graph TD;" in mermaid
        # 空 DAG 应该没有节点和边
        assert len(mermaid.split("\n")) == 5  # 图定义 + 4个样式定义行

    def test_simple_dag_mermaid(self):
        """测试简单 DAG 的 mermaid 生成（两个节点一条边）"""
        with DAG(
            dag_id="simple_dag", validate_single_end=False, enable_auto_run=False
        ) as dag:
            task1 = PythonOperator(
                task_id="task1", python_callable=lambda context: "task1"
            )
            task2 = PythonOperator(
                task_id="task2", python_callable=lambda context: "task2"
            )
            task1 >> task2

        mermaid = dag.to_mermaid()

        # 验证节点
        assert 'task1["task1"];' in mermaid.replace("    ", "")
        assert 'task2["task2"];' in mermaid.replace("    ", "")

        # 验证边
        assert "task1-->task2;" in mermaid.replace("    ", "")

        # 所有任务默认应该是 PENDING 状态，不会有特殊样式
        assert "class task1" not in mermaid
        assert "class task2" not in mermaid

    def test_complex_dag_mermaid(self):
        """测试复杂 DAG 的 mermaid 生成（菱形结构）"""
        with DAG(
            dag_id="complex_dag", validate_single_end=False, enable_auto_run=False
        ) as dag:
            start = PythonOperator(
                task_id="start", python_callable=lambda context: "start"
            )
            branch1 = PythonOperator(
                task_id="branch1", python_callable=lambda context: "branch1"
            )
            branch2 = PythonOperator(
                task_id="branch2", python_callable=lambda context: "branch2"
            )
            end = PythonOperator(task_id="end", python_callable=lambda context: "end")

            start >> [branch1, branch2]
            [branch1, branch2] >> end

        mermaid = dag.to_mermaid()

        # 验证节点
        node_ids = ["start", "branch1", "branch2", "end"]
        for node_id in node_ids:
            assert f'{node_id}["{node_id}"];' in mermaid.replace("    ", "")

        # 验证边
        edges = ["start-->branch1", "start-->branch2", "branch1-->end", "branch2-->end"]
        for edge in edges:
            assert f"{edge};" in mermaid.replace("    ", "")

    def test_dag_with_status_mermaid(self):
        """测试带有不同状态的任务的 mermaid 生成"""
        with DAG(
            dag_id="status_dag", validate_single_end=False, enable_auto_run=False
        ) as dag:
            task1 = PythonOperator(
                task_id="task1", python_callable=lambda context: "task1"
            )
            task2 = PythonOperator(
                task_id="task2", python_callable=lambda context: "task2"
            )
            task3 = PythonOperator(
                task_id="task3", python_callable=lambda context: "task3"
            )
            task4 = PythonOperator(
                task_id="task4", python_callable=lambda context: "task4"
            )

            task1 >> [task2, task3] >> task4

        # 手动设置不同的任务状态
        dag.task_status["task1"] = TaskStatus.SUCCESS
        dag.task_status["task2"] = TaskStatus.FAILED
        dag.task_status["task3"] = TaskStatus.RUNNING
        dag.task_status["task4"] = TaskStatus.SKIPPED

        mermaid = dag.to_mermaid()

        # 验证图表基本结构
        assert "graph TD;" in mermaid.replace("    ", "")

        # 验证节点是否存在
        node_ids = ["task1", "task2", "task3", "task4"]
        for node_id in node_ids:
            assert f'{node_id}["{node_id}"];' in mermaid.replace("    ", "")

        # 验证边的连接是否正确
        edges = ["task1-->task2", "task1-->task3", "task2-->task4", "task3-->task4"]
        for edge in edges:
            assert f"{edge};" in mermaid.replace("    ", "")

        # 验证状态样式
        assert "class task1 success;" in mermaid.replace("    ", "")
        assert "class task2 failure;" in mermaid.replace("    ", "")
        assert "class task3 running;" in mermaid.replace("    ", "")
        assert "class task4 skipped;" in mermaid.replace("    ", "")

        # 验证样式定义
        assert "classDef success fill:#a3be8c;" in mermaid.replace("    ", "")
        assert "classDef failure fill:#bf616a;" in mermaid.replace("    ", "")
        assert "classDef running fill:#ebcb8b;" in mermaid.replace("    ", "")
        assert "classDef skipped fill:#b48ead;" in mermaid.replace("    ", "")

    def test_dag_with_pending_status_mermaid(self):
        """测试带有PENDING状态任务的mermaid生成（默认状态不应有特殊样式）"""
        with DAG(
            dag_id="pending_dag", validate_single_end=False, enable_auto_run=False
        ) as dag:
            task1 = PythonOperator(
                task_id="task1", python_callable=lambda context: "task1"
            )
            task2 = PythonOperator(
                task_id="task2", python_callable=lambda context: "task2"
            )

            task1 >> task2

        # 默认状态为PENDING，无需手动设置

        mermaid = dag.to_mermaid()

        # 验证图表基本结构
        assert "graph TD;" in mermaid.replace("    ", "")

        # 验证节点存在
        assert 'task1["task1"];' in mermaid.replace("    ", "")
        assert 'task2["task2"];' in mermaid.replace("    ", "")

        # 验证边连接
        assert "task1-->task2;" in mermaid.replace("    ", "")

        # 验证PENDING状态的任务不应有特殊样式
        assert "class task1" not in mermaid
        assert "class task2" not in mermaid

        # 验证样式定义仍然存在
        assert "classDef success fill:#a3be8c;" in mermaid.replace("    ", "")
        assert "classDef failure fill:#bf616a;" in mermaid.replace("    ", "")
        assert "classDef running fill:#ebcb8b;" in mermaid.replace("    ", "")
        assert "classDef skipped fill:#b48ead;" in mermaid.replace("    ", "")

    def test_dag_status_change_mermaid(self):
        """测试DAG运行前后状态变化在mermaid图中的反映"""
        # 创建一个简单的成功DAG
        with DAG(
            dag_id="status_change_dag", validate_single_end=False, enable_auto_run=False
        ) as dag:
            task1 = PythonOperator(
                task_id="task1", python_callable=lambda context: "task1 result"
            )
            task2 = PythonOperator(
                task_id="task2", python_callable=lambda context: "task2 result"
            )

            task1 >> task2

        # 运行前验证
        before_run_mermaid = dag.to_mermaid()
        # 运行前应为PENDING状态，无特殊样式
        assert "class task1" not in before_run_mermaid
        assert "class task2" not in before_run_mermaid

        # 运行DAG
        dag.run()

        # 运行后验证
        after_run_mermaid = dag.to_mermaid()
        # 运行后应为SUCCESS状态
        assert "class task1 success;" in after_run_mermaid.replace("    ", "")
        assert "class task2 success;" in after_run_mermaid.replace("    ", "")

        # 创建一个包含失败任务的DAG
        with DAG(
            dag_id="status_change_fail_dag",
            validate_single_end=False,
            enable_auto_run=False,
        ) as dag:

            def succeed_task(context):
                return "success"

            def fail_task(context):
                raise ValueError("故意失败")

            task1 = PythonOperator(task_id="task1", python_callable=succeed_task)
            task2 = PythonOperator(task_id="task2", python_callable=fail_task)

            task1 >> task2

        # 运行DAG（会在task2处失败）
        try:
            dag.run()
        except ValueError:
            pass  # 预期的失败

        # 验证失败状态
        fail_mermaid = dag.to_mermaid()
        assert "class task1 success;" in fail_mermaid.replace("    ", "")
        assert "class task2 failure;" in fail_mermaid.replace("    ", "")

    def test_branch_dag_mermaid(self):
        """测试带有BranchPythonOperator的DAG的Mermaid表示"""
        with DAG(
            dag_id="branch_dag", validate_single_end=False, enable_auto_run=False
        ) as dag:
            # 定义分支条件
            def branch_condition(context):
                return "path_a"  # 选择path_a分支

            start = PythonOperator(
                task_id="start", python_callable=lambda context: "start"
            )

            branch = BranchPythonOperator(
                task_id="branch", python_callable=branch_condition
            )

            path_a = PythonOperator(
                task_id="path_a", python_callable=lambda context: "path_a"
            )

            path_b = PythonOperator(
                task_id="path_b", python_callable=lambda context: "path_b"
            )

            end = PythonOperator(task_id="end", python_callable=lambda context: "end")

            # 定义DAG结构
            start >> branch >> [path_a, path_b]
            [path_a, path_b] >> end

        # 运行前验证
        before_mermaid = dag.to_mermaid()
        # 验证节点和边
        assert 'branch["branch"];' in before_mermaid.replace("    ", "")
        assert "branch-->path_a;" in before_mermaid.replace("    ", "")
        assert "branch-->path_b;" in before_mermaid.replace("    ", "")

        # 运行DAG（应该只执行path_a分支）
        dag.run()

        # 运行后验证
        after_mermaid = dag.to_mermaid()

        # 验证状态
        assert "class start success;" in after_mermaid.replace("    ", "")
        assert "class branch success;" in after_mermaid.replace("    ", "")
        assert "class path_a success;" in after_mermaid.replace("    ", "")
        assert "class path_b skipped;" in after_mermaid.replace("    ", "")
        assert "class end success;" in after_mermaid.replace("    ", "")

        # 验证跳过的分支的样式定义
        assert "classDef skipped fill:#b48ead;" in after_mermaid.replace("    ", "")
