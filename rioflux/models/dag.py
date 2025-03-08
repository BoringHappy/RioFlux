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

        # 添加一个集合来跟踪已经处理的任务
        self.processed_tasks: Set[str] = set()
        # 添加一个标志来表示 DAG 是否正在运行
        self.is_running: bool = False

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
        """添加任务到 DAG

        如果 DAG 已经运行，新的任务将被添加并标记为 PENDING 状态，
        但不会影响已经运行的任务。
        """
        if task.task_id in self.tasks:
            raise ValueError(f"任务 {task.task_id} 已存在于 DAG {self.dag_id} 中")

        self.tasks[task.task_id] = task
        self.task_status[task.task_id] = TaskStatus.PENDING
        task.dag = self

    def add_edge(self, upstream_task: "BaseOperator", downstream_task: "BaseOperator"):
        """添加边（依赖关系）到 DAG

        如果 DAG 已经运行，只有当下游任务尚未执行时，新的边才会生效。
        已经执行的任务不会受到影响。
        """
        edge_is_safe = self._check_edge_safety(
            upstream_task.task_id, downstream_task.task_id
        )
        self.edges.add((upstream_task.task_id, downstream_task.task_id))
        upstream_task.downstream_task_ids.add(downstream_task.task_id)
        downstream_task.upstream_task_ids.add(upstream_task.task_id)

        # 如果 DAG 正在运行并且边不安全，发出警告
        if not edge_is_safe:
            print(
                f"警告：任务 {downstream_task.task_id} 已经执行，添加的依赖关系 {upstream_task.task_id} -> {downstream_task.task_id} 不会生效"
            )

    def _check_edge_safety(
        self, upstream_task_id: str, downstream_task_id: str
    ) -> bool:
        """检查添加的边是否安全（不会影响已处理的任务）

        安全条件：
        1. DAG 没有运行
        2. 或者下游任务尚未处理

        返回：
            bool: 如果边是安全的返回 True，否则返回 False
        """
        return not self.is_running or downstream_task_id not in self.processed_tasks

    def _get_ready_tasks(self) -> List[str]:
        """获取所有依赖已满足的任务"""
        ready_tasks = []
        for task_id in self.tasks:
            # 跳过非 PENDING 状态的任务
            if (
                task_id not in self.task_status
                or self.task_status[task_id] != TaskStatus.PENDING
            ):
                continue

            task = self.tasks[task_id]
            # 确保新添加的上游任务已经在任务状态字典中
            upstream_status_ok = True
            for upstream_id in task.upstream_task_ids:
                if upstream_id not in self.task_status:
                    # 如果上游任务是新添加的，为其设置 PENDING 状态
                    self.task_status[upstream_id] = TaskStatus.PENDING
                    upstream_status_ok = False
                elif (
                    self.task_status[upstream_id] != TaskStatus.SUCCESS
                    and self.task_status[upstream_id] != TaskStatus.SKIPPED
                ):
                    upstream_status_ok = False

            if upstream_status_ok:
                ready_tasks.append(task_id)

        return ready_tasks

    def run(self, initial_context: Dict[str, Any] = None) -> None:
        """执行 DAG"""

        if TaskStatus.is_finished(self.dag_status):
            print(f"DAG {self.dag_id} 已完成，跳过执行")
            return
        else:
            self.dag_status = TaskStatus.PENDING
            self.is_running = True  # 标记 DAG 开始运行

        # 验证终止节点
        self._validate_single_end_node()
        # 重置所有任务状态
        self.task_status = {task_id: TaskStatus.PENDING for task_id in self.tasks}
        self.processed_tasks.clear()  # 清空已处理任务集合
        self.context.clear()

        # 初始化上下文
        if initial_context:
            for key, value in initial_context.items():
                self.context.set_var(key, value)

        try:
            # 当存在 PENDING 状态的任务时，继续执行
            while any(
                status == TaskStatus.PENDING for status in self.task_status.values()
            ):
                ready_tasks = self._get_ready_tasks()
                if not ready_tasks:
                    # 检查是否有新添加的任务需要设置状态
                    self._update_task_status_for_new_tasks()
                    # 重新获取准备好的任务
                    ready_tasks = self._get_ready_tasks()

                    # 如果仍然没有准备好的任务，检查是否存在循环依赖
                    if not ready_tasks:
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
                        # 添加到已处理任务集合
                        self.processed_tasks.add(task_id)
                    except Exception as e:
                        self.task_status[task_id] = TaskStatus.FAILED
                        self.dag_status = TaskStatus.FAILED
                        print(f"任务 {task_id} 失败: {e}")
                        raise
            self.dag_status = TaskStatus.SUCCESS
        finally:
            self.is_running = False  # 标记 DAG 运行结束

    def _update_task_status_for_new_tasks(self) -> None:
        """更新新添加的任务的状态

        当 DAG 运行过程中添加了新任务时，需要更新任务状态字典
        """
        # 找出所有没有状态的任务，为它们设置 PENDING 状态
        for task_id in self.tasks:
            if task_id not in self.task_status:
                self.task_status[task_id] = TaskStatus.PENDING
                print(f"发现新添加的任务: {task_id}")

    def __enter__(self):
        if not hasattr(self._thread_local, "active_dag_stack"):
            self._thread_local.active_dag_stack = []
        self._thread_local.active_dag_stack.append(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dag_status == TaskStatus.PENDING and self.enable_auto_run:
            self.run()
        self._thread_local.active_dag_stack.pop()

    def to_mermaid(self) -> str:
        """
        生成 DAG 的 Mermaid 图表表示

        返回：
            str: mermaid 语法的图表字符串
        """
        mermaid_lines = ["graph TD;"]

        # 添加所有节点
        for task_id in self.tasks:
            # 使用节点ID和展示名称，并添加样式
            mermaid_lines.append(f'    {task_id}["{task_id}"];')

        # 添加所有边
        for edge in self.edges:
            upstream_id, downstream_id = edge
            mermaid_lines.append(f"    {upstream_id}-->{downstream_id};")

        # 添加节点状态样式
        for task_id, status in self.task_status.items():
            if status == TaskStatus.SUCCESS:
                mermaid_lines.append(f"    class {task_id} success;")
            elif status == TaskStatus.FAILED:
                mermaid_lines.append(f"    class {task_id} failure;")
            elif status == TaskStatus.RUNNING:
                mermaid_lines.append(f"    class {task_id} running;")
            elif status == TaskStatus.SKIPPED:
                mermaid_lines.append(f"    class {task_id} skipped;")

        # 添加样式定义
        mermaid_lines.append("    classDef success fill:#a3be8c;")
        mermaid_lines.append("    classDef failure fill:#bf616a;")
        mermaid_lines.append("    classDef running fill:#ebcb8b;")
        mermaid_lines.append("    classDef skipped fill:#b48ead;")

        return "\n".join(mermaid_lines)

    @classmethod
    def get_current_dag(cls) -> Optional["DAG"]:
        if (
            hasattr(cls._thread_local, "active_dag_stack")
            and cls._thread_local.active_dag_stack
        ):
            return cls._thread_local.active_dag_stack[-1]
        return None

    def add_dynamic_task(
        self,
        task: "BaseOperator",
        upstream_tasks: List["BaseOperator"] = None,
        downstream_tasks: List["BaseOperator"] = None,
        replace_conflicting_edges: bool = False,
    ) -> "BaseOperator":
        """
        在 DAG 运行过程中动态添加任务及其依赖关系

        Args:
            task: 要添加的任务
            upstream_tasks: 该任务的上游任务列表
            downstream_tasks: 该任务的下游任务列表
            replace_conflicting_edges: 是否替换冲突的边。如果为 True，则在添加新边时，
                                     会删除可能导致冲突的现有边（即上游任务们与下游任务们之间的直接连接）

        Returns:
            添加的任务对象

        Note:
            如果 DAG 正在运行，且下游任务已经执行，则添加的边不会生效，会发出警告
        """
        # 添加任务
        self.add_task(task)

        # 如果设置了替换冲突边，则先删除可能冲突的直接连接
        if replace_conflicting_edges and upstream_tasks and downstream_tasks:
            for upstream_task in upstream_tasks:
                for downstream_task in downstream_tasks:
                    # 检查是否有直接连接的边
                    if (upstream_task.task_id, downstream_task.task_id) in self.edges:
                        print(
                            f"删除冲突的边: {upstream_task.task_id} -> {downstream_task.task_id}"
                        )
                        self.delete_edge(upstream_task, downstream_task)

        # 添加上游依赖关系
        if upstream_tasks:
            for upstream_task in upstream_tasks:
                edge_is_safe = self._check_edge_safety(
                    upstream_task.task_id, task.task_id
                )
                self.add_edge(upstream_task, task)
                if not edge_is_safe:
                    print(
                        f"警告：任务 {task.task_id} 已经执行，添加的依赖关系 {upstream_task.task_id} -> {task.task_id} 不会生效"
                    )

        # 添加下游依赖关系
        if downstream_tasks:
            for downstream_task in downstream_tasks:
                edge_is_safe = self._check_edge_safety(
                    task.task_id, downstream_task.task_id
                )
                self.add_edge(task, downstream_task)
                if not edge_is_safe:
                    print(
                        f"警告：任务 {downstream_task.task_id} 已经执行，添加的依赖关系 {task.task_id} -> {downstream_task.task_id} 不会生效"
                    )

        return task

    def delete_edge(
        self, upstream_task: "BaseOperator", downstream_task: "BaseOperator"
    ) -> bool:
        """删除 DAG 中的边（依赖关系）

        Args:
            upstream_task: 上游任务
            downstream_task: 下游任务

        Returns:
            bool: 如果成功删除返回 True，如果边不存在返回 False

        Note:
            如果 DAG 正在运行，且下游任务已经执行，删除操作不会影响已执行的任务路径
        """
        edge = (upstream_task.task_id, downstream_task.task_id)

        # 检查边是否存在
        if edge not in self.edges:
            print(
                f"警告：边 {upstream_task.task_id} -> {downstream_task.task_id} 不存在于 DAG 中"
            )
            return False

        # 从 edges 集合中删除边
        self.edges.remove(edge)

        # 更新任务的依赖关系
        upstream_task.downstream_task_ids.discard(downstream_task.task_id)
        downstream_task.upstream_task_ids.discard(upstream_task.task_id)

        # 如果 DAG 正在运行且下游任务已经执行，发出警告
        if self.is_running and downstream_task.task_id in self.processed_tasks:
            print(
                f"警告：任务 {downstream_task.task_id} 已经执行，删除依赖关系 {upstream_task.task_id} -> {downstream_task.task_id} 不会影响已执行的路径"
            )

        return True


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
