# rioflux 设计文档

## 项目概述

rioflux 是一个基于 DAG（有向无环图）的工作流引擎，用于构建和执行复杂的任务流程。它提供了灵活的任务编排能力，支持任务间的依赖关系管理、分支控制等功能。

## 核心概念

### DAG（有向无环图）

DAG 是工作流的核心抽象，它由以下几个关键部分组成：

- **任务（Task）**：工作流中的基本执行单元
- **边（Edge）**：定义任务之间的依赖关系
- **上下文（Context）**：在任务之间共享数据的机制

### 任务状态

任务在执行过程中可能处于以下状态：

- **PENDING**：等待执行
- **SUCCESS**：执行成功
- **FAILED**：执行失败
- **SKIPPED**：被跳过（用于分支控制）

## 架构设计

### 核心组件

1. **BaseOperator**
   - 所有具体任务的基类
   - 提供任务 ID 管理
   - 实现任务依赖关系的管理
   - 支持 `>>` 和 `<<` 操作符来定义任务依赖

2. **DAG**
   - 管理任务集合和依赖关系
   - 提供任务执行的调度逻辑
   - 维护任务状态
   - 管理上下文数据

3. **DAGContext**
   - 提供任务间数据共享机制
   - 存储任务执行结果
   - 支持变量的设置和获取

4. **BranchOperator**
   - 实现条件分支控制
   - 根据条件函数结果选择执行路径
   - 自动跳过未选中的分支

## 实现细节

### 任务依赖管理

```python
# 使用操作符语法定义依赖
task1 >> task2  # task2 依赖 task1
[task2, task3] >> task4  # task4 依赖 task2 和 task3
```

### 任务执行流程

1. DAG 初始化时重置所有任务状态为 PENDING
2. 循环检查并执行所有依赖满足的任务
3. 任务执行完成后更新状态和结果
4. 直到所有任务执行完成或出现错误

### 分支控制

BranchOperator 通过以下方式实现分支控制：

1. 执行用户提供的条件函数
2. 根据返回的 task_id 选择执行路径
3. 将未选中的分支标记为 SKIPPED

## 使用示例

```python
from rioflux.branch_operator import BranchOperator
from rioflux.dag import DAG
from rioflux.operator import BaseOperator

# 定义具体的 Operator
class DataLoadOperator(BaseOperator):
    def execute(self, context):
        data = {"value": 5}
        context.set_var("loaded_data", data)
        return data

# 定义分支条件
def branch_func(context):
    data = context.get_var("loaded_data")
    return "process_high" if data["value"] > 10 else "process_low"

# 创建 DAG 并定义任务
with DAG(dag_id="example_dag") as dag:
    load_task = DataLoadOperator("load_task")
    branch_task = BranchOperator(
        task_id="branch_task",
        python_callable=branch_func
    )
    
    # 定义任务依赖
    load_task >> branch_task >> [process_high, process_low]

```

## 最佳实践

1. **任务粒度**
   - 保持任务功能单一
   - 合理划分任务边界
   - 避免任务间过度耦合

2. **错误处理**
   - 在任务中妥善处理异常
   - 提供清晰的错误信息
   - 考虑添加重试机制

3. **上下文使用**
   - 合理使用上下文共享数据
   - 避免在上下文中存储过大数据
   - 及时清理不需要的上下文数据

## 未来展望

1. 支持任务并行执行
2. 添加任务重试机制
3. 提供任务执行监控和可视化
4. 支持子 DAG
5. 添加更多类型的操作符