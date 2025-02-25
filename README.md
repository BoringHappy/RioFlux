# RioFlux

一个基于 DAG（有向无环图）的 Agent 工作流引擎，用于构建和执行复杂的任务流程。

## 特性

- 基于 DAG 的任务编排
- 灵活的任务依赖关系管理
- 支持条件分支控制
- 任务间数据共享机制
- Python 原生任务支持
- Agent 任务支持（开发中）

## 安装

```bash
pip install rioflux
```

## 快速开始

以下是一个简单的示例，展示如何使用 RioFlux 创建和执行一个包含分支逻辑的工作流：

```python
from rioflux import DAG, PythonOperator, BranchPythonOperator

# 定义任务函数
def load_data(context):
    data = {"value": 5}
    context.set_var("loaded_data", data)
    return data

def process_high(context):
    data = context.get_var("loaded_data")
    return f"Processing high value: {data['value']}"

def process_low(context):
    data = context.get_var("loaded_data")
    return f"Processing low value: {data['value']}"

# 定义分支条件
def branch_func(context):
    data = context.get_var("loaded_data")
    return "process_high" if data["value"] > 10 else "process_low"

# 创建 DAG 并定义任务
with DAG(dag_id="example_dag") as dag:
    load_task = PythonOperator(
        task_id="load_task",
        python_callable=load_data
    )
    
    branch_task = BranchPythonOperator(
        task_id="branch_task",
        python_callable=branch_func
    )
    
    high_task = PythonOperator(
        task_id="process_high",
        python_callable=process_high
    )
    
    low_task = PythonOperator(
        task_id="process_low",
        python_callable=process_low
    )
    
    # 定义任务依赖
    load_task >> branch_task >> [high_task, low_task]

# 执行 DAG
dag.run()
```

## 核心组件

### BaseOperator
- 所有具体任务的基类
- 提供任务 ID 管理
- 实现任务依赖关系的管理
- 支持 `>>` 和 `<<` 操作符来定义任务依赖

### DAG
- 管理任务集合和依赖关系
- 提供任务执行的调度逻辑
- 维护任务状态
- 管理上下文数据

### DAGContext
- 提供任务间数据共享机制
- 存储任务执行结果
- 支持变量的设置和获取

### 内置操作符
- **PythonOperator**: 执行 Python 可调用对象
- **BranchPythonOperator**: 实现条件分支控制
- **BaseAgent**: Agent 任务基类（开发中）

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

## 开发计划

- 支持任务并行执行
- 添加任务重试机制
- 提供任务执行监控和可视化
- 支持子 DAG
- 添加更多类型的操作符

## 要求

- Python >= 3.13
