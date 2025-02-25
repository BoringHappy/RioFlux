# RioFlux

A DAG (Directed Acyclic Graph) based Agent workflow engine for building and executing complex task processes.

[English](README.md) | [简体中文](README_zh-CN.md)

## Features

- DAG-based task orchestration
- Flexible task dependency management
- Conditional branching control
- Inter-task data sharing mechanism
- Native Python task support
- Agent task support (in development)

## Installation

```bash
pip install rioflux
```

## Quick Start

Here's a simple example showing how to create and execute a workflow with branching logic using RioFlux:

```python
from rioflux import DAG, PythonOperator, BranchPythonOperator

# Define task functions
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

# Define branch condition
def branch_func(context):
    data = context.get_var("loaded_data")
    return "process_high" if data["value"] > 10 else "process_low"

# Create DAG and define tasks
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
    
    # Define task dependencies
    load_task >> branch_task >> [high_task, low_task]

# Execute DAG
dag.run()
```

## Core Components

### BaseOperator
- Base class for all concrete tasks
- Provides task ID management
- Implements task dependency management
- Supports `>>` and `<<` operators for defining dependencies

### DAG
- Manages task collections and dependencies
- Provides task execution scheduling logic
- Maintains task states
- Manages context data

### DAGContext
- Provides inter-task data sharing mechanism
- Stores task execution results
- Supports variable setting and retrieval

### Built-in Operators
- **PythonOperator**: Executes Python callables
- **BranchPythonOperator**: Implements conditional branching control
- **BaseAgent**: Base class for Agent tasks (in development)

## Best Practices

1. **Task Granularity**
   - Keep task functionality singular
   - Properly define task boundaries
   - Avoid excessive coupling between tasks

2. **Error Handling**
   - Handle exceptions properly in tasks
   - Provide clear error messages
   - Consider adding retry mechanisms

3. **Context Usage**
   - Use context data sharing judiciously
   - Avoid storing large data in context
   - Clean up unnecessary context data timely

## Development Roadmap

- Support parallel task execution
- Add task retry mechanism
- Provide task execution monitoring and visualization
- Support sub-DAGs
- Add more types of operators

## Requirements

- Python >= 3.13
