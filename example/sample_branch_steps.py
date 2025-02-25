from rioflux import DAG, BaseOperator, BranchPythonOperator


# 定义具体的 Operator
class DataLoadOperator(BaseOperator):
    def execute(self, context):
        # 模拟加载数据
        data = {"value": 5}
        context.set_var("loaded_data", data)
        return data


class ProcessHighValueOperator(BaseOperator):
    def execute(self, context):
        data = context.get_var("loaded_data")
        print(f"处理高值数据: {data}")
        return "high_value_result"


class ProcessLowValueOperator(BaseOperator):
    def execute(self, context):
        data = context.get_var("loaded_data")
        print(f"处理低值数据: {data}")
        return "low_value_result"


# 添加一个新的结束节点操作符
class FinalizeOperator(BaseOperator):
    def execute(self, context):
        # 获取上游任务的结果
        high_result = context.get_task_result("process_high")
        low_result = context.get_task_result("process_low")

        # 根据实际执行的分支获取结果
        result = high_result if high_result else low_result
        print(f"最终处理结果: {result}")
        return result


# 定义分支条件函数
def branch_func(context):
    data = context.get_var("loaded_data")
    # 根据数据值选择处理路径
    return "process_high" if data["value"] > 10 else "process_low"


# 创建 DAG 并定义任务
with DAG(dag_id="branch_example_dag", validate_single_end=False) as dag:
    # 创建任务
    load_task = DataLoadOperator("load_task")
    branch_task = BranchPythonOperator(
        task_id="branch_task", python_callable=branch_func
    )
    process_high = ProcessHighValueOperator("process_high")
    process_low = ProcessLowValueOperator("process_low")
    finalize_task = FinalizeOperator("finalize")

    # 定义任务依赖
    load_task >> branch_task >> [process_high, process_low] >> finalize_task
