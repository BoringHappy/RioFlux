from rioflux import DAG, BaseOperator


class Step1Operator(BaseOperator):
    def execute(self, context):
        print("执行步骤1：数据准备")
        data = {"step": 1, "message": "Hello"}
        context.set_var("step1_data", data)
        return data


class Step2Operator(BaseOperator):
    def execute(self, context):
        step1_data = context.get_var("step1_data")
        print(f"执行步骤2：处理来自步骤1的数据 - {step1_data}")
        processed_data = {"step": 2, "message": step1_data["message"] + " World"}
        context.set_var("step2_data", processed_data)
        return processed_data


class Step3Operator(BaseOperator):
    def execute(self, context):
        step2_data = context.get_var("step2_data")
        print(f"执行步骤3：最终处理 - {step2_data}")
        final_result = f"完成所有步骤！最终消息：{step2_data['message']}"
        return final_result


# 创建 DAG 并定义任务
with DAG(dag_id="simple_steps_dag") as dag:
    # 创建任务
    step1 = Step1Operator("step1")
    step2 = Step2Operator("step2")
    step3 = Step3Operator("step3")

    # 定义任务依赖（按顺序执行）
    step1 >> step2 >> step3
