import pytest
from unittest.mock import MagicMock, patch
import logging
from typing import Dict, Any, List

from rioflux import DAG
from rioflux.operator.base_agent import BaseAgent, AgentConfig
from rioflux.models.status import TaskStatus


# 创建一个简单的BaseAgent子类用于测试
class TestAgent(BaseAgent):
    def __init__(self, task_id: str, api_key: str = "fake_key", **kwargs):
        # 我们需要在初始化之前模拟OpenAI客户端
        with patch("openai.OpenAI"):
            super().__init__(task_id=task_id, api_key=api_key, **kwargs)
        self.processed_input = None

    def process_input(self, context: Dict[str, Any]) -> str:
        self.processed_input = f"处理输入: {context.get_var('input_data', '无数据')}"
        return self.processed_input

    # 覆盖execute方法，避免调用真正的API
    def execute(self, context: Dict[str, Any]) -> Any:
        # 处理输入
        user_prompt = self.process_input(context)
        self.add_message_to_history("user", user_prompt)

        # 模拟AI响应
        ai_response = "这是AI的回复"
        self.add_message_to_history("assistant", ai_response)

        # 处理输出
        result = self.process_output(ai_response, context)
        return result


# 创建一个带有自定义output处理的Agent
class CustomOutputAgent(BaseAgent):
    def __init__(self, task_id: str, api_key: str = "fake_key", **kwargs):
        # 我们需要在初始化之前模拟OpenAI客户端
        with patch("openai.OpenAI"):
            super().__init__(task_id=task_id, api_key=api_key, **kwargs)

    def process_input(self, context: Dict[str, Any]) -> str:
        return f"分析数据: {context.get_var('data', '无数据')}"

    def process_output(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # 假设AI回复是JSON格式的摘要
        return {
            "original_response": response,
            "processed_data": f"处理后的数据: {response}",
        }

    # 覆盖execute方法，避免调用真正的API
    def execute(self, context: Dict[str, Any]) -> Any:
        # 处理输入
        user_prompt = self.process_input(context)
        self.add_message_to_history("user", user_prompt)

        # 模拟AI响应
        ai_response = "这是AI的回复"
        self.add_message_to_history("assistant", ai_response)

        # 处理输出
        result = self.process_output(ai_response, context)
        return result


# 创建一个会抛出异常的Agent
class ErrorAgent(BaseAgent):
    def __init__(self, task_id: str, api_key: str = "fake_key", **kwargs):
        # 我们需要在初始化之前模拟OpenAI客户端
        with patch("openai.OpenAI"):
            super().__init__(task_id=task_id, api_key=api_key, **kwargs)

    def process_input(self, context: Dict[str, Any]) -> str:
        return "处理数据"

    def execute(self, context: Dict[str, Any]) -> Any:
        raise Exception("API错误")


def test_agent_basic():
    """测试基本的Agent功能"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        # 创建Agent
        agent = TestAgent(task_id="test_agent")

        # 设置初始上下文
        initial_context = {"input_data": "测试数据"}

        # 运行DAG
        dag.run(initial_context=initial_context)

        # 验证Agent的process_input被调用并处理了上下文
        assert agent.processed_input == "处理输入: 测试数据"

        # 验证任务状态和结果
        assert dag.task_status["test_agent"] == TaskStatus.SUCCESS
        assert dag.context.get_task_result("test_agent") == "这是AI的回复"


def test_agent_custom_config():
    """测试带有自定义配置的Agent"""
    # 创建自定义配置
    custom_config = AgentConfig(
        model="gpt-4",
        system_prompt="你是一个数据分析专家",
        temperature=0.1,
        max_tokens=2000,
    )

    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        # 创建带自定义配置的Agent
        agent = TestAgent(
            task_id="test_agent", api_key="test_key", config=custom_config
        )

        # 运行DAG
        dag.run(initial_context={"input_data": "分析数据"})

        # 验证Agent配置
        assert agent.config.model == "gpt-4"
        assert agent.config.system_prompt == "你是一个数据分析专家"
        assert agent.config.temperature == 0.1
        assert agent.config.max_tokens == 2000


def test_agent_conversation_history():
    """测试Agent的对话历史记录功能"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        agent = TestAgent(task_id="test_agent")

        # 运行DAG
        dag.run(initial_context={"input_data": "第一条消息"})

        # 验证对话历史
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[0]["content"] == "处理输入: 第一条消息"
        assert agent.conversation_history[1]["role"] == "assistant"
        assert agent.conversation_history[1]["content"] == "这是AI的回复"

        # 再次运行，验证历史被保留
        dag.dag_status = TaskStatus.PENDING  # 重置DAG状态
        dag.run(initial_context={"input_data": "第二条消息"})

        # 验证对话历史包含了之前的交互
        assert len(agent.conversation_history) == 4
        assert agent.conversation_history[2]["role"] == "user"
        assert agent.conversation_history[2]["content"] == "处理输入: 第二条消息"
        assert agent.conversation_history[3]["role"] == "assistant"
        assert agent.conversation_history[3]["content"] == "这是AI的回复"


def test_agent_clear_history():
    """测试Agent的清除历史功能"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        agent = TestAgent(task_id="test_agent")

        # 运行DAG添加一些历史
        dag.run(initial_context={"input_data": "测试消息"})
        assert len(agent.conversation_history) == 2

        # 清除历史
        agent.clear_history()
        assert len(agent.conversation_history) == 0

        # 再次运行
        dag.dag_status = TaskStatus.PENDING
        dag.run(initial_context={"input_data": "新消息"})

        # 验证只有新消息，没有历史消息
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[0]["content"] == "处理输入: 新消息"


def test_agent_custom_output_processing():
    """测试Agent的自定义输出处理"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        agent = CustomOutputAgent(task_id="test_agent")

        # 运行DAG
        dag.run(initial_context={"data": "测试数据集"})

        # 验证输出被自定义处理
        result = dag.context.get_task_result("test_agent")
        assert isinstance(result, dict)
        assert result["original_response"] == "这是AI的回复"
        assert result["processed_data"] == "处理后的数据: 这是AI的回复"


def test_agent_error_handling():
    """测试Agent的错误处理"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        agent = ErrorAgent(task_id="test_agent")

        # 运行DAG，应该捕获并重新抛出异常
        with pytest.raises(Exception, match="API错误"):
            dag.run(initial_context={"input_data": "测试数据"})

        # 验证任务状态
        assert dag.task_status["test_agent"] == TaskStatus.FAILED
        assert dag.dag_status == TaskStatus.FAILED
