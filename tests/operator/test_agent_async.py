import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from rioflux import DAG
from rioflux.operator.base_agent import BaseAgent, AgentConfig
from rioflux.models.status import TaskStatus


# 创建一个简单的BaseAgent子类用于测试异步特性
class AsyncTestAgent(BaseAgent):
    def __init__(self, task_id: str, api_key: str = "fake_key", **kwargs):
        # 模拟OpenAI客户端
        with patch("openai.OpenAI"):
            super().__init__(task_id=task_id, api_key=api_key, **kwargs)
        self.processed_input = None
        self.stream_chunks = []

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

    # 覆盖execute_stream方法，避免调用真正的API
    async def execute_stream(self, context: Dict[str, Any]) -> Any:
        # 处理输入
        user_prompt = self.process_input(context)

        # 模拟流式响应
        chunks = ["这是", "AI的", "流式", "回复"]
        for chunk in chunks:
            self.stream_chunks.append(chunk)
            yield chunk
            # 模拟网络延迟
            await asyncio.sleep(0.01)

        # 添加完整响应到历史
        complete_response = "".join(chunks)
        self.add_message_to_history("user", user_prompt)
        self.add_message_to_history("assistant", complete_response)


@pytest.mark.asyncio
async def test_agent_execute_stream():
    """测试Agent的流式执行功能"""
    with DAG(
        dag_id="test_dag", validate_single_end=False, enable_auto_run=False
    ) as dag:
        agent = AsyncTestAgent(task_id="test_agent")

        # 设置上下文
        initial_context = {"input_data": "流式测试"}
        dag.context.set_var("input_data", "流式测试")

        # 获取流式生成器
        stream = agent.execute_stream(context=dag.context)

        # 收集所有生成的块
        collected_chunks = []
        async for chunk in stream:
            collected_chunks.append(chunk)

        # 验证流式响应
        assert collected_chunks == ["这是", "AI的", "流式", "回复"]
        assert agent.stream_chunks == ["这是", "AI的", "流式", "回复"]

        # 验证历史记录
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[0]["content"] == "处理输入: 流式测试"
        assert agent.conversation_history[1]["role"] == "assistant"
        assert agent.conversation_history[1]["content"] == "这是AI的流式回复"
