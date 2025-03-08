from typing import Any, Callable, Dict, Optional, Union, List
import requests
from openai import OpenAI, NOT_GIVEN
from openai.types.chat import ChatCompletion
import logging
from dataclasses import dataclass
from .base_operator import BaseOperator


@dataclass
class AgentConfig:
    """Agent 配置类"""

    model: str = "gpt-3.5-turbo"
    system_prompt: str = "你是一个helpful的AI助手"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    presence_penalty: float = 0
    frequency_penalty: float = 0
    top_p: float = 1.0
    stop: Optional[Union[str, List[str]]] = None


class BaseAgent(BaseOperator):
    """
    AI Agent 基类：用于执行 AI 相关的任务

    特性：
    - 支持自定义配置参数
    - 内置错误处理机制
    - 支持对话历史记录
    - 支持流式输出

    用法示例：
    ```python
    class MyAgent(BaseAgent):
        def __init__(self, task_id: str, api_key: str):
            config = AgentConfig(
                model="gpt-4",
                system_prompt="你是一个专业的数据分析师"
            )
            super().__init__(task_id=task_id, api_key=api_key, config=config)

        def process_input(self, context: Dict[str, Any]) -> str:
            return f"请分析以下数据：{context.get_var('input_data')}"
    ```
    """

    def __init__(
        self,
        task_id: str,
        api_key: str,
        config: Optional[AgentConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化 AI Agent

        Args:
            task_id: 任务ID
            api_key: OpenAI API密钥
            config: Agent配置参数
            logger: 日志记录器
        """
        super().__init__(task_id=task_id)
        self.client = OpenAI(api_key=api_key)
        self.config = config or AgentConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.conversation_history: List[Dict[str, str]] = []

    def add_message_to_history(self, role: str, content: str) -> None:
        """添加消息到对话历史"""
        self.conversation_history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """清空对话历史"""
        self.conversation_history = []

    def process_input(self, context: Dict[str, Any]) -> str:
        """
        处理输入数据，生成用户提示
        子类应该重写此方法以自定义输入处理逻辑

        Args:
            context: DAG上下文

        Returns:
            str: 用户提示
        """
        raise NotImplementedError("子类必须实现process_input方法")

    def process_output(self, response: str, context: Dict[str, Any]) -> Any:
        """
        处理AI响应
        子类可以重写此方法以自定义输出处理逻辑

        Args:
            response: AI的响应文本
            context: DAG上下文

        Returns:
            Any: 处理后的结果
        """
        return response

    def _create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        """创建消息列表"""
        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def execute(self, context: Dict[str, Any]) -> Any:
        """
        执行AI任务

        Args:
            context: DAG上下文

        Returns:
            Any: 任务执行结果

        Raises:
            Exception: API调用失败时抛出异常
        """
        try:
            # 处理输入
            user_prompt = self.process_input(context)
            messages = self._create_messages(user_prompt)

            # 调用OpenAI API
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty,
                top_p=self.config.top_p,
                stop=self.config.stop,
            )

            # 获取AI响应
            ai_response = response.choices[0].message.content

            # 记录对话历史
            self.add_message_to_history("user", user_prompt)
            self.add_message_to_history("assistant", ai_response)

            # 处理输出
            result = self.process_output(ai_response, context)
            return result

        except Exception as e:
            self.logger.error(f"执行AI任务时发生错误: {str(e)}")
            raise

    async def execute_stream(self, context: Dict[str, Any]) -> Any:
        """
        流式执行AI任务

        Args:
            context: DAG上下文

        Yields:
            str: 生成的文本片段
        """
        try:
            user_prompt = self.process_input(context)
            messages = self._create_messages(user_prompt)

            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )

            collected_response = []
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_response.append(content)
                    yield content

            complete_response = "".join(collected_response)
            self.add_message_to_history("user", user_prompt)
            self.add_message_to_history("assistant", complete_response)

        except Exception as e:
            self.logger.error(f"流式执行AI任务时发生错误: {str(e)}")
            raise
