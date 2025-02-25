import pytest

from rioflux import DAG


@pytest.fixture
def dag():
    """创建一个测试用的 DAG 实例，关闭单终止节点验证"""
    return DAG(dag_id="test_dag", validate_single_end=False)
