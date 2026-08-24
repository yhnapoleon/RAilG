"""RAilG —— 端到端 RAG Chatbot。

从数据索引到用户端问答的完整流水线:三级切块、混合召回、父块还原、
文档级权限、引用归因、检索评测。检索走 OpenSearch,模型走 OpenAI 兼容接口。
"""

__version__ = "0.2.0"
