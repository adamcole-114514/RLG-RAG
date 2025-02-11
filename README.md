# RLG-RAG
该Github项目用于存储可以本地部署的RLG-RAG工作流代码，基于开源基座模型运作，我们推荐您使用Llama3.1系列模型以保证系统性能，项目默认使用Llama3.1-8B-Instruct模型以确保在低资源情况下系统的正常使用
## Quick Start
项目提供了快速开始方法，操作如下：
1.通过`knowledgebase_processing.py`对知识库文件进行编码，项目建议使用DPR项目中的知识库，具体可见DPR项目中项目资源下载脚本中的`psgs_w100.tsv`文件下载链接
2.在编码知识库文件后，调整`RLG-RAG_test.py`中的可选参数以测试RLG-RAG的效果
## Document Introduction
`Knowledge_Graph_Node.py`：用于定义推理图结点
`Knowledge_Graph_Util.py`：用于定义推理图运作的必需工具函数
`RLG-RAG_test.py`：用于定义完整的系统运行流程，并实现完整的RLG-RAG推理流程
`kb.py`：用于定义知识库加载、使用的工具类
`knowledgebase_processing.py`：用于处理形如`psgs_w100.tsv`格式的知识库文件
