from typing import Any,List
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
class KB():
    """一个用于构建和搜索知识库的类

    Attributes:
        embeddings: 一个 HuggingFaceBgeEmbeddings 类的实例对象，用于将文本转换为向量。
        vs_path: 一个字符串，表示知识库的本地保存路径。
        vector_store: 一个 FAISS 类的实例对象，用于存储和检索向量化的文档。
    """

    def __init__(self, vs_path="demo-vs") -> None:
        """
        Args:
            vs_path: 一个字符串，表示知识库的本地保存路径，默认为 "demo-vs"。
        """
        # 使用 BAAI/bge-large-zh-v1.5 模型
        self.embeddings = HuggingFaceBgeEmbeddings(model_name="../BAAIbge-large-en-v1.5",model_kwargs={'device': "cuda"})
        #self.embeddings = HuggingFaceEmbeddings(model_name="facebook-dpr-ctx_encoder-multiset-base",model_kwargs={'device': "cuda"})
        self.vs_path = vs_path
        self.vector_store = None
        if os.path.exists(vs_path):
            self.vector_store = FAISS.load_local(self.vs_path, self.embeddings, allow_dangerous_deserialization=True)
            #self.vector_store = FAISS.load_local(self.vs_path, self.embeddings, allow_dangerous_deserialization=True)

    def build_kb(self, texts:List[str],metadata:dict) -> Any:
        """

        Args:
            texts: 一个字符串列表，表示一组文本，每个文本对应一个知识库中的文档。
            metadata: 一个字典，表示一组元数据，用于给文档添加额外的信息。

        Returns:
            None
        """
        # 转换为Document对象数组并向量化
        texts = [Document(page_content=text,metadata=metadata) for text in texts]
        vector_store = FAISS.from_documents(texts, self.embeddings)
        vector_store.save_local(self.vs_path)

    def search_kb(self, query:str, top_k:int) -> str:
        """类的搜索知识库的方法，从知识库中搜索与查询最相关的文档，并返回它们的内容。

        Args:
            query: 一个字符串，表示用户输入的查询。
            top_k: 一个整数，表示从知识库中返回的最相关的文档的数量。

        Returns:
            一个字符串，表示从知识库中搜索到的最相关的文档的内容

        Raises:
            Exception: 如果 vector_store 属性为 None，表示知识库不存在，抛出异常。
        """
        if self.vector_store is None:
            raise Exception("知识库不存在")
        # related_docs_with_score = (文档，相关度得分)
        related_docs_with_score = self.vector_store.similarity_search_with_score(query=query, k=top_k)
        contexts = []
        for pack in related_docs_with_score:
            doc, socre = pack
            content = doc.page_content
            contexts.append(content)
        return '\n'.join(contexts)
