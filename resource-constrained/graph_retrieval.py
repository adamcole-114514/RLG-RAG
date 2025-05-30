from zhipuai import ZhipuAI
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from tqdm import trange

def load_model_and_database(model_path,database_path):
    client = ZhipuAI(api_key="3dcccccb295047f581e5b4b54fc6876f.MNya69E3tiW1qONN")
    embeddings = HuggingFaceBgeEmbeddings(model_path,model_kwargs={'device': "cuda"})
    vector_store = FAISS.load_local(database_path, embeddings, allow_dangerous_deserialization=True)
    return client,vector_store

def QA(question, model, tem):
    response = model.chat.completions.create(
    model="glm-4-flash",  
    messages=[
        {"role": "user", "content": question},
        ],
    temperature=tem
    )
    return response.choices[0].message.content

def knowledge_retrieval(prompt,knowledgebase,top_k):
    """
    一次检索
    """
    if knowledgebase is None:
        raise Exception("知识库不存在")
    # related_docs_with_score = (文档，相关度得分)
    related_docs_with_score = knowledgebase.similarity_search_with_score(query=prompt, k=top_k)
    contexts = []
    for pack in related_docs_with_score:
        doc, score = pack
        if score >= 0.7:
            content = doc.page_content
            contexts.append(content)
    return contexts

def graph_knowledge_retrieval(graph,model,knowledgebase,top_k,question):
    graph_with_knowledge = []
    for i in trange(0,len(graph)):
        # 对每个结点进行检索和摘要
        tmp_contexts = knowledge_retrieval(graph[i]["content"],knowledgebase,top_k)
        abstract_prompt = "Please generate a short text summarizing the content of the given text, keeping the core content.\nThe given text is as follows:\n{0}".format("\n".join(tmp_contexts))
        abstract = QA(abstract_prompt,model,0.1)
        # 评估结点的知识充分性
        evaluation_prompt = "Evaluate whether the given text can answer the following questions, if so output 'yes', otherwise output 'no'.\nThe question is:{0}\nThe given text is:{1}".format(question,abstract)
        evaluation_response = QA(evaluation_prompt,model,0.1)
        if "yes" in evaluation_response:
            graph_with_knowledge.append({"step":graph[i]["step"],"content":graph[i]["content"],"sons":graph[i]["sons"],"knowledge":abstract,"if_support":True})
        else:
            graph_with_knowledge.append({"step":graph[i]["step"],"content":graph[i]["content"],"sons":graph[i]["sons"],"knowledge":abstract,"if_support":False})
    return graph_with_knowledge