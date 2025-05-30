from zhipuai import ZhipuAI
from tqdm import trange

def load_model():
    client = ZhipuAI(api_key="3dcccccb295047f581e5b4b54fc6876f.MNya69E3tiW1qONN")
    return client

def QA(question, model, tem):
    response = model.chat.completions.create(
    model="glm-4-flash",  
    messages=[
        {"role": "user", "content": question},
        ],
    temperature=tem
    )
    return response.choices[0].message.content

def graph_knowledge_retrieval(graph,model,question):
    graph_with_knowledge = []
    for i in trange(0,len(graph)):
        # 对每个结点进行检索和摘要
        abstract_prompt = "Please generate a short knowledge text based on the given text.\nThe given text is: {0}".format(graph[i]["content"])
        abstract = QA(abstract_prompt,model,0.1)
        # 评估结点的知识充分性
        evaluation_prompt = "Evaluate whether the given text can answer the following questions, if so output 'yes', otherwise output 'no'.\nThe question is:{0}\nThe given text is:{1}".format(question,abstract)
        evaluation_response = QA(evaluation_prompt,model,0.1)
        if "yes" in evaluation_response:
            graph_with_knowledge.append({"step":graph[i]["step"],"content":graph[i]["content"],"sons":graph[i]["sons"],"knowledge":abstract,"if_support":True})
        else:
            graph_with_knowledge.append({"step":graph[i]["step"],"content":graph[i]["content"],"sons":graph[i]["sons"],"knowledge":abstract,"if_support":False})
    return graph_with_knowledge