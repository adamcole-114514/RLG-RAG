from zhipuai import ZhipuAI
from tqdm import trange
import json

def QA(question, model, tem):
    response = model.chat.completions.create(
    model="glm-4-flash",  
    messages=[
        {"role": "user", "content": question},
        ],
    temperature=tem
    )
    return response.choices[0].message.content

def load_model():
    client = ZhipuAI(api_key="3dcccccb295047f581e5b4b54fc6876f.MNya69E3tiW1qONN")
    return client

def generate_new_nodes(model,prompt,question,num_of_new_nodes,step_num):
    """
    生成新结点，并将其与旧结点相连，步骤包括：

    1.按照数量限制生成新结点

    2.大模型根据新结点进行摘要

    3.确定所有结点是否为结束结点

    输出：每个结点有以下字段：步骤号（用来找当前步骤的结点）、内容、父结点内容、是否结束
    """
    nodes_prompt = "Please generate the next step of reasoning based on the question and the current reasoning step.\nThe question is: {0}\nThe current reasoning step is: {1}".format(question,prompt)
    # 生成新结点
    new_nodes = []
    new_nodes_dict = []
    for i in range(0,num_of_new_nodes):
        response = QA(nodes_prompt,model,0.7)
        new_nodes.append(response)
    # 对新结点进行摘要
    new_nodes_copy = new_nodes.copy()
    for i in range(0,len(new_nodes)):
        for j in range(i,len(new_nodes)):
            abstract_prompt = "Please judge whether the contents of the given text A and text B are similar or not, if similar then output 'similar', otherwise output 'not similar'.\nText A is: {0}\nText B is: {1}".format(new_nodes[i],new_nodes[j])
            label = QA(abstract_prompt,model,0.1)
            if label == "similar":
                merge_prompt = "Please generate a short paragraph that summarize the content of the given Text A and Text B, keeping the core content.\nText A is: {0}\nText B is:{1}".format(new_nodes[i],new_nodes[j])
                merge_response = QA(merge_prompt,model,0.1)
                if new_nodes[i] in new_nodes_copy:
                    new_nodes_copy.remove(new_nodes[i])
                if new_nodes[j] in new_nodes_copy:
                    new_nodes_copy.remove(new_nodes[j])
                new_nodes.append(merge_response)
            else:
                continue
    # 判断结束结点
    for i in range(0,len(new_nodes)):
        if_sink_prompt = "Please decide whether the given text can answer the following questions, if yes please answer 'yes', otherwise answer 'no'.\nThe text is: {0}\nThe question is: {1}".format(new_nodes[i],question)
        if_sink_response = QA(if_sink_prompt,model,0.1)
        if 'yes' in if_sink_response:
            new_nodes_dict.append({"step":step_num,"content":new_nodes[i],"prev":prompt,"if_sink":True})
        else:
            new_nodes_dict.append({"step":step_num,"content":new_nodes[i],"prev":prompt,"if_sink":False})
    return new_nodes_dict

def init_new_nodes(model,question,num_of_new_nodes,step_num):
    """
    生成第一批新结点，步骤包括：

    1.按照数量限制生成新结点

    2.大模型根据新结点进行摘要

    3.确定所有结点是否为结束结点

    输出：每个结点有以下字段：步骤号（用来找当前步骤的结点）、内容、是否结束
    """
    nodes_prompt = "Please generate the next step of reasoning based on the question .\nThe question is: {0}".format(question)
    # 生成新结点
    new_nodes = []
    new_nodes_dict = []
    for i in range(0,num_of_new_nodes):
        response = QA(nodes_prompt,model,0.7)
        new_nodes.append(response)
    # 对新结点进行摘要
    new_nodes_copy = new_nodes.copy()
    for i in range(0,len(new_nodes)):
        for j in range(i,len(new_nodes)):
            abstract_prompt = "Please judge whether the contents of the given text A and text B are similar or not, if similar then output 'similar', otherwise output 'not similar'.\nText A is: {0}\nText B is: {1}".format(new_nodes[i],new_nodes[j])
            label = QA(abstract_prompt,model,0.1)
            if label == "similar":
                merge_prompt = "Please generate a short paragraph that summarize the content of the given Text A and Text B, keeping the core content.\nText A is: {0}\nText B is:{1}".format(new_nodes[i],new_nodes[j])
                merge_response = QA(merge_prompt,model,0.1)
                if new_nodes[i] in new_nodes_copy:
                    new_nodes_copy.remove(new_nodes[i])
                if new_nodes[j] in new_nodes_copy:
                    new_nodes_copy.remove(new_nodes[j])
                new_nodes.append(merge_response)
            else:
                continue
    # 判断结束结点
    for i in range(0,len(new_nodes)):
        if_sink_prompt = "Please decide whether the given text can answer the following questions, if yes please answer 'yes', otherwise answer 'no'.\nThe text is: {0}\nThe question is: {1}".format(new_nodes[i],question)
        if_sink_response = QA(if_sink_prompt,model,0.1)
        if 'yes' in if_sink_response:
            new_nodes_dict.append({"step":step_num,"content":new_nodes[i],"if_sink":True})
        else:
            new_nodes_dict.append({"step":step_num,"content":new_nodes[i],"if_sink":False})
    return new_nodes_dict

def generate_last_nodes(model,prompt,question,num_of_new_nodes,step_num):
    """
    生成新结点，并将其与旧结点相连，步骤包括：

    1.按照数量限制生成新结点

    2.大模型根据新结点进行摘要

    3.确定所有结点是否为结束结点

    输出：每个结点有以下字段：步骤号（用来找当前步骤的结点）、内容、父结点内容、是否结束
    """
    nodes_prompt = "Please generate the next step of reasoning based on the question and the current reasoning step.\nThe question is: {0}\nThe current reasoning step is: {1}".format(question,prompt)
    # 生成新结点
    new_nodes = []
    new_nodes_dict = []
    for i in range(0,num_of_new_nodes):
        response = QA(nodes_prompt,model,0.7)
        new_nodes.append(response)
    # 对新结点进行摘要
    new_nodes_copy = new_nodes.copy()
    for i in range(0,len(new_nodes)):
        for j in range(i,len(new_nodes)):
            abstract_prompt = "Please judge whether the contents of the given text A and text B are similar or not, if similar then output 'similar', otherwise output 'not similar'.\nText A is: {0}\nText B is: {1}".format(new_nodes[i],new_nodes[j])
            label = QA(abstract_prompt,model,0.1)
            if label == "similar":
                merge_prompt = "Please generate a short paragraph that summarize the content of the given Text A and Text B, keeping the core content.\nText A is: {0}\nText B is:{1}".format(new_nodes[i],new_nodes[j])
                merge_response = QA(merge_prompt,model,0.1)
                if new_nodes[i] in new_nodes_copy:
                    new_nodes_copy.remove(new_nodes[i])
                if new_nodes[j] in new_nodes_copy:
                    new_nodes_copy.remove(new_nodes[j])
                new_nodes.append(merge_response)
            else:
                continue
    # 所有内容都是结束结点
    for i in range(0,len(new_nodes)):
        new_nodes_dict.append({"step":step_num,"content":new_nodes[i],"prev":prompt,"if_sink":True})

    return new_nodes_dict

def iteration_graph(question,max_steps,num_of_new_nodes,model):
    graph = []
    prev_nodes = []
    for i in trange(0,max_steps):
        if i == 0:
            prev_nodes.append(question)
            # 初始化结点
            init_nodes = init_new_nodes(model,question,num_of_new_nodes,i)
            # 将问题结点加入图中
            sons = []
            for j in range(0,len(init_nodes)):
                sons.append(init_nodes[j]["content"])
            graph.append({"step":-1,"content":question,"sons":sons})
            prev_nodes.clear()
            # 将结束结点加入图中，将非结束结点作为下一步的启动条件
            for j in range(0,len(init_nodes)):
                if init_nodes[j]["if_sink"] == True:
                    graph.append({"step":0,"content":init_nodes[j]["content"],"sons":"None"})
                else:
                    prev_nodes.append(init_nodes[j]["content"])
        elif i == max_steps - 1:
            # 基于prev生成新结点，并将prev的结点和新结点全部放入图中
            for j in range(0,len(prev_nodes)):
                # 获得新结点
                tmp_nodes = generate_last_nodes(model,prev_nodes[j],question,num_of_new_nodes,i)
                # 将对应的prev_nodes信息补全
                sons = []
                for k in range(0,len(tmp_nodes)):
                    sons.append(tmp_nodes[k]["content"])
                graph.append({"step":i-1,"content":prev_nodes[j],"sons":sons})
                # 将所有新结点放入图中
                for k in range(0,len(tmp_nodes)):
                    graph.append({"step":i,"content":tmp_nodes[k]["content"],"sons":"None"})
        else:
            if_end = True
            for j in range(0,len(prev_nodes)):
            # 基于prev生成新结点
                tmp_nodes = generate_new_nodes(model,prev_nodes[j],question,3,i)
            # 将prev结点加入图中
                sons = []
                for k in range(0,len(tmp_nodes)):
                    sons.append(tmp_nodes[k]["content"])
                graph.append({"step":i-1,"content":prev_nodes[j],"sons":sons})
                prev_nodes.clear()
            # 将结束结点加入图中，将非结束结点作为下一步启动条件(如果全是结束结点就直接结束循环)
                for k in range(0,len(tmp_nodes)):
                    if tmp_nodes[k]["if_sink"] == True:
                        graph.append({"step":i,"content":tmp_nodes[k]["content"],"sons":"None"})
                    else:
                        if_end = False
                        prev_nodes.append(tmp_nodes[k]["content"])
            if if_end == True:
                break
    return graph

if __name__ == "__main__":
    model = load_model()
    question = input("input your question: ")
    graph = iteration_graph(question,2,2,model)
    with open('example_graph.json', 'w',encoding='utf-8') as f:
        json.dump(graph, f, indent=4, ensure_ascii=False)