from Knowledge_Graph_Node import Knowledge_Graph_Node
from sklearn.metrics.pairwise import cosine_similarity

def init_knowledge_graph(question,kb):
    retrieval_result = kb.search_kb(question, top_k=5)
    root_node = Knowledge_Graph_Node(0, question, retrieval_result,[],[],False)
    nodes_set = [root_node]
    return nodes_set

def extend_knowledge_graph(nodes_set, question, model, tokenizer, num_of_nodes, max_steps, retrieval_model):
    for i in range(0,max_steps):
        print("Epoch: {0}".format(i + 1))
        reponses = []
        if len(nodes_set) == 1:
            prompt = "You will be given a master question with a sub-question, the master question is the question you will eventually need to answer and the sub-question is the question you have currently been given the answer to.\nBased on the sub-question you have been given, please answer what question you should answer next if you need to solve the master question, the answer should be a question.\nThe total question is:{0}\nThe sub-question is:{1}".format(question, nodes_set[0].subquestion)
            for j in range(0,num_of_nodes):
                response = model_generate(model, tokenizer, prompt)
                reponses.append((nodes_set[0].id,response))
            simple_reponses = nodes_similarity_operation(retrieval_model, reponses, 0.65, model, tokenizer)
            for j in range(0,len(simple_reponses)):
                new_node = Knowledge_Graph_Node(nodes_set[0].id + j + 1, simple_reponses[j][1], "None", simple_reponses[j][0], [], True)
                nodes_set[0].next_nodes.append(new_node.id)
                nodes_set.append(new_node)
        else:
            sink_node_id = []
            for j in range(0,len(nodes_set)):
                if nodes_set[j].if_sink == True:
                    sink_node_id.append(j)
                    nodes_set[j].if_sink = False
            for j in range(0,len(sink_node_id)):
                prompt = "You will be given a master question with a sub-question, the master question is the question you will eventually need to answer and the sub-question is the question you have currently been given the answer to.\nBased on the sub-question you have been given, please answer what question you should answer next if you need to solve the master question, the answer should be a question.\nThe total question is:{0}\nThe sub-question is:{1}".format(
                    question, nodes_set[sink_node_id[j]].subquestion)
                for k in range(0, num_of_nodes):
                    response = model_generate(model, tokenizer, prompt)
                    reponses.append((nodes_set[sink_node_id[j]].id, response))
            simple_reponses = nodes_similarity_operation(retrieval_model, reponses, 0.65, model, tokenizer)
            for j in range(0,len(simple_reponses)):
                max_id = find_max_id(nodes_set)
                new_node = Knowledge_Graph_Node(max_id + j + 1, simple_reponses[j][1], "None", simple_reponses[j][0], [], True)
                for k in range(simple_reponses[j][0]):
                    list_id = nodeid_convert_listid(nodes_set, simple_reponses[j][0][k])
                    nodes_set[list_id].next_nodes.append(new_node.id)
    return nodes_set

def retrieval_and_aggregation_with_graphnode(nodes_set, kb, model, tokenizer):
    for i in range(0,len(nodes_set)):
        retrieval_result = kb.search_kb(nodes_set[i].subquestion, top_k=5)
        aggregate_prompt = "Based on the question given, please summarize the key evidence in the following text that supports the answer to the question.\nThe question is: {0}\nThe texts are: {0}".format(nodes_set[i].subquestion, retrieval_result)
        aggregate_knowledge = model_generate(model, tokenizer, aggregate_prompt)
        nodes_set[i].knowledge = aggregate_knowledge
    return nodes_set

def evaluation_with_graphnode(nodes_set, model, tokenizer, question):
    avaliable_nodes = []
    for i in range(0,len(nodes_set)):
        prompt = "Please score the knowledge according to the role of the given knowledge in answering the question, the optional scores are divided into 6 options, indicated by A-F, which are described in detail below:\n\nA: 0 points, the given knowledge cannot answer the question\nB: 1 point, the knowledge given has a similar description of the topic to the question, but cannot answer the question\nC: 2 points, the knowledge given has a consistent description of the topic with the question, but cannot answer the question\nD: 3 points, the knowledge given has a consistent description of the topic with the question and includes a small amount of key evidence needed to answer the question\nE: 4 points, the knowledge given is consistent with the topic description of the question and includes some of the key evidence needed to answer the questionE: 4 points, the knowledge given is consistent with the topic description of the question and includes some of the key evidence needed to answer the question\nF: 5 points, the knowledge given is consistent with the topic description of the question and includes all of the key evidence needed to answer the question\n\nThe knowledge given is as follows:\n{0}\nThe question is as follows:\n{1}".format(nodes_set[i].knowledge,question)
        level = model_generate(model,tokenizer,prompt)
        if 'A' in level:
            score = 0
        if 'B' in level:
            score = 1
        if 'C' in level:
            score = 2
        if 'D' in level:
            score = 3
        if 'E' in level:
            score = 4
        if 'F' in level:
            score = 5
        if score >= 3:
            avaliable_nodes.append(nodes_set[i])
    knowledges = []
    for i in range(0,len(avaliable_nodes)):
        knowledges.append(avaliable_nodes[i].knowledge)
    return "\n".join(avaliable_nodes)

def model_generate(model, tokenizer, question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def nodes_similarity_operation(retrieval_model, texts_with_id, thereshold, model, tokenizer):
    vector_list = []
    for i in range(0,len(texts_with_id)):
        vector = retrieval_model.encode(texts_with_id[i][1], convert_to_tensor=True).cpu()
        vector_list.append((texts_with_id[i][0],vector))
    nodes_clusters = []
    for i in range(0,len(vector_list) - 1):
        clusting_node = [item for sublist in nodes_clusters for item in sublist]
        if texts_with_id[i] in clusting_node:
            continue
        else:
            node_cluster = [texts_with_id[i]]
            for j in range(i + 1,len(vector_list)):
                similarity = cosine_similarity([vector_list[i][1]], [vector_list[j][1]])[0][0]
                if similarity < thereshold:
                    continue
                else:
                    node_cluster.append(texts_with_id[j])
            nodes_clusters.append(node_cluster)
    new_nodes = []
    for i in range(0,len(nodes_clusters)):
        previd_list = []
        content_list = []
        for j in range(0,len(nodes_clusters[i])):
            previd_list.append(nodes_clusters[i][j][0])
            content_list.append(nodes_clusters[i][j][1])
        prompt = "Please summarize the following questions, which read as follows:\n{0}".format("\n".join(content_list))
        new_content = model_generate(model, tokenizer, prompt)
        new_node = (list(set(previd_list)),new_content)
        new_nodes.append(new_node)
    return new_nodes

def nodeid_convert_listid(nodes_set, nodeid):
    for i in range(0,len(nodes_set)):
        if nodes_set[i].id == nodeid:
            target = i
    return target

def find_max_id(nodes_set):
    max_id = 0
    for i in range(0,len(nodes_set)):
        if nodes_set[i].id >= max_id:
            max_id = nodes_set[i].id
    return max_id