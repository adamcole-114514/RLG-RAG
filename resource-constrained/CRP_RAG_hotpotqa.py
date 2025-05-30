from graph_construction import iteration_graph
from graph_retrieval_generate import graph_knowledge_retrieval
from CRP_RAG_demo_generate import graph_answering
from zhipuai import ZhipuAI
import json
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

def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    question = []
    gold_answer = []
    print("----------dataset loading----------")
    for i in trange(0, len(dataset)):
        question.append(dataset[i]["prompt"])
        gold_answer.append(dataset[i]["response"])
    return question[:700], gold_answer[:700]

def question_answering(question,model):
    graph = iteration_graph(question,2,2,model)
    graph_with_knowledge = graph_knowledge_retrieval(graph,model,question)
    answer = graph_answering(graph_with_knowledge,question,model)
    return answer

def testing(questions,model):
    answer = []
    for i in trange(0,len(questions)):
        try:
            response = question_answering(questions[i],model)
        except:
            response = "Error"
        answer.append(response)
    return answer

def evaluation_and_save(question, answer, gold_answer, output_path, model):
    em_count = 0
    acc_count = 0
    dict_list = []
    for i in trange(0,len(answer)):
        flag = False
        if '\n' in gold_answer[i]:
            answer_list = gold_answer[i].split('\n')
            for j in range(0,len(answer_list)):
                if answer_list[j] in answer[i]:
                    count += 1
                    flag = True
                    break
        else:
            if gold_answer[i] in answer[i]:
                count += 1
                flag = True
        acc_prompt = "Please judge whether the given answer is consistent with the golden answer, if so output 'yes', otherwise output 'no'.\nThe answer given is:{0}\nThe golden answer is:{1}".format(answer[i],gold_answer[i])
        acc_response = QA(acc_prompt,model,0.1)
        if "yes" in acc_response:
            acc_count += 1
        temp_dict = {"question": question[i], "answer": answer[i], "gold_answer": gold_answer[i], "if_correct": flag}
        dict_list.append(temp_dict)

    em_score = em_count / len(answer)
    acc_score = acc_count / len(answer)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dict_list:
            json.dump(item, f, ensure_ascii=False)  # 将字典转换为JSON格式并写入文件
            f.write('\n')  # 在每行之后添加换行符
    
    print("EM: {0}\nAcc: {1}".format(em_score,acc_score))

if __name__ == "__main__":
    model = load_model()
    question,gold_answer = load_dataset("datasets/hotpotqa-dev.jsonl")
    answer = testing(question,model)
    evaluation_and_save(question,answer,gold_answer,"output_hotpotqa.json",model) 