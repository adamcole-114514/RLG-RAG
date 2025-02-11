from sentence_transformers import SentenceTransformer
from Knowledge_Graph_Util import init_knowledge_graph, extend_knowledge_graph, retrieval_and_aggregation_with_graphnode, evaluation_with_graphnode, model_generate
import json
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from kb import KB

def load_models(model_path, retrieval_model_path, kb_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    kb = KB(vs_path=kb_path)
    retrieval_model = SentenceTransformer(retrieval_model_path)
    return model, tokenizer, retrieval_model, kb

def process_qa_pair(file_path):
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
    return question, gold_answer

def answer_generation(question, model, tokenizer, kb, retrieval_model):
    answer = []
    for i in trange(0,len(question)):
        root_nodeset = init_knowledge_graph(question[i], kb)
        reasoning_nodeset = extend_knowledge_graph(root_nodeset, question[i], model, tokenizer, 3, 5, retrieval_model)
        retrieval_nodeset = retrieval_and_aggregation_with_graphnode(reasoning_nodeset, kb, model, tokenizer)
        relevant_knowledge = evaluation_with_graphnode(retrieval_nodeset, model, tokenizer, question[i])
        prompt = "Please answer the questions based on relevant knowledge.\nThe relevant knowledge is: {0}\nThe question is: {1}".format(relevant_knowledge, question[i])
        response = model_generate(model, tokenizer, prompt)
        answer.append(response)
    return answer

def evaluation_and_save(question, answer, gold_answer, output_path):
    count = 0
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
        temp_dict = {"question": question[i], "answer": answer[i], "gold_answer": gold_answer[i], "if_correct": flag}
        dict_list.append(temp_dict)

    em_score = count / len(answer)
    print("EM: {0}".format(em_score))

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dict_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():
    model_path = "../Llama3.1-8B-Instruct"
    retrieval_model_path = "../BAAIbge-large-en-v1.5"
    kb_path = "../database/index/"
    dataset_path = "dataset/nq-dev.jsonl"
    model, tokenizer, retrieval_model, kb = load_models(model_path, retrieval_model_path, kb_path)
    questions, gold_answers = process_qa_pair(dataset_path)
    answers = answer_generation(questions, model, tokenizer, kb, retrieval_model)
    evaluation_and_save(questions, answers,gold_answers, "output/result.json")