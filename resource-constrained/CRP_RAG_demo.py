from graph_construction import iteration_graph
from graph_retrieval import graph_knowledge_retrieval
from zhipuai import ZhipuAI
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

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

def graph_answering(graph_with_knowledge,question,model):
    reasoning_path = []
    for i in range(0,len(graph_with_knowledge)):
        if graph_with_knowledge[i]["if_support"] == True:
            reasoning_path.append(graph_with_knowledge[i]["knowledge"])
    qa_prompt = "Please answer the questions briefly based on the knowledge given.\nThe given knowledge is:\n{0}\nThe question is:{1}".format("\n".join(reasoning_path),question)
    answer = QA(qa_prompt,model,0.1)
    return answer

if __name__ == "__main__":
    model,knowledge_base = load_model_and_database("../BAAIbge-en-large","../database/index")
    question = input()
    print("----------graph construction----------")
    graph = iteration_graph(question,3,3,model)
    print("----------knowledge retrieval----------")
    graph_with_knowledge = graph_knowledge_retrieval(graph,model,knowledge_base,3,question)
    print("----------question answering----------") 
    answer = graph_answering(graph_with_knowledge,question,model)