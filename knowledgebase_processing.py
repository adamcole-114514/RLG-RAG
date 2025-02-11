import pandas as pd
from tqdm import trange
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

filename = 'psgs_w100.tsv'
chunk_size = 10000  # 你可以根据内存大小调整这个值
chunks = []
for chunk in pd.read_csv(filename, sep='\t', chunksize=chunk_size):
    # 在这里可以处理每个块，例如：清洗、聚合或分析
    #print(chunk)
    chunks.append(chunk)

# 合并所有块
df = pd.concat(chunks, axis=0)
#print(df.iloc[0,1])

contexts = []
for i in trange(0,len(df)):
    context = "{0}\n{1}".format(df.iloc[i,2],df.iloc[i,1])
    contexts.append(context)

print(contexts[0])

list_chunks = []
for i in trange(0, len(contexts), 10000):
    # 从当前位置i开始，截取长度为10000的子列表
    # 如果i+10000超过了原列表的长度，切片操作会自动处理，只截取到列表末尾
    list_chunks.append(contexts[i:i+10000])

print(len(list_chunks))

# 转换为Document对象数组并向量化
embeddings = HuggingFaceEmbeddings(model_name="../BAAIbge-large-en-v1.5", model_kwargs={'device': "cuda"})
texts = [Document(page_content=context,metadata={}) for context in list_chunks[0]]
vector_store_total = FAISS.from_documents(texts, embeddings)
vector_store_total.save_local("index")
for i in trange(1,len(list_chunks)):
    chunk = [Document(page_content=context,metadata={}) for context in list_chunks[i]]
    #vector_store_temp = FAISS.from_documents(chunk, embeddings)
    #vector_store_total.merge_from(vector_store_temp)
    vector_store_total.add_documents(chunk)
    vector_store_total.save_local("index")
