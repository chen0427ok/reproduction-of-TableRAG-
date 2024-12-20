import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import json 
import requests
import time
import requests
import json
import time
import random
def get_chat_response(base_url, model_name, messages, temperature=0.8, ignore_eos=False):
   """
   透過 POST 請求與指定的模型交互，並獲取回應的 JSON 資料。
   Args:
       base_url (str): API 的基底 URL。
       model_name (str): 模型的名稱或路徑。
       messages (list): 要傳遞的訊息列表。
       temperature (float): 溫度參數，用於控制模型的回應隨機性。
       ignore_eos (bool): 是否忽略 EOS 標誌。
   Returns:
       dict: API 回應的 JSON 結果。
   """
   headers = {
       "Content-Type": "application/json"
   }
   data = {
       "messages": messages,
       "temperature": temperature,
       "model": model_name,
       "ignore_eos": ignore_eos
   }
   #start_time = time.time()  # 記錄開始時間
   response = requests.post(url=base_url, data=json.dumps(data), headers=headers)
   #print(f"Request duration: {time.time() - start_time:.2f} seconds")
   
   return response.json()
#response_json = get_chat_response(BASE_URL, MODEL_NAME, messages)
#print(response_json)
def extract_content_list(response_json):
   """
   從 response_json 中提取 message 裡 content 的 JSON list。
   Args:
       response_json (dict): 包含回應資料的 JSON 字典。
   Returns:
       list: content 中的 JSON list 值。
   """
   # 獲取 content 字串
   content = response_json.get('choices', [])[0].get('message', {}).get('content', '')
   # 去掉包裹的反引號和 "```json" 字樣，並轉換為 Python 列表
   content_json = content.strip('```').replace('json\n', '').strip()
   # 將 JSON 字串轉換為 Python 列表
   return json.loads(content_json)
def encode_text(texts, tokenizer, model):
   """
   將文本編碼為嵌入向量
   :param texts: 文本列表
   :param tokenizer: 編碼器的 tokenizer
   :param model: 編碼器模型
   :return: 嵌入向量 (torch.Tensor)
   """
   inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
   with torch.no_grad():
       outputs = model(**inputs)
   return outputs.last_hidden_state[:, 0, ]  # 返回 [CLS] token 的嵌入
##user add input function
#a = input('asfdhuogahsdiough: ')
#print(a)
#result = extract_content_list(response_json)
#print(result)
# 初始化模型（此處使用本地模型路徑）
local_model_path = "your_local_model_path"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)
# 使用範例
BASE_URL = 'your_base_url'
MODEL_NAME = "your_model_name"
schema_messages = [
    {
        "role": "user", 
        "content": "Given a large table regarding “company employee_name and wealth and extension_number data”, I want to answer a question: “What is the wealth of mei and claire?” Since I cannot view the table directly, please suggest some column names that might contain the necessary data to answer this question. Please answer with a list of column names in JSON format without any additional explanation.Example:[column1, column2, column3] "
    }
]
cell_messages = [
    {
        "role": "user", 
        "content": "Given a large table regarding “company employee_name and wealth and extension_number data”, I want to answer a question: “What is the wealth of mei and claire?” Please extract some keywords which might appear in the table cells and help answer the question. The keywords should be categorical values rather than numerical values. The keywords should be contained in the question. Please answer with a list of keywords in JSON format without any additional explanation.Don't answer the column name of the table,such as extension_number and employee_name. Example: [keyword1, keyword2, keyword3] "
    }
]
# 載入表格
excel_file = 'your_path_to_excel_file'
table = pd.read_excel(excel_file)
# 載入表格
# 將 DataFrame 轉換為 JSON 格式
#table_as_json = table.to_json(orient="split", force_ascii=False)
# Step 1: Schema Retrieval - 找到相關列名
schema_query = extract_content_list(get_chat_response(BASE_URL, MODEL_NAME, schema_messages))
#print(schema_query)
table_schema = list(table.columns)  # 假設表格的列名稱為 ["name", "english_name", "extension_number"]
print(table_schema)
schema_embeddings = encode_text(table_schema, tokenizer, model)
# Step 2: 計算每個 cell_query 的相似度，逐一處理
similarity_scores_sum1 = torch.zeros(len(schema_embeddings))  # 初始化相似度累加張量
for query in schema_query:
   query_embedding = encode_text([query], tokenizer, model)  # 為每個 query 單獨編碼
   similarity_scores1 = torch.nn.functional.cosine_similarity(query_embedding, schema_embeddings, dim=-1)
   similarity_scores_sum1 += similarity_scores1  # 累加相似度
# Step 3: 找到最相關的列
schema_query_relevant_column_index = similarity_scores_sum1.argmax().item()
schema_query_relevant_column = table_schema[schema_query_relevant_column_index]
# Step 4: 輸出結果
print(f"schema_query_Relevant Column: {schema_query_relevant_column}")
# query_embedding = encode_text(schema_query, tokenizer, model)
# # 計算相似度
# similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, schema_embeddings, dim=-1)
# relevant_column_index = similarity_scores.argmax().item()
# relevant_column = table_schema[relevant_column_index]
#print(f"Relevant Column: {relevant_column}")
# 取得 dtype
schema_query_relevant_column_dtype = table[schema_query_relevant_column].dtype
# 隨機選取三個值作為 cell_examples
cell_examples = random.sample(table[schema_query_relevant_column].dropna().astype(str).tolist(), min(3, len(table[schema_query_relevant_column])))
# 輸出 Schema Retrieval 結果
schema_retrieval_results = {
   "column_name": schema_query_relevant_column,
   "dtype": str(schema_query_relevant_column_dtype),
   "cell_examples": cell_examples
}
print(f"Schema Retrieval Results: {json.dumps(schema_retrieval_results, ensure_ascii=False)}")
# Step 2: Cell Retrieval - 找到目標行索引
cell_query = extract_content_list(get_chat_response(BASE_URL, MODEL_NAME, cell_messages))  # 查詢的目標值，例如人名
#print(cell_query)
# 格式化輸出 Cell Retrieval Queries
cell_query_output = f"Cell Retrieval Queries: {', '.join(cell_query)}"
print(cell_query_output) # output will be "Cell Retrieval Queries: claire, mei"
#print(cell_query)
#cell_query = ['claire','mei']
# Step 1: 為表格的每個列構建嵌入，包括列名和部分數據
column_with_all_values = [
   f"{col}: {', '.join(map(str, table[col].dropna().astype(str).tolist()))}" for col in table_schema
]
schema_embeddings_with_all_values = encode_text(column_with_all_values, tokenizer, model)
#print(column_with_all_values)
'''
cell_query_embedding = encode_text(cell_query, tokenizer, model)
# 計算相似度
similarity_scores = torch.nn.functional.cosine_similarity(cell_query_embedding, schema_embeddings_with_all_values, dim=-1)
cell_query_relevant_column_index = similarity_scores.argmax().item()
#print(cell_query_relevant_column_index)
cell_query_relevant_column = table_schema[cell_query_relevant_column_index]
print(f"cell_query_Relevant Column: {cell_query_relevant_column}") 
'''
# Step 2: 計算每個 cell_query 的相似度，逐一處理
similarity_scores_sum = torch.zeros(len(schema_embeddings_with_all_values))  # 初始化相似度累加張量
for query in cell_query:
   query_embedding = encode_text([query], tokenizer, model)  # 為每個 query 單獨編碼
   similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, schema_embeddings_with_all_values, dim=-1)
   similarity_scores_sum += similarity_scores  # 累加相似度
# Step 3: 找到最相關的列
cell_query_relevant_column_index = similarity_scores_sum.argmax().item()
cell_query_relevant_column = table_schema[cell_query_relevant_column_index]
# Step 4: 輸出結果
print(f"cell_query_Relevant Column: {cell_query_relevant_column}")
cell_embeddings = encode_text(table[cell_query_relevant_column].astype(str).tolist(), tokenizer, model)
# 計算每個查詢人名的相似度，並找到最相關的行
results = {}
for query in cell_query:
   query_embedding = encode_text([query], tokenizer, model)
   #query_embedding = query_embeddings[i].unsqueeze(0)  # 獲取該人名的嵌入向量
   similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, cell_embeddings, dim=-1)
   relevant_row_index = similarity_scores.argmax().item()
   relevant_row = table.iloc[relevant_row_index]
   results[query] = relevant_row[schema_query_relevant_column]
'''
# Step 3: 輸出結果
for person, extension in results.items():
   print(f"{person}: {extension}")
'''   
# Step 3: 將結果格式化
Cell_retrieval_results = {
   "column_name": cell_query_relevant_column,
   "cell_value": ", ".join([f"{person}: {extension}" for person, extension in results.items()])
}
# Step 4: 輸出格式化的結果
print(f"Cell Retrieval Results: {json.dumps(Cell_retrieval_results, ensure_ascii=False)}")
# 定義初始 user_input
user_input = None
# 定義初始 Prompt 模板
prompt_template = """
You are working with a pandas dataframe regarding "company employee_name and wealth and extension_number data" in Python. The name of the dataframe is ‘df‘. Your task is to use ‘python_repl_ast‘ to answer the question: "What is the wealth of mei and claire?" Tool description: - ‘python_repl_ast‘: A Python interactive shell. Use this to execute python commands. Input should be a valid single line python command. Since you cannot view the table directly, here are some schemas and cell values retrieved from the table.
Schema Retrieval Results: {schema_retrieval}
Cell Retrieval Results: {cell_retrieval}
user input: {user_input}
Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the single line Python command to execute
Notes:
- Do not output "Observation" or "Final Answer" until the user input is provided in "user input".
- If "user input" is None, provide only the Thought and Action without completing the Observation or Final Answer.
- If "user input" is provided, then see if "user input" is sufficient to answer the question, if sufficient, then only output the "Final Answer", don't output "Thought" or "Action" or "Observation"
"""
# 初始化第一次請求的 Prompt
current_prompt = prompt_template.format(
   user_input=user_input,
   schema_retrieval=json.dumps(schema_retrieval_results, ensure_ascii=False),
   cell_retrieval=json.dumps(Cell_retrieval_results, ensure_ascii=False)
)
# 初始化迴圈條件
final_answer_found = False
while not final_answer_found:
   # Step 1: 發送 API 請求
   solver_messages = [
       {
           "role": "user",
           "content": current_prompt
       }
   ]
   response = get_chat_response(BASE_URL, MODEL_NAME, solver_messages)
   # 提取 LLM 回應內容
   content = response.get('choices', [])[0].get('message', {}).get('content', '')
   # 輸出 LLM 回應
   print(content)
   # 檢查是否包含 "Final Answer"
   if "Final Answer:" in content:
       final_answer_found = True
   else:
       # 如果沒有 "Final Answer"，等待使用者輸入 Observation
       user_input = input("Observation (user input required): ")
       # 更新 Prompt，將使用者輸入插入到模板中
       current_prompt = prompt_template.format(
           user_input=user_input,
           schema_retrieval=json.dumps(schema_retrieval_results, ensure_ascii=False),
           cell_retrieval=json.dumps(Cell_retrieval_results, ensure_ascii=False)
       )
# 最終輸出結果
print("Process completed. Final Answer:")
print(content)
#print(final_answer)
# 提取 content 的部分
#print(final_answer)

