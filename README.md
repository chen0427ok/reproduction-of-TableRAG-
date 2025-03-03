# Table RAG with Qwen Model

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for querying structured table data using a language model. The system consists of a Python script (`table_rag.py`) that processes tabular data and an LLM inference server (`Qwen.sh`) that runs the Qwen2.5-1.5B-Instruct model using vLLM.

## Features
- Uses an LLM to extract relevant schema and cell information from a tabular dataset.
- Implements semantic similarity search using embeddings to find relevant columns and values.
- Leverages a structured query approach to generate answers based on table content.
- Provides a pipeline for schema retrieval, cell retrieval, and response generation.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Docker with NVIDIA runtime enabled
- PyTorch
- Transformers (Hugging Face)
- Pandas
- Requests

### Install Dependencies
```bash
pip install torch transformers pandas requests openpyxl
```

## Usage

### 1. Start the Qwen Model Server
Run the following script to start the Qwen2.5-1.5B-Instruct model using vLLM:

```bash
sudo docker run --runtime nvidia --gpus='"device=1"' \
--mount type=bind,source=your_model_path,target=/media/user/data \
--rm \
-p 7415:8000 \
--ipc=host \
vllm/vllm-openai:v0.5.3.post1  \
--model /media/user/data/Qwen2.5-1.5B-Instruct \
--gpu-memory-utilization 0.95
```

Ensure you replace `your_model_path` with the actual path to your Qwen model.

### 2. Run the Table RAG Script
Modify the script `table_rag.py` with your table file path and model API details:

```python
BASE_URL = 'http://localhost:7415/v1/chat/completions'
MODEL_NAME = "Qwen2.5-1.5B-Instruct"
excel_file = 'your_path_to_excel_file.xlsx'
```

Run the script:
```bash
python table_rag.py
```

## How It Works
1. **Schema Retrieval**: Identifies the most relevant column(s) in the table based on the query.
2. **Cell Retrieval**: Extracts specific values matching the user query using embeddings.
3. **LLM Interaction**: Constructs a structured prompt and iteratively refines the query.
4. **Final Answer Generation**: Outputs the most relevant information from the table.

## Example Query
Suppose we have a table with employee data and we want to know the wealth of "Mei" and "Claire".

Example output:
```json
{
   "column_name": "wealth",
   "cell_value": "Mei: $1,000,000, Claire: $500,000"
}
```

## Notes
- The script requires a local LLM server running (via Docker) to process queries.
- Ensure that the table schema is structured properly for better retrieval results.

## License
This project is licensed under the MIT License.

