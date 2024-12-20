Qwen.sh
sudo docker run --runtime nvidia --gpus='"device=1"' \
--mount type=bind,source=your_model_path ,target=/media/user/data \
--rm \
-p 7415:8000 \
--ipc=host \
vllm/vllm-openai:v0.5.3.post1  \
--model /media/user/data/Qwen2.5-1.5B-Instruct \
--gpu-memory-utilization 0.95