conda create --name dp1 python=3.10
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers openmind  datasets peft bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U swanlab -i https://pypi.tuna.tsinghua.edu.cn/simple

huggingface-cli download chaiting/conversation-02 --repo-type dataset --local-dir ./



python finetune-multi-args.py \
    --model_name_or_path /root/model/deepseek-ai/deepseek-llm-7b-base \
    --output_dir ./train \
    --experiment_name chai-epoch_10_base_2 \
    --train_file /root/cleaned_session-02_complex_quality_20.jsonl \
    --max_seq_length 2048 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3

python3 -m fastchat.serve.cli --model-path /root/model/deepseek-ai/deepseek-llm-7b-base --conv-template one_shot