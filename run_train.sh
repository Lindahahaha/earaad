#!/bin/bash
#SBATCH --job-name=ear           # 任务名称
#SBATCH --gres=gpu:1             # 申请 1 张 GPU
#SBATCH --cpus-per-task=8        # 申请 8 个 CPU
#SBATCH --mem=32G                # 申请 32G 内存
#SBATCH --output=train_log_%j.out # 【关键】将终端输出保存到这个文件 (%j 代表任务ID)
#SBATCH --error=train_err_%j.err  # 将错误报错保存到这个文件

# 1. 加载 Conda 环境
# 注意：有些服务器需要先 source .bashrc 才能使用 conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate superhuge

# 2. 打印一些调试信息（可选）
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Directory: $(pwd)"

# 3. 跳转到 train 目录 (因为你的 main.py 在 train/main.py，且配置文件相对路径通常基于 train 目录)
cd train

# 4. 运行训练命令
# python -u 表示不缓存输出，让日志实时写入文件
python -u main.py \
      --task_config configs/task_config.yaml \
      --data configs/data_config.yaml \
      --model configs/models/simple_cnn.yaml \
      --trainer configs/trainer_config.yaml

echo "Training finished."