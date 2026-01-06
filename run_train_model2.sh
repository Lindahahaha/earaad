#!/bin/bash
#SBATCH --job-name=ear_vcnn      # 【修改1】改个名字，方便和上一个区分
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=8        
#SBATCH --mem=32G                
#SBATCH --output=train_log_vcnn_%j.out # 【修改2】日志文件名也改一下，加上模型标识
#SBATCH --error=train_err_vcnn_%j.err  

# 1. 加载环境
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate superhuge

# 2. 打印信息
echo "Start running Model 2..."
echo "Date: $(date)"
echo "Directory: $(pwd)"

# 3. 跳转目录
cd train

# 4. 运行训练命令
python -u main.py \
      --task_config configs/task_config.yaml \
      --data configs/data_config.yaml \
      --model configs/models/lsm_cnn.yaml \
      --trainer configs/trainer_config.yaml

# 【修改3】一定要改成你实际存在的另一个模型配置文件的路径！

echo "Training finished."