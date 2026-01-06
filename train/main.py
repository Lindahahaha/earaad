# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This package is adopted based on Pytorch Lightning Template project.
# Author: Yuanming Zhang

"""This main entrance of the whole project.

Most of the code should not be changed, please directly
add all the input arguments of your model's constructor
and the dataset file's constructor. The MInterface and
DInterface can be seen as transparent to all your args.
"""
import os
import sys
import numpy as np

os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch

from superhuge.utils.multi_run_cli import MultiRunCLI
from superhuge.utils.pick_model_config import pick_file

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
    config_path = os.path.join(project_path, "configs")
    
    # 检查是否有命令行参数输入
    if len(sys.argv) > 1:
        # 使用命令行参数
        # 注意：sys.argv[1:] 包含了你输入的 --task_config 等所有参数
        cli = MultiRunCLI(*sys.argv[1:])
    else:
        # 没有参数时，回退到默认的硬编码配置
        model_config = os.path.join(project_path, "configs", "models", "simple_cnn.yaml")
        cli = MultiRunCLI(
            "fit",
            "--task_config",
            os.path.join(config_path, "task_config.yaml"),
            "--config",
            os.path.join(config_path, "config.yaml"),
            "--model",
            model_config,
        )
    
    # 关键步骤：清空 sys.argv，防止 LightningCLI 报错 "both args and command line arguments"
    # MultiRunCLI 已经内部保存了参数，这里清空不会影响执行
    sys.argv = [sys.argv[0]]

    cli.run()