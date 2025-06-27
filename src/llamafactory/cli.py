# Copyright 2025 the LlamaFactory team.
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

import os
import subprocess
import sys
from copy import deepcopy
from functools import partial


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"  # 启动一个OpenAI风格的API服务器
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"  # 启动一个CLI风格的聊天界面
    + "|   llamafactory-cli eval -h: evaluate models                        |\n"  # 评估模型
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"  # 合并LoRA适配器并导出模型
    + "|   llamafactory-cli train -h: train models                          |\n"  # 训练模型
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"  # 启动一个WebUI风格的聊天界面
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"  # 启动LlamaBoard webui
    + "|   llamafactory-cli version: show version info                      |\n"  # 显示版本信息
    + "-" * 70
)


def main():
    from . import launcher
    from .api.app import run_api
    from .chat.chat_model import run_chat
    from .eval.evaluator import run_eval
    from .extras import logging
    from .extras.env import VERSION, print_env
    from .extras.misc import find_available_port, get_device_count, is_env_enabled, use_ray
    from .train.tuner import export_model, run_exp
    from .webui.interface import run_web_demo, run_web_ui

    logger = logging.get_logger(__name__)

    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to LLaMA Factory, version {VERSION}"
        + " " * (21 - len(VERSION))
        + "|\n|"
        + " " * 56
        + "|\n"
        + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
        + "-" * 58
    )

    COMMAND_MAP = {
        "api": run_api,
        "chat": run_chat,
        "env": print_env,
        "eval": run_eval,
        "export": export_model,
        "train": run_exp,
        "webchat": run_web_demo,
        "webui": run_web_ui,
        "version": partial(print, WELCOME),
        "help": partial(print, USAGE),
    }

    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"  # 获取第一个参数，如果没有则使用help
    if command == "train" and (is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray())):  # 如果命令是train，并且启用了FORCE_TORCHRUN或者设备数量大于1且没有使用ray
        # launch distributed training
        nnodes = os.getenv("NNODES", "1")  # 获取NNODES环境变量，如果没有则使用1
        node_rank = os.getenv("NODE_RANK", "0")  # 获取NODE_RANK环境变量，如果没有则使用0
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))  # 获取NPROC_PER_NODE环境变量，如果没有则使用设备数量
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")  # 获取MASTER_ADDR环境变量，如果没有则使用127.0.0.1
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))  # 获取MASTER_PORT环境变量，如果没有则使用一个可用的端口
        logger.info_rank0(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")  # 打印初始化分布式任务的信息
        if int(nnodes) > 1:
            logger.info_rank0(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")  # 打印多节点训练的信息

        # elastic launch support
        max_restarts = os.getenv("MAX_RESTARTS", "0")  # 获取MAX_RESTARTS环境变量，如果没有则使用0
        rdzv_id = os.getenv("RDZV_ID")  # 获取RDZV_ID环境变量
        min_nnodes = os.getenv("MIN_NNODES")  # 获取MIN_NNODES环境变量
        max_nnodes = os.getenv("MAX_NNODES")  # 获取MAX_NNODES环境变量

        env = deepcopy(os.environ)  # 复制环境变量
        if is_env_enabled("OPTIM_TORCH", "1"):  # 如果启用了OPTIM_TORCH环境变量
            # optimize DDP, see https://zhuanlan.zhihu.com/p/671834539
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 设置PYTORCH_CUDA_ALLOC_CONF环境变量
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"  # 设置TORCH_NCCL_AVOID_RECORD_STREAMS环境变量

        if rdzv_id is not None:  # 如果RDZV_ID环境变量不为空
            # launch elastic job with fault tolerant support when possible
            # see also https://docs.pytorch.org/docs/stable/elastic/train_script.html
            rdzv_nnodes = nnodes  # 设置RDZV_NNODES环境变量
            # elastic number of nodes if MIN_NNODES and MAX_NNODES are set
            if min_nnodes is not None and max_nnodes is not None:  # 如果MIN_NNODES和MAX_NNODES环境变量不为空
                rdzv_nnodes = f"{min_nnodes}:{max_nnodes}"  # 设置RDZV_NNODES环境变量

            process = subprocess.run(
                (
                    "torchrun --nnodes {rdzv_nnodes} --nproc-per-node {nproc_per_node} "
                    "--rdzv-id {rdzv_id} --rdzv-backend c10d --rdzv-endpoint {master_addr}:{master_port} "
                    "--max-restarts {max_restarts} {file_name} {args}"
                )
                .format(
                    rdzv_nnodes=rdzv_nnodes,
                    nproc_per_node=nproc_per_node,
                    rdzv_id=rdzv_id,
                    master_addr=master_addr,
                    master_port=master_port,
                    max_restarts=max_restarts,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )
        else:
            # NOTE: DO NOT USE shell=True to avoid security risk
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=nnodes,
                    node_rank=node_rank,
                    nproc_per_node=nproc_per_node,
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split(),
                env=env,
                check=True,
            )

        sys.exit(process.returncode)
    elif command in COMMAND_MAP:
        COMMAND_MAP[command]()
    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
