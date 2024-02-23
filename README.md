# ComfyUI OOTDiffusion

A ComfyUI custom node that simply integrates the [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) functionality.

一个简单接入 OOTDiffusion 的 ComfyUI 节点。

![](./assets/graph.png)

## Instruction 指南

根据 https://git-lfs.com 安装 git lfs：

Linux:

```txt
sudo apt install git-lfs
```

git lfs 初始化：

```txt
git lfs install
```

拉取 Huggingface 库至 ComfyUI 根目录下的 `models/OOTDiffusion` 目录：

```txt
git clone https://huggingface.co/levihsu/OOTDiffusion models/OOTDiffusion
```

拉取 Huggingface 时大约会下载 8 个模型，假如断开连接，可以使用下面命令恢复下载：

```txt
cd models/OOTDiffusion
git lfs fetch
git checkout main
```

创建环境并下载依赖：

```txt
conda create -n ootd
conda activate ootd
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 修复潜在的一些问题
conda install cudatoolkit-dev -c conda-forge

# 安装此项目的依赖
pip install -r custom_nodes/ComfyUI-OOTDiffusion/requirements.txt
```

启动 ComfyUI 即可。

## 节点

LoadOOTDPipeline: 加载 OOTD Pipeline

OOTDGenerate: 生成图像

## 示例图片

[衣服 1](./assets/cloth_1.jpg)

[模特 1](./assets/model_1.png)

## 注意事项

目前此项目只是对 OOTDiffusion 的功能做了个简单的迁移。
OOTDiffusion 本体依赖于 `diffusers==0.24.0` 实现，所以假如有其他节点的依赖冲突是没办法解决的（本就不该依赖 diffusers）。

在 `Ubuntu 22.02` / `Python 3.10.x` 下可以正常运行。Windows 没有测试过。
