# ComfyUI OOTDiffusion

A ComfyUI custom node that simply integrates the OOTDiffusion functionality.

一个简单接入 OOTDiffusion 的 ComfyUI 节点。

## Instruction 指南

根据 https://git-lfs.com 安装 git lfs：

```txt
sudo apt install git-lfs
git lfs install
```

拉取 Huggingface 库至 `models/OOTDiffusion` 目录：

```txt
git clone https://huggingface.co/levihsu/OOTDiffusion models/OOTDiffusion
```

启动 ComfyUI 即可。

## 节点

LoadOOTDPipeline: 加载 OOTD Pipeline

OOTDGenerate: 生成图像
