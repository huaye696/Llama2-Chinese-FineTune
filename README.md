# Llama2-Chinese-FineTune
本仓库用于使用LoRA对Llama2进行微调的学习，大部分代码均写了注释，供学习参考。
## 数据集
关于数据集，选择HuggingFace中的 [train_0.5M_CN_llama2](https://huggingface.co/datasets/SimonSun/train_0.5M_CN_llama2) 中文数据集，下载到本地，然后再用transformes进行解析。
处理脚本位于[precessData.py](transformers%2FprecessData.py)中。
## 开始训练
修改 [lanch.sh](transformers%2Flanch.sh) 中的详细文本配置即可
~~~shell
bash lanch.sh