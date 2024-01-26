import json
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
# 从HuggingFace Hub加载数据集
rd_ds = load_dataset("data")
rd_df = pd.DataFrame(rd_ds['train'])
# 去掉首尾空白字符
rd_df['instruction'] = rd_df['instruction'].str.strip()
# 获取回车分割的指令描述输入
rd_df['temp'] = rd_df['instruction'].str.split('\n', n=1)
# 获取后面的input
rd_df['input'] = rd_df['temp'].apply(lambda x: x[1] if len(x) > 1 else x)
# 获取前面的指令
rd_df['instruction'] = rd_df['temp'].apply(lambda x: x[0] if len(x) > 1 else x)
# 分割后可能是一个list，所以去list的第一个元素
rd_df['instruction'] = rd_df['instruction'].apply(lambda x: x[0] if isinstance(x, list) else x)
# 删除无用的元素，最后只剩 prompt 和 output
rd_df.drop('temp', axis=1, inplace=True)
rd_df.drop('input_ids', axis=1, inplace=True)
rd_df.drop('attention_mask', axis=1, inplace=True)
rd_df.drop('labels', axis=1, inplace=True)
rd_df.to_csv("data/allData.csv", index=False)