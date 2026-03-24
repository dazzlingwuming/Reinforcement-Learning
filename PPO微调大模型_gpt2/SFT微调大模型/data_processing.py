#数据选取
'''
本仓库提供了一个名为`Dataset.rar`的资源文件，该文件包含了中文电影评论数据集。该数据集规模较大，适用于大型模型的训练，同时也适合小型模型使用其中的一部分数据。数据集的质量较高，适合用于自然语言处理、情感分析等任务。
##数据集描述
-**文件名**:`Dataset.rar`
-**内容**:中文电影评论数据集
-**适用范围**:
-大型模型：可直接使用整个数据集进行训练。
-小型模型：可根据需求选择部分数据进行训练。
-**数据质量**:数据集质量较高，适合用于各种自然语言处理任务。
'''
from functools import partial

#构造数据集
'''
原始数据形状
1	如果我无聊时网上乱逛偶尔看到这部电影我可能会给它打四星，但是TNND姐是花大洋去电影院看姐在电影院睡着，姐现在非常心疼电影票钱一部无聊至极电影
标签+文本
数据预处理步骤
1. 数据清洗：去除无关字符、标点符号等。
2. 分词：将文本分割成单词或子词。
3. 去除停用词：去除常见但无意义的词汇。
4. 词干提取或词形还原：将单词还原为基本形式。
'''
import pandas as pd
from datasets import load_dataset

def preprocess_data(data_path,data_output_path):
    # 读取原始数据（无表头）
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'context'])

    # 添加 idx 列（从0开始）
    df['idx'] = range(len(df))

    # 调整列顺序为 idx, context, label
    df = df[['idx', 'context', 'label']]

    # 保存为 JSONL 格式
    df.to_json(data_output_path, orient='records', lines=True, force_ascii=False)


# 定义分词函数：接收一个 batch 字典，返回一个字典（键是新的列名，值是列表）
def tokenize(batch,tokenizer):
    return tokenizer(batch['context'])

#清理数据长度国小的
def filter_short_samples(batch, min_length=10):
    # 计算输入 ID 的长度
    input_ids_length = len(batch['input_ids'])
    # 只有当长度大于或等于 min_length和小于等于100 时才保留样本
    return (input_ids_length >= min_length and input_ids_length <= 150)



def load_data(data_path,tokenizer):
    dataset = load_dataset('json', data_files={
        'train': data_path[0],
        'valid': data_path[1],
        'test': data_path[2]
    })
    # 查看结构
    print(dataset['train'][0])
    # 输出：{'idx': 0, 'context': '如果 我 无聊 时 ...', 'label': 1}

    map_kwargs = {
        'batched': True,  # 启用批处理模式，tokenize 接收的是 batch 字典
        'batch_size': 512,  # 每批 512 个样本
        'remove_columns': ['idx', 'context', 'label']  # 处理后移除这三列
    }

    # 在 map 时绑定 tokenizer
    tokenize_with_tokenizer = partial(tokenize, tokenizer=tokenizer)

    # 对训练集和验证集应用 map
    tokenized_dataset_train = dataset["train"].map(tokenize_with_tokenizer, **map_kwargs)
    tokenized_dataset_val = dataset["valid"].map(tokenize_with_tokenizer, **map_kwargs)

    print(tokenized_dataset_train[0])
    print(tokenized_dataset_val[0])


    # 对训练集和验证集应用过滤函数
    filtered_dataset_train = tokenized_dataset_train.filter(filter_short_samples)
    filtered_dataset_val = tokenized_dataset_val.filter(filter_short_samples)
    print(f"训练集过滤前样本数量: {len(tokenized_dataset_train)}")
    print(f"训练集过滤后样本数量: {len(filtered_dataset_train)}")
    print(f"验证集过滤前样本数量: {len(tokenized_dataset_val)}")
    print(f"验证集过滤后样本数量: {len(filtered_dataset_val)}")

    # 放入torch
    import torch
    filtered_dataset_train.set_format(type='torch')
    filtered_dataset_val.set_format(type='torch')
    print(filtered_dataset_train[0])
    print(filtered_dataset_val[0])

    return filtered_dataset_train, filtered_dataset_val, dataset["test"]






if __name__ == "__main__":
    data_path = ["data_raw/train.txt","data_raw/valid.txt","data_raw/test.txt"]
    data_output_path = ["data_raw/train.jsonl","data_raw/valid.jsonl","data_raw/test.jsonl"]
    # for i in range(3):
    #     preprocess_data(data_path[i],data_output_path[i])
    # load_data('data_raw/test.jsonl')
    from modelscope import GPT2Tokenizer, GPT2LMHeadModel
    hf_model_path = 'Fengshenbang/Wenzhong-GPT2-110M-chinese-v2'
    tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
    dataset = load_data(data_output_path,tokenizer)


