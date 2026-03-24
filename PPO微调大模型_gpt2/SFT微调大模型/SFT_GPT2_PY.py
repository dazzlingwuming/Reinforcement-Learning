import torch

from PPO微调大模型_gpt2.SFT微调大模型.data_processing import load_data
from modelscope import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling




#加载数据
data_path = ["data_raw/train.txt", "data_raw/valid.txt", "data_raw/test.txt"]
data_output_path = ["data_raw/train.jsonl", "data_raw/valid.jsonl", "data_raw/test.jsonl"]
hf_model_path = 'Fengshenbang/Wenzhong-GPT2-110M-chinese-v2'
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
dataset = load_data(data_output_path, tokenizer)
filtered_dataset_train, filtered_dataset_val, _ = load_data(data_output_path, tokenizer)

#填充
print(tokenizer.pad_token)
print(tokenizer.eos_token)
#将填充标记设置为 EOS 标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# mlm=False，将数据整理成“因果语言建模”需要的数据格式,mlm=True，将数据整理成“掩码语言建模”需要的数据格式
# “因果语言建模”就是“预测下一个token”类型的任务，也就是gpt风格的自回归模型

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) # labels

# 这个 collate_fn 会在 DataLoader 中被调用，自动处理批次中的样本，进行填充和标签处理
dataloader_params = {
    'batch_size': 16, #根据需要调整批量大小
    'collate_fn': data_collator
}
# 创建 DataLoader，传入过滤后的数据集和 collate_fn
train_dataloader = DataLoader(filtered_dataset_train, **dataloader_params)
val_dataloader = DataLoader(filtered_dataset_val, **dataloader_params)

#打印一下数据加载器的长度和一个批次的内容
print(len(train_dataloader))
batch = next(iter(train_dataloader))
print(batch.keys())
print(batch['input_ids'].shape)
print(batch['input_ids'][0])
print(batch['labels'][0])
print(batch['attention_mask'][0])

#加载模型监督微调
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained(hf_model_path).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#训练一个epoch
num_epochs = 1


# 训练循环
#先评估一下初始模型在验证集上的表现
def validate(epoch):
    model.eval()
    total_loss = 0.0
    for i, batch in enumerate(val_dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss # 损失
            total_loss += loss.item()
    print(f'val_loss at {epoch} epoch:', total_loss / len(val_dataloader))
validate(1)
#正式训练
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss # 损失
        total_loss += loss.item()
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        print(f'train_loss at {i} epoch:', loss.item() / 16)
    print("train_loss at {epoch} epoch:", total_loss / len(train_dataloader))
    validate(epoch+1)

#保存微调后的模型
model.save_pretrained('models/fine_tuned_gpt2')
tokenizer.save_pretrained('models/fine_tuned_gpt2')






