from transformers import GPT2LMHeadModel, BertTokenizer, GPT2Config
import torch

model_path = r"C:\Users\Dell\PycharmProjects\NLP（1）\BERT&GPT\gpt2-chinese-local"

# 强制设置GPT2标准配置
config = GPT2Config.from_pretrained(model_path)
config.update({
    "bos_token_id": 101,
    "eos_token_id": 102,
    "pad_token_id": 0,  # 显式设置PAD位置
    "max_length": 150    # 配置最大生成长度
})

# 初始化分词器并修复特殊符号
tokenizer = BertTokenizer.from_pretrained(
    model_path,
    do_lower_case=False,
    tokenize_chinese_chars=True
)
# 添加缺失的符号映射
tokenizer.add_special_tokens({
    'bos_token': '[CLS]',
    'eos_token': '[SEP]',
    'pad_token': '[PAD]'
})

# 加载模型并调整词表尺寸
model = GPT2LMHeadModel.from_pretrained(
    model_path,
    config=config,
    ignore_mismatched_sizes=True
)
model.resize_token_embeddings(len(tokenizer))  # 关键步骤：对齐词表

# 生成策略参数（经过严格测试的组合）
generation_config = {
    "max_new_tokens": 100,          # 控制新增文本长度
    "temperature": 0.95,            # 提高创造性
    "top_k": 50,                    # 限制候选词数量
    "top_p": 0.92,                  # 动态筛选范围
    "repetition_penalty": 1.5,      # 增强重复惩罚
    "do_sample": True,              # 必须启用采样
    "no_repeat_ngram_size": 3,      # 禁止3词重复
    "num_beams": 3,                 # 减小搜索束宽
    "early_stopping": True
}

# 编码输入并生成
input_text = "我走进了那扇从未打开过的门。"
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding='max_length',  # 标准化输入格式
    max_length=50,
    truncation=True
)

with torch.no_grad():
    output = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        **generation_config
    )

# 后处理输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)