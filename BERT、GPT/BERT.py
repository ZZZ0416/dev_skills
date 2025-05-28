from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 定义待分类句子
sentence_movie = "演员表演浮夸，完全无法让人产生代入感。"
sentence_food = "食物完全凉了，吃起来像隔夜饭，体验极差。"

# 加载模型
model_path = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    return "正面" if torch.argmax(probabilities) == 1 else "负面"

# 输出结果
print("影评分类结果：", predict_sentiment(sentence_movie))  
print("外卖评价分类结果：", predict_sentiment(sentence_food))