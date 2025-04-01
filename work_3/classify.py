import re
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words
all_words = []
def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]


top_words = get_top_words(100)
# 构建词-个数映射表
vector = []
for words in all_words:
    '''
    words:
    ['国际', 'SCI', '期刊', '材料', '结构力学', '工程', '杂志', '国际', 'SCI', '期刊', '先进', '材料科学',
    '材料', '工程', '杂志', '国际', 'SCI', '期刊', '图像处理', '模式识别', '人工智能', '工程', '杂志', '国际',
    'SCI', '期刊', '数据', '信息', '科学杂志', '国际', 'SCI', '期刊', '机器', '学习', '神经网络', '人工智能',
    '杂志', '国际', 'SCI', '期刊', '能源', '环境', '生态', '温度', '管理', '结合', '信息学', '杂志', '期刊',
    '网址', '论文', '篇幅', '控制', '以上', '英文', '字数', '以上', '文章', '撰写', '语言', '英语', '论文',
    '研究', '内容', '详实', '方法', '正确', '理论性', '实践性', '科学性', '前沿性', '投稿', '初稿', '需要',
    '排版', '录用', '提供', '模版', '排版', '写作', '要求', '正规', '期刊', '正规', '操作', '大牛', '出版社',
    '期刊', '期刊', '质量', '放心', '检索', '稳定', '邀请函', '推荐', '身边', '老师', '朋友', '打扰', '请谅解']
    '''
    word_map = list(map(lambda word: words.count(word), top_words))
    '''
    word_map:
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    '''
    vector.append(word_map)
vector = np.array(vector)
# 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
labels = np.array([1]*127 + [0]*24)
model = MultinomialNB()
model.fit(vector, labels)
def predict(filename):
    """对未知邮件分类"""
    # 构建未知邮件的词向量
    words = get_words(filename)
    current_vector = np.array(
        tuple(map(lambda word: words.count(word), top_words)))
    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'
print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt')))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt')))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt')))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt')))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt')))


# 采用sklearn.feature_extraction.text.TfidfVectorizer实现TF-IDF值计算
class FeatureSelector(BaseEstimator, TransformerMixin):
    """支持高频词/TF-IDF切换的特征工程类"""

    def __init__(self, mode='high_freq', top_n=100, max_df=0.8, min_df=2):
        self.mode = mode
        self.top_n = top_n
        self.max_df = max_df
        self.min_df = min_df

        # 继承原逻辑的自定义分词器
        def custom_tokenizer(text):
            text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)
            return [word for word in cut(text) if len(word) > 1]

        # 初始化特征提取器（移除token_pattern参数）
        if self.mode == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                tokenizer=custom_tokenizer,
                max_features=self.top_n,
                max_df=self.max_df,
                min_df=self.min_df
            )
        else:
            self.vectorizer = CountVectorizer(
                tokenizer=custom_tokenizer,
                max_features=self.top_n
            )

    def fit(self, raw_documents, y=None):
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        return self.vectorizer.transform(raw_documents)

# 加载邮件原始文本
def load_corpus():
    file_list = [f'邮件_files/{i}.txt' for i in range(151)]
    return [open(f, encoding='utf-8').read() for f in file_list]

# 初始化特征选择器（解决token_pattern警告）
feature_selector = FeatureSelector(
    mode='tfidf',  # 可选'tfidf'或'high_freq'
    top_n=100,
    max_df=0.8,
    min_df=2
)

# 构建特征矩阵
corpus = load_corpus()
X = feature_selector.fit_transform(corpus)
labels = np.array([1] * 127 + [0] * 24)

# 训练模型
model = MultinomialNB()
model.fit(X, labels)

# 预测函数
def predict(filename):
    text = open(filename, encoding='utf-8').read()
    vec = feature_selector.transform([text])
    return '垃圾邮件' if model.predict(vec)[0] == 1 else '普通邮件'

# 测试预测
print('151.txt分类情况:', predict('邮件_files/151.txt'))
print('152.txt分类情况:', predict('邮件_files/152.txt'))
print('153.txt分类情况:', predict('邮件_files/153.txt'))
print('154.txt分类情况:', predict('邮件_files/154.txt'))
print('155.txt分类情况:', predict('邮件_files/155.txt'))


# 样本平衡处理：采用imblearn.over_sampling.SMOTE实现
class FeatureSelector(BaseEstimator, TransformerMixin):
    """支持高频词/TF-IDF切换的特征工程类"""

    def __init__(self, mode='high_freq', top_n=100, max_df=0.8, min_df=2):
        self.mode = mode
        self.top_n = top_n
        self.max_df = max_df
        self.min_df = min_df

        # 自定义分词器（继承原逻辑）
        def custom_tokenizer(text):
            text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)
            return [word for word in cut(text) if len(word) > 1]

        # 初始化特征提取器
        if self.mode == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                tokenizer=custom_tokenizer,
                max_features=self.top_n,
                max_df=self.max_df,
                min_df=self.min_df
            )
        else:
            self.vectorizer = CountVectorizer(
                tokenizer=custom_tokenizer,
                max_features=self.top_n
            )

    def fit(self, raw_documents, y=None):
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        return self.vectorizer.transform(raw_documents)

# 加载邮件原始文本
def load_corpus():
    file_list = [f'邮件_files/{i}.txt' for i in range(151)]
    return [open(f, encoding='utf-8').read() for f in file_list]

# 初始化特征选择器
feature_selector = FeatureSelector(
    mode='tfidf',  # 可选'tfidf'或'high_freq'
    top_n=100,
    max_df=0.8,
    min_df=2
)

# 构建初始特征矩阵
corpus = load_corpus()
X = feature_selector.fit_transform(corpus)
labels = np.array([1] * 127 + [0] * 24)

# 新增SMOTE过采样处理
sm = SMOTE(random_state=42, sampling_strategy={0: 127})  # 将普通邮件扩增到127条
X_resampled, y_resampled = sm.fit_resample(X, labels)

# 训练模型
model = MultinomialNB()
model.fit(X_resampled, y_resampled)  # 使用平衡后的数据集

# 预测函数（保持不变）
def predict(filename):
    text = open(filename, encoding='utf-8').read()
    vec = feature_selector.transform([text])
    return '垃圾邮件' if model.predict(vec)[0] == 1 else '普通邮件'

# 测试预测
print('151.txt分类情况:', predict('邮件_files/151.txt'))
print('152.txt分类情况:', predict('邮件_files/152.txt'))
print('153.txt分类情况:', predict('邮件_files/153.txt'))
print('154.txt分类情况:', predict('邮件_files/154.txt'))
print('155.txt分类情况:', predict('邮件_files/155.txt'))


# 增加模型评估指标：通过sklearn.metrics.classification_report实现多维度模型评估。
from sklearn.metrics import classification_report

# 原代码训练集特征矩阵与标签
X_train = vector[:127]  # 前127条为训练集垃圾邮件
y_train = labels[:127]

# 测试集构造（假设保留最后24条普通邮件+5条测试样本）
X_test = vector[-29:]   # 24普通邮件+5测试邮件
y_test = labels[-29:]

# 模型预测
y_pred = model.predict(X_test)

# 生成分类评估报告
report = classification_report(
    y_test,
    y_pred,
    target_names=['普通邮件', '垃圾邮件'],
    digits=4
)
print("分类评估报告：\n", report)
