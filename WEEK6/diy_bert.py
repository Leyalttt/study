import torch
import math
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer

'''
Bert的使用, 文本分类, 文本匹配都可以用
关于transformers自带的序列化工具
模型文件下载 https://huggingface.co/models

'''
# 加载了一个预训练的 BERT 模型(相当于别人帮我训练好了)
# from_pretrained用于加载预训练模型和分词器。这个函数可以帮助我们轻松地从模型库或本地路径加载已经训练好的模型
# 原始字符串不会处理字符串中的反斜杠\作为转义字符
# 也要使用bert-base-chinese中的vocab的词表和分词方式
# bert = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)
tokenizer = BertTokenizer.from_pretrained(r"D:\TTT\NLP算法\bert-base-chinese\bert-base-chinese")

string = "咱呀么老百姓今儿个真高兴"
# 分字
# tokenize方法将输入字符串分割成一系列的子字符串（tokens），这些子字符串是BERT模型可以理解的词汇单元
tokens = tokenizer.tokenize(string)
print("分字：", tokens)  # ['咱', '呀', '么', '老', '百', '姓', '今', '儿', '个', '真', '高', '兴']

# 这行代码使用分词器对输入字符串进行编码。encode方法将输入字符串转换为一个整数序列，这些整数是BERT模型可以理解的词汇单元的索引
# 编码，前后自动添加了[cls]和[sep] 也就是101 和 102
encoding = tokenizer.encode(string)
print("编码：", encoding)  # [101, 1493, 1435, 720, 5439, 4636, 1998, 791, 1036, 702, 4696, 7770, 1069, 102]

# 文本对编码, 形式[cls] string1 [sep] string2 [sep]
string1 = "今天天气真不错"
string2 = "明天天气怎么样"
encoding = tokenizer.encode(string1, string2)
print("文本对编码：", encoding)  # [101, 791, 1921, 1921, 3698, 4696, 679, 7231, 102, 3209, 1921, 1921, 3698, 2582, 720, 3416, 102]

# 同时输出attention_mask和token_type编码
encoding = tokenizer.encode_plus(string1, string2)
print("全部编码：", encoding)
# {'input_ids': [101, 791, 1921, 1921, 3698, 4696, 679, 7231, 102, 3209, 1921, 1921, 3698, 2582, 720, 3416, 102],=》encode
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], =》 segmentEmbedding区分是第几句
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] =》 1参与运算， 0(padding)不参与运算}
