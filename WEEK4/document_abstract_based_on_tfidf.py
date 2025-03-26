import jieba
import re
import json
from calculate_tfidf import calculate_tfidf, tf_idf_topk

"""
基于tfidf实现简单文本摘要
"""

jieba.initialize()


# 加载文档数据（可以想象成网页数据），计算每个网页的tfidf字典
def load_data(file_path):
    corpus = []
    with open(file_path, encoding="utf8") as f:
        documents = json.loads(f.read())
        for document in documents:
            # 对于每个文档，检查其标题和内容是否包含换行符\n，如果不包含，则将标题和内容连接起来
            assert "\n" not in document["title"]
            assert "\n" not in document["content"]
            corpus.append(document["title"] + "\n" + document["content"])
        # print('corpus', corpus)  # corpus ['叠穿：越多穿越瘦是可能的（组图）\n编者按：今年冬季让叠穿打造你的完美身材，
        tf_idf_dict = calculate_tfidf(corpus)
        # print('tf_idf_dict', tf_idf_dict)  # {'换发': 0.01559446501768832, '金融': 0.021522636267003663}

    return tf_idf_dict, corpus


# 计算每一篇文章的摘要
# 输入该文章的tf_idf词典，和文章内容
# top为人为定义的选取的句子数量
# 过滤掉一些正文太短的文章，因为正文太短在做摘要意义不大
def generate_document_abstract(document_tf_idf, document, top=3):
    # print('document_tf_idf', document_tf_idf) # {'哈勃': 0.1773145089919276, '望远镜': 0.047518512433511725, '重新': 0.030562331283566738,
    # 使用正则表达式 re.split("？|！|。", document) 将文档拆分为句子列表 sentences
    sentences = re.split("？|！|。", document)
    # print('sentences', sentences) # 一个文章里所有句子
    # 过滤掉正文在五句以内的文章
    if len(sentences) <= 5:
        return None
    result = []
    for index, sentence in enumerate(sentences):
        sentence_score = 0
        # 对句子进行分词
        words = jieba.lcut(sentence)
        # print('words', words)  #  ['一般', '情况', '下', '为', ':', '影院', '50%', '左右',
        for word in words:
            # 每个词，如果它在document_tf_idf字典中存在，则将其对应的值加到sentence_score上
            # word一个分词
            sentence_score += document_tf_idf.get(word, 0)
        sentence_score /= (len(words) + 1)
        result.append([sentence_score, index])
    result = sorted(result, key=lambda x: x[0], reverse=True)
    # 权重最高的可能依次是第10，第6，第3句，将他们调整为出现顺序比较合理，即3,6,10
    important_sentence_indexs = sorted([x[1] for x in result[:top]])
    return "。".join([sentences[index] for index in important_sentence_indexs])


# 生成所有文章的摘要
def generate_abstract(tf_idf_dict, corpus):
    res = []
    for index, document_tf_idf in tf_idf_dict.items():
        title, content = corpus[index].split("\n")
        abstract = generate_document_abstract(document_tf_idf, content)
        # print("abstract", abstract)  # ['编者按：今年冬季让叠穿打造你的完美身材，越穿越瘦是可能', '哪怕是不了解流行，也要对时尚又显瘦的叠穿造型吸引，现在就开始行动吧', '搭配Tips：亮红色的皮外套给人光彩夺目的感觉，内搭短版的黑色T恤，露出带有线条的腹部是关键，展现你健美的身材', ' 搭配Tips：简单款型的机车装也是百搭的单品，内搭一条长版的连衣裙打造瘦身的中性装扮', '软硬结合的mix风同样备受关注', ' 搭配Tips：贴身的黑色装最能达到瘦身的效果，即时加上白色的长外套也不会发福', '长款的靴子同样很好的修饰了你的小腿线条', ' 搭配Tips：高腰线的抹胸装很有拉长下身比例的效果，A字形的荷叶摆同时也能掩盖腰部的赘肉', '外加一件短款的羽绒服，配上贴腿的仔裤，也很修长', '']
        if abstract is None:
            continue
        corpus[index] += "\n" + abstract
        res.append({"标题": title, "正文": content, "摘要": abstract})
    return res


if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, corpus = load_data(path)
    res = generate_abstract(tf_idf_dict, corpus)
    # print('res', res)
    writer = open("abstract.json", "w", encoding="utf8")
    writer.write(json.dumps(res, ensure_ascii=False, indent=2))
    writer.close()
