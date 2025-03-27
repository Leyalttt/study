import jieba
import math
import os
import json
from collections import defaultdict
from calculate_tfidf import calculate_tfidf, tf_idf_topk
"""
基于tfidf实现简单搜索引擎
"""

# 初始化jieba
jieba.initialize()

#加载文档数据（可以想象成网页数据），计算每个网页的tfidf字典
def load_data(file_path):
    # print('file_path', file_path)  # news.json
    # [
    #   {
    #     "title": "叠穿：越多穿越瘦是可能的（组图）",
    #     "content": "编者按：今年冬季让叠穿打造你的完美身材，越穿越瘦是可能。哪怕是不了解流行，也要对时尚又显瘦的叠穿造型吸引，现在就开始行动吧！搭配Tips：亮红色的皮外套给人光彩夺目的感觉，内搭短版的黑色T恤，露出带有线条的腹部是关键，展现你健美的身材。 搭配Tips：简单款型的机车装也是百搭的单品，内搭一条长版的连衣裙打造瘦身的中性装扮。软硬结合的mix风同样备受关注。 搭配Tips：贴身的黑色装最能达到瘦身的效果，即时加上白色的长外套也不会发福。长款的靴子同样很好的修饰了你的小腿线条。 搭配Tips：高腰线的抹胸装很有拉长下身比例的效果，A字形的荷叶摆同时也能掩盖腰部的赘肉。外加一件短款的羽绒服，配上贴腿的仔裤，也很修长。"
    #   },
    # {}, {},
    #  ]
    corpus = []
    with open(file_path, encoding="utf8") as f:
        documents = json.loads(f.read())
        for document in documents:
            # print('document', document)  # 一个 {}
            corpus.append(document["title"] + "\n" + document["content"])
        # print('len', len(corpus))  # 360
        tf_idf_dict = calculate_tfidf(corpus)
    # print('tf_idf_dict', tf_idf_dict)
    return tf_idf_dict, corpus

def search_engine(query, tf_idf_dict, corpus, top=3):
    # 对输入进行jieba分词
    query_words = jieba.lcut(query)
    res = []
    # 遍历每个词的ifidf
    # doc_id 是第几个
    for doc_id, tf_idf in tf_idf_dict.items():
        # print('tf_idf', tf_idf) # {'徐翀': 0.008063597594550016, '：': 0.0009520255790161628, '网络': 0.03364037516897155, ...}
        score = 0
        # 遍历输入的jieba分词
        for word in query_words:
            score += tf_idf.get(word, 0)
        res.append([doc_id, score])
    # print('res', res)  # [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],...]
    res = sorted(res, reverse=True, key=lambda x:x[1])
    print('res', res)  # [[286, 0.06266923892133364], [43, 0.04452814344410548], [278, 0.03796309665426942]
    for i in range(top):
        doc_id = res[i][0]
        print(corpus[doc_id])
        print("--------------")
    #     WCG2009世界总决赛场馆分布图
    # WCG2009世界总决赛即将于11月11-15日在成都世纪城新国际会展中心（场馆介绍）进行，我们也从主办方拿到了场馆的分布图，本次场馆有3600个座位，容纳5000以上的观众，相信届时在现场将会有十分热闹的场景。
    # --------------
    # 世界上最长的火车线路 沿途风光惊艳世界
    # 这是一场横跨亚欧大陆的旅行，是世界上最长的火车线路。每周三或周六，都会有列车由北京发往莫斯科，别着急抵达目的地，沿途风光绝对令人惊艳！123...10下一页声明：本网站所提供的信息仅供参考之用,并不代表本网赞同其观点，也不代表本网对其真实性负责。您若对该稿件内容有任何疑问或质疑，请尽快与上海热线联系，本网将迅速给您回应并做相关处理。联系方式:shzixun@online.sh.cn本文来源：网易旅游  作者：责任编辑：徐盈
    # --------------
    # 魔兽争霸 Fly2：0华丽推倒对手进级
    # 快讯：继我们遗憾得知Sky离开WCG2009世界总决赛舞台后，中国队终于迎来首场胜利。Fly100%以2：0的比分华丽推倒对手，再进一步。让我们期待Fly100%更好的表现！
    # --------------
    return res

if __name__ == "__main__":
    path = "news.json"
    tf_idf_dict, corpus = load_data(path)
    while True:
        query = input("请输入您要搜索的内容:")
        search_engine(query, tf_idf_dict, corpus)
