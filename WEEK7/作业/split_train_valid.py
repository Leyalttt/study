import random
'''切割训练集和验证集'''
def split_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        # f.readlines() 返回一个包含文件所有行的列表，每一行是一个字符串。[1:] 是一个切片操作，它从列表的第二个元素开始（索引为1）取到列表的最后一个元素
        lines = f.readlines()[1:]
    # 将列表 lines 中的元素随机打乱
    random.shuffle(lines)
    # print('lines', lines)
    num_lines = len(lines)
    # print('num_lines', num_lines)  # 11987
    num_train = int(0.8 * num_lines)

    train_lines = lines[:num_train]
    valid_lines = lines[num_train:]

    with open('train_data.txt', 'w', encoding='utf8') as f_train:
        # writelines 函数会将这些字符串写入文件，每个字符串后面都会自动添加一个换行符
        f_train.writelines(train_lines)

    with open('valid_data.txt', 'w', encoding='utf8') as f_valid:
        f_valid.writelines(valid_lines)

split_file(r'D:\TTT\NLP算法\预习\week7 文本分类问题\week7 文本分类问题\文本分类练习数据集\文本分类练习.csv')
