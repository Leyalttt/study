import re
import random
import time

"""
介绍正则表达式的常用操作
"""

# re.match(pattern, string) 验证字符串起始位置是否与pattern匹配
print(re.match('北医[一二三]院', '北医三院怎么走'))  # 在起始位置匹配  <re.Match object; span=(0, 4), match='北医三院'>
# w+ 多个w
print(re.match('w+', 'www.runoob.com'))  # <re.Match object; span=(0, 3), match='www'>

# # re.search(pattern, string) 验证字符串中是否与有片段与pattern匹配 # 不在起始位置匹配
print(re.search('www', 'www.runoob.com'))  # <re.Match object; span=(0, 3), match='www'>
print(re.search('run', 'www.runoob.com'))  # <re.Match object; span=(4, 7), match='run'>
print(re.search('北京|上海', 'www.run上海oob.com'))  # <re.Match object; span=(7, 9), match='上海'>

# #pattern中加括号，可以实现多个pattern的抽取
# .任意字符, * 任意多个
line = "Cats are smarter than dogs"
matchObj = re.match(r'(.*) are (.*?) .*', line)
if matchObj:
    print("matchObj.group() : ", matchObj.group())  # 返回完整匹配的字符串（忽略捕获组 Cats are smarter than dogs
    print("matchObj.group(1) : ", matchObj.group(1))  # 返回第一个捕获组（即第一个 () 内的内容） Cats
    print("matchObj.group(2) : ", matchObj.group(2))  # 返回第二个捕获组 smarter
else:
    print("No match!!")

###########################################

# re.sub(pattern, repl, string, count=0) 利用正则替换文本
# 将string中匹配到pattern的部分，替换为repl, 相当于数据清洗
phone = "2004-959-559#这是一个国外电话号码"
# # 删除字符串中的 # 后注释, $代表结尾
num = re.sub('#.*$', "", phone)
print("电话号码是: ", num)  # 2004-959-559
# # 删除非数字(-)的字符串  \D 代表非数字  \d 代表数字  脱敏
num = re.sub('\d', "*", phone)
print("电话号码是 : ", num)  # ****-***-***#这是一个国外电话号码


# repl 参数可以是一个函数,要注意传入的参数不是值本身，是match对象
# 将匹配的数字乘以 2
def double(matched):
    print('matched', matched)
    return str(int(matched.group()) * 2)


string = 'A23G4HFD567'
print(re.sub('\d', double, string))  # A46G8HFD101214

# count参数决定替换几次，默认是全部替换
string = "00000"
print(re.sub("0", "1", string, count=2))  # 11000

#############################

# re.findall(string[, pos[, endpos]])
# 在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表
pattern = re.compile('\d+')  # 查找数字
result1 = pattern.findall('runoob 123 google 456')
result2 = pattern.findall('run88oob123google456', 0, 10)
print(result1)  # ['123', '456']
print(result2)  # ['88', '12']

print(re.findall("北京|上海|广东", "我从北京去上海"))  # ['北京', '上海']

#################################

# re.split(pattern, string[, maxsplit=0]) 照能够匹配的子串将字符串分割后返回列表
string = "1、不评价别人; 2、不给别人建议; 3、没有共同利益,不必追求共识"
print(re.split("\d、", string))  # 任意数字+、 ['', '不评价别人; ', '不给别人建议; ', '没有共同利益,不必追求共识']
print(re.split(";|、", string))  # ['1', '不评价别人', ' 2', '不给别人建议', ' 3', '没有共同利益,不必追求共识']

###############################
# 匹配汉字  汉字unicode编码范围[\u4e00-\u9fa5]
print(re.findall("[\u4e00-\u9fa5]", "ad噶是的12范德萨发432文"))  # ['噶', '是', '的', '范', '德', '萨', '发', '文']

###############################
# 如果需要匹配，在正则表达式中有特殊含义的符号，需做转义
print(re.search("(图)", "贾玲成中国影史票房最高女导演(图)").group())  # 图
print(re.search("\(图\)", "贾玲成中国影史票房最高女导演(图)").group())  # (图)
print(re.sub("(图)", "", "贾玲成中国影史票房最高女导演(图)"))  # 贾玲成中国影史票房最高女导演()
print(re.sub("\(图\)", "", "贾玲成中国影史票房最高女导演(图)"))  # 贾玲成中国影史票房最高女导演

################################
pattern = "\d12\w"
re_pattern = re.compile(pattern)
print(re.search(pattern, "432312d"))  # <re.Match object; span=(3, 7), match='312d'>


# # 效率
import time
import random

chars = list("abcdefghijklmnopqrstuvwxyz")
# 随机生成长度为n的字母组成的字符串
string = "".join([random.choice(chars) for i in range(100)])
pattern = "".join([random.choice(chars) for i in range(4)])
re_pattern = re.compile(pattern)
start_time = time.time()
for i in range(50000):
    # pattern = "".join([random.choice(chars) for i in range(3)])
    # re.search(pattern, string)
    re.search(re_pattern, string)
print("正则查找耗时：", time.time() - start_time)

start_time = time.time()
for i in range(50000):
    # pattern = "".join([random.choice(chars) for i in range(3)])
    pattern in string
print("python in查找耗时：", time.time() - start_time)
