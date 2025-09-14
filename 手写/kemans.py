import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KMeans:
    # k: 聚类数量，默认为3(分成3份), max_iters: 最大迭代次数，默认为100, tol: 收敛阈值，默认为1e-4(表示0.0001。)
    # max_iters K-Means算法通过迭代优化质心的位置，每次迭代都会重新分配点到簇并更新质心, 设置这个参数是为了防止算法在特殊情况下无限循环。
    # tol 在K-Means算法中，每次迭代后质心的位置会发生变化。当质心位置的变化小于这个容差时，我们认为算法已经收敛，可以停止迭代。
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    # 训练
    # X训练数据
    def fit(self, X):
        # 随机选择k个数据点作为初始质心
        # X.shape[0]获取数据点的数量（行数）, np.random.choice从给定范围内随机选择self.k个不重复的索引
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        # print('X.shape[0]', X.shape[0])  # 300
        # print('X.shape', X.shape)  # X.shape (300, 2) 每个数据有两个特征
        # print('indices', indices)  # [ 85 267 226] 从0-299取三个不重复的数据

        centroids = X[indices]  # 三个质心的值, np.array格式
        # print('centroids', centroids)
        # [[ 3.66800921  0.15565258]
        #  [-2.66974818  1.82044485]
        #  [-1.68272799  2.27872681]]

        # 开始迭代
        # 迭代更新质心直到收敛或达到最大迭代次数：
        for _ in range(self.max_iters):
            # 步骤一:将每个点分配给最近的质心，形成簇
            # 创建一个包含self.k个空列表的列表，用于存储每个簇的数据点
            # 每个空列表将对应一个簇，最终存储属于该簇的所有数据点
            clusters = [[] for _ in range(self.k)]
            # print('clusters', clusters)  # [[], [], []]

            # 遍历数据集中的每个数据点
            # features代表当前数据点的所有特征值
            for features in X:
                # 计算当前数据点到所有质心的欧氏距离
                # np.linalg.norm计算向量之间的欧氏距离（即直线距离）
                # 使用列表推导式创建一个距离列表，包含当前点到每个质心的距离
                distances = [np.linalg.norm(features - centroid) for centroid in centroids]
                # print('distances', distances)  # 当前点到三个质心的距离 [3.1852682825454712, 0.5197585606697636, 2.766486152505185]
                # 找到距离当前数据点最近的质心的索引
                # np.argmin返回数组中最小值的索引
                # 这个索引表示当前点应该被分配到哪个簇
                closest_cluster = np.argmin(distances)
                # 将当前数据点添加到最近的簇中
                # closest_cluster是簇的索引，clusters[closest_cluster]是对应的簇列表
                clusters[closest_cluster].append(features)  # 二维数组

            # 步骤2: 计算新的质心, 在所有点分配完成后，即在内层循环外）
            new_centroids = []  # 普通list
            # 遍历每个簇, cluster是当前簇的所有数据点列表
            for cluster in clusters:
                if cluster:  # 非空簇
                    # 检查当前簇是否为空（即是否有数据点被分配到该簇）
                    # 如果簇非空，计算该簇中所有点的均值作为新的质心
                    # np.mean(cluster, axis=0)计算簇中所有点在每个特征维度上的平均值
                    # 将计算出的新质心添加到new_centroids列表中
                    new_centroids.append(np.mean(cluster, axis=0))
                else:  # 空簇，随机重新初始化
                    # 如果当前簇为空（没有数据点被分配到该簇），则随机选择一个数据点作为新的质心
                    # 这是一种处理空簇的策略，确保算法能够继续运行
                    # np.random.choice(X.shape[0]) 只是下标
                    new_centroids.append(X[np.random.choice(X.shape[0])])

            # 步骤3: 检查质心是否变化（在所有点分配完成后）
            # 检查新旧质心是否足够接近，如果是，则算法已收敛，跳出循环
            # np.allclose比较两个数组是否在给定的容差范围内相等
            # atol=self.tol设置绝对容差，即质心位置变化小于这个值时认为收敛
            # 如果质心变化很小，说明算法已经找到了稳定的簇结构，可以提前结束迭代
            if np.allclose(centroids, new_centroids, atol=self.tol):
                break

            # 步骤四:更新质心（在所有点分配完成后）
            # 更新质心为新的质心，准备下一次迭代
            # 将质心和簇保存为实例变量，以便在其他方法（如predict）中使用
            # self.centroids存储最终的质心位置
            # self.clusters存储每个簇包含的数据点
            # 将新质心列表转换为NumPy数组, 初始定义就是np.array
            # 这样可以方便后续的数学运算和比较
            centroids = np.array(new_centroids)
            # print(type(new_centroids))  # <class 'list'>
            # print(type(centroids))  # <class 'numpy.ndarray'>
            self.centroids = centroids
            self.clusters = clusters

    # 预测
    def predict(self, X):
        # 为每个数据点找到最近的质心，返回其所属簇的索引
        # 防止调用predict方法之前没有先调用fit方法
        y_pred = [np.argmin([np.linalg.norm(x - centroid) for centroid in self.centroids]) for x in X]
        # print('y_pred', y_pred)  # y_pred [0, 1, 0, 2, 2, 2, 1, 0, 2, 0, 1, 1, 1...]
        return np.array(y_pred)


# 生成一些示例数据
# make_blobs函数从sklearn.datasets模块生成合成的聚类数据集
# make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0),shuffle=True, random_state=None)
# n_samples指定生成的数据点总数, n_features：默认值= 2,每个样本的特征数量。centers指定要生成的中心点（簇）的数量, cluster_std定每个簇的标准差, random_state = 0 这个参数确保每次运行代码时生成的数据集是相同的
#  X 类型：NumPy数组，形状为(n_samples, n_features) (300, 2)
# y_true 类型：NumPy数组，形状为(n_samples,) 含义：每个数据点所属的真实簇标签
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.70, random_state=0)
# print('X', X)  # [[ 0.33729452  5.08569873] [ 1.54734936 -0.07069111] [ 1.50899649  4.38895984]...]
# print(X.shape)  # (300, 2)
# print(y_true.shape)  # (300,)
# print('y_true', y_true)  #  [0 1 0 2 2 2 1 0 2 2 1 1 ...]

# 可视化原始数据
# 创建一个新的图形窗口（figure）figsize：一个元组，指定图形的宽度和高度（单位是英寸） (15, 5)：宽度15英寸，高度5英寸
plt.figure(figsize=(15, 5))
# 在图形窗口中创建一个子图（subplot）
# 第一个参数1：行数，表示图形窗口有1行子图
# 第二个参数3：列数，表示图形窗口有3列子图
# 第三个参数1：当前子图的索引（从1开始）
plt.subplot(1, 3, 1)
# 在当前子图上绘制散点图
# X[:, 0]：数据点的x坐标
# X是一个二维数组，X[:, 0]表示取所有行的第0列（即第一个特征）
# X[:, 1]：数据点的y坐标
# X[:, 1]表示取所有行的第1列（即第二个特征）
# s=50：点的大小（marker size）
# 数值越大，点越大
# 默认值通常是20，这里设置为50使点更加明显
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("原始数据")

# 使用我们的K-Means算法聚类
kmeans = KMeans(k=3, max_iters=100, tol=1e-4)
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.predict(X)
centroids = kmeans.centroids

# 可视化聚类结果
plt.subplot(1, 3, 2)
# cmap='viridis'：颜色映射（colormap），用于将数字标签转换为颜色,
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# c=labels：点的颜色，根据每个数据点的簇标签决定
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("我们的K-Means结果")
plt.show()
# 打印质心位置对比
print("我们的K-Means质心位置:")
print(centroids)
# [[-1.66756126  2.82778909]
#  [ 0.91840186  4.33954178]
#  [ 1.94745135  0.80883715]]
