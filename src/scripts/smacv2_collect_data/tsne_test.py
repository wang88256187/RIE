from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt

# 0-9的数字数据
digits = load_digits()
embeddings = TSNE().fit_transform(digits.data)  # t-SNE降维，默认降为二维
vis_x = embeddings[:, 0]  # 0维
vis_y = embeddings[:, 1]  # 1维

index0 = [i for i in range(len(digits.target)) if digits.target == 0]
index1 = [i for i in range(len(digits.target)) if digits.target == 1]
index2 = [i for i in range(len(digits.target)) if digits.target == 2]
index3 = [i for i in range(len(digits.target)) if digits.target == 3]
index4 = [i for i in range(len(digits.target)) if digits.target == 4]
index5 = [i for i in range(len(digits.target)) if digits.target == 5]
index6 = [i for i in range(len(digits.target)) if digits.target == 6]
index7 = [i for i in range(len(digits.target)) if digits.target == 7]
index8 = [i for i in range(len(digits.target)) if digits.target == 8]
index9 = [i for i in range(len(digits.target)) if digits.target == 9]

colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', 'yellow', 'yellowgreen', 'wheat']
plt.scatter(vis_x[index0], vis_y[index0], c=colors[0], cmap='brg', marker='h', label='0')
plt.scatter(vis_x[index1], vis_y[index1], c=colors[1], cmap='brg', marker='<', label='1')
plt.scatter(vis_x[index2], vis_y[index2], c=colors[2], cmap='brg', marker='x', label='2')
plt.scatter(vis_x[index3], vis_y[index3], c=colors[3], cmap='brg', marker='.', label='3')
plt.scatter(vis_x[index4], vis_y[index4], c=colors[4], cmap='brg', marker='p', label='4')
plt.scatter(vis_x[index5], vis_y[index5], c=colors[5], cmap='brg', marker='>', label='5')
plt.scatter(vis_x[index6], vis_y[index6], c=colors[6], cmap='brg', marker='^', label='6')
plt.scatter(vis_x[index7], vis_y[index7], c=colors[7], cmap='brg', marker='d', label='7')
plt.scatter(vis_x[index8], vis_y[index8], c=colors[8], cmap='brg', marker='s', label='8')
plt.scatter(vis_x[index9], vis_y[index9], c=colors[9], cmap='brg', marker='o', label='9')

plt.title(u't-SNE')
plt.legend()
plt.show()
