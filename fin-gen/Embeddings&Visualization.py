import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os


print("----开始加载保存的Embeddings----")
all_embeddings = np.load("all_embeddings.npy")

# (我们也需要原始数据中的 'default' 标签来进行上色，所以重新加载一下df)
import kagglehub
import os
dataset_dir = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")
dataset_path = os.path.join(dataset_dir,"UCI_Credit_Card.csv")
df = pd.read_csv(dataset_path)
df = df.drop("ID",axis = 1)
df = df.rename(columns={'default.payment.next.month': 'default', 'PAY_0': 'PAY_1'})
labels = df['default'].values

num_samples_to_plot = 30000 # 先从10000个样本开始，如果还死机，可以降到5000
if len(all_embeddings) > num_samples_to_plot:
    print(f"--- 数据量较大，随机抽取 {num_samples_to_plot} 个样本进行可视化 ---")
    # 生成随机索引
    random_indices = np.random.choice(len(all_embeddings), num_samples_to_plot, replace=False)
    # 根据索引选择对应的 embeddings 和 labels
    embeddings_subset = all_embeddings[random_indices]
    labels_subset = labels[random_indices]
else:
    embeddings_subset = all_embeddings
    labels_subset = labels
print(f"--- 开始使用 t-SNE 对 {len(embeddings_subset)} 个样本进行降维 (这可能需要几分钟) ---")# 初始化 t-SNE 模型
# n_components=2 表示我们想降到二维
# perplexity 是一个关键参数，通常在5-50之间，30是一个很好的起点
# random_state 保证每次运行结果一致
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings_subset)
print("--- 降维完成，开始绘图 ---")

# 创建一个 DataFrame 以便使用 seaborn 绘图
plot_df = pd.DataFrame(embeddings_2d, columns=['tsne_1', 'tsne_2'])
plot_df['default'] = labels_subset
plot_df['default'] = plot_df['default'].map({0: 'No Default', 1: 'Default'}) # 将0/1标签换成更易读的文字

# 使用 seaborn 绘制散点图
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x="tsne_1", y="tsne_2",
    hue="default",
    palette=sns.color_palette("hls", 2), # 使用hls调色盘，指定两种颜色
    data=plot_df,
    legend="full",
    alpha=0.6 # 设置点的透明度，方便观察重叠区域
)

plt.title('t-SNE Visualization of Customer Embeddings', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(title='Customer Status')
plt.savefig("t-SNE Visualization of Customer Embeddings.png")