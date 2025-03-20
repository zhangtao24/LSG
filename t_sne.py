# 导入必要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sn  # 导入seaborn用于绘图
from matplotlib import rcParams

# 设置字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'

# 读取数据
df = pd.read_csv(r"D:\studysoft\python_project\t_sne\S2_DEM.csv")

# 打印前4行数据
print(df.head(4))

# 提取标签和像素数据
l = df['label']
d = df.drop("label", axis=1)

# 数据标准化
standardized_data = StandardScaler().fit_transform(d)
print(standardized_data.shape)

# 选择前1000个数据点用于t-SNE
data_1000 = standardized_data[0:13200, :]
labels_1000 = l[0:13200]

# 使用t-SNE进行降维
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(data_1000)

# 创建DataFrame以便绘图
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# 创建FacetGrid图表
g = sn.FacetGrid(tsne_df, hue="label", height=6)
g.map(plt.scatter, 'Dim_1', 'Dim_2', s=5)

# 添加图例
g.add_legend()

# 显示右轴和上轴
g.set_axis_labels('Dim_1', 'Dim_2')  # 设置X轴和Y轴标签
g.fig.subplots_adjust(right=0.85, top=0.85)  # 调整子图布局避免遮挡

# 显示上轴和右轴
for ax in g.axes.flat:
    ax.spines['top'].set_visible(True)  # 显示上边框
    ax.spines['right'].set_visible(True)  # 显示右边框
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=16)
    # ax.set_xticklabels([])  # 删除x轴刻度标签
    # ax.set_yticklabels([])  # 删除y轴刻度标签
    # ax.set_xticks([])  # 删除x轴刻度线
    # ax.set_yticks([])  # 删除y轴刻度线

plt.savefig(r"D:\studysoft\python_project\t_sne\S2_DEM.png", dpi=300)
# 显示图形
plt.show()
