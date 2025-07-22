import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sn  # 导入seaborn用于绘图
from matplotlib import rcParams

# Times New Roman
rcParams['font.family'] = 'Times New Roman'

# Read data
df = pd.read_csv(r"D:\studysoft\python_project\t_sne\S2_DEM.csv")

# Print the first four lines of data
print(df.head(4))

# Extract label and pixel data
l = df['label']
d = df.drop("label", axis=1)

# Data standardization
standardized_data = StandardScaler().fit_transform(d)
print(standardized_data.shape)

# Select 13,200 data points for t-SNE
data_1000 = standardized_data[0:13200, :]
labels_1000 = l[0:13200]

#Use t-SNE for dimensionality reduction
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(data_1000)

tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
g = sn.FacetGrid(tsne_df, hue="label", height=6)
g.map(plt.scatter, 'Dim_1', 'Dim_2', s=5)

g.add_legend()

g.set_axis_labels('Dim_1', 'Dim_2')  # 设置X轴和Y轴标签
g.fig.subplots_adjust(right=0.85, top=0.85)  # 调整子图布局避免遮挡

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
plt.show()
