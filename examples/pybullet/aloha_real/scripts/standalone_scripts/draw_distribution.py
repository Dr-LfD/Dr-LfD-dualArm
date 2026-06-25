import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap

# 1. 生成单个簇的随机点
np.random.seed(0)
N = 100
mean, cov = [0, 0], [[1, 0], [0, 1]]
X = np.random.multivariate_normal(mean, cov, N)

# 2. 计算 KDE 密度估计
kde = gaussian_kde(X.T)

# 3. 创建网格来计算密度值
xmin, ymin = X.min(axis=0) - 2
xmax, ymax = X.max(axis=0) + 2
xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([xx.ravel(), yy.ravel()])
density = kde(positions).reshape(xx.shape)

# 4. 自定义颜色映射，使最低密度区域为白色
colors = ["white", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]  # 蓝色系
custom_cmap = ListedColormap(colors)

# 5. 画等高线填充图
plt.figure(figsize=(6, 5), facecolor='white')
contourf = plt.contourf(xx, yy, density, levels=6, cmap=custom_cmap, extend='both')
plt.contour(xx, yy, density, levels=6, colors='black', alpha=0.4)

# 6. 叠加原始随机点（白色填充，蓝色描边）
plt.scatter(X[:, 0], X[:, 1], color='green', edgecolors='white', s=40, label='Cluster', alpha= 0.7)

# 7. 生成低密度区域的随机点
num_low_density_points = 40
low_density_threshold = np.percentile(density, 60)  # 选择20%分位以下作为低密度区域

# 采样低密度点
low_density_points = []
while len(low_density_points) < num_low_density_points:
    x_rand = np.random.uniform(xmin, xmax)
    y_rand = np.random.uniform(ymin, ymax)
    d = kde([x_rand, y_rand])[0]
    if d < low_density_threshold:
        low_density_points.append([x_rand, y_rand])

low_density_points = np.array(low_density_points)

# 8. 画出低密度区域的橘黄色点
plt.scatter(low_density_points[:, 0], low_density_points[:, 1], color='orange', s=50, label='Low Density')

# 9. 显示
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
# plt.legend()
plt.axis('off')
# plt.title('KDE Contour with Low Density Points')
plt.show()
