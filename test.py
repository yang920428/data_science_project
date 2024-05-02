import numpy as np

# 假设 data_array 是你的二维数组
data_array = np.array([
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19, 20, 21]
])

# 获取前6列数据
sub_array = data_array[:, :6]

# 打印前6列数据
print(sub_array)
