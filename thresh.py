import numpy as np

# Các giá trị của hai vector
vector1 = np.array([617, 656])
vector2 = np.array([609, 710])

# Tính khoảng cách Euclidean
euclidean_distance = np.linalg.norm(vector1 - vector2)

# Tính khoảng cách Manhattan
manhattan_distance = np.sum(np.abs(vector1 - vector2))

print(f"Khoảng cách Euclidean: {euclidean_distance}")
print(f"Khoảng cách Manhattan: {manhattan_distance}")
