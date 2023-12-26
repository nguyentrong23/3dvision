import numpy as np

def distance_between_lines(line1, line2):
    # Hàm để tính vectơ pháp tuyến của đoạn thẳng
    def normal_vector(line):
        x1, y1, x2, y2 = line
        return np.array([y2 - y1, x1 - x2])

    # Lấy vectơ pháp tuyến của mỗi đoạn thẳng
    normal1 = normal_vector(line1)
    normal2 = normal_vector(line2)

    # Lấy vectơ giữa hai điểm đầu của chúng
    vector_between_points = np.array([line2[0] - line1[0], line2[1] - line1[1]])

    # Tính khoảng cách giữa hai đoạn thẳng
    distance = abs(np.dot(vector_between_points, normal1)) / np.linalg.norm(normal1)

    return distance

# Đoạn thẳng 1: (434, 183) đến (469, 183)
line1 = [434, 183, 469, 183]

# Đoạn thẳng 2: (1714, 220) đến (1733, 220)
line2 = [1714, 220, 1733, 220]

# Tính khoảng cách giữa hai đoạn thẳng
distance = distance_between_lines(line1, line2)

print(f"Khoảng cách giữa hai đoạn thẳng là: {distance}")