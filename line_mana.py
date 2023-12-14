import cv2
import numpy as np
import  sys
class Line:
    def __init__(self, line_id, start_point, end_point):
        self.line_id = line_id
        self.start_point = start_point
        self.end_point = end_point

    def __str__(self):
        return f"{self.line_id} {self.start_point[0]} {self.start_point[1]} {self.end_point[0]} {self.end_point[1]}/"

class LineManager:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.lines = []

    def preprocess_and_highlight_edges(self):
        blurred_image = cv2.GaussianBlur(self.image, (5, 5), 0)
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, 140, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(thresholded_image, 50, 150)
        cv2.imshow('edges', edges)
        lines_data = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=15)

        if lines_data is not None:
            for i, line_data in enumerate(lines_data):
                x1, y1, x2, y2 = line_data[0]
                line = Line(i + 1, (x1, y1), (x2, y2))
                self.lines.append(line)
        return self.lines

    def draw_lines_on_image(self):
        for line in self.lines:
            cv2.line(self.image, line.start_point, line.end_point, (0, 0, 255), 1, cv2.LINE_AA)

    def display_image_with_lines(self):
        cv2.imshow('Image with Lines', self.image)

    def find_line_by_id(self, line_id):
        for line in self.lines:
            if line.line_id == line_id:
                return line
        return None


def main():
    if len(sys.argv[0]) < 6 and len(sys.argv) < 2:
        print("path error")
    else:
        image_path = sys.argv[1]
        try:
            line_manager = LineManager(image_path)
            line_manager.preprocess_and_highlight_edges()
            line_manager.draw_lines_on_image()
            line_manager.display_image_with_lines()
            print(*line_manager.lines)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            return 'file not valid'

if __name__ == "__main__":
    try:
        main()
    except:
        image_path = "datafornichi/src/realsense/image.bmp"
        line_manager = LineManager(image_path)
        line_manager.preprocess_and_highlight_edges()
        line_manager.draw_lines_on_image()
        line_manager.display_image_with_lines()
        print(*line_manager.lines)
        line_id_to_find = 20
        found_line = line_manager.find_line_by_id(line_id_to_find)
        if found_line:
            print(f"\nĐã tìm thấy đường thẳng với ID {line_id_to_find}: {found_line}")
        else:
            print(f"\nKhông tìm thấy đường thẳng với ID {line_id_to_find}")
