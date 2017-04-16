# Rapid Object Detection using a Boosted Cascade of Simple Features

## Phần 1. Giới thiệu

Bài báo **Rapid Object Detection using a Boosted Cascade of Simple Features** trình bày về Viola–Jones object detection framework, một framework sử dụng để xây dựng hệ thống phát hiện đối tượng trong ảnh theo thời gian thực (real-time).

Các vấn đề chính trong bài báo:

1. Trích chọn đặc trưng của ảnh sử dụng **Haar Feature Selection** (có sửa đổi thuật toán filter)
1. Lựa chọn tập các đặc trưng quan trọng của ảnh bằng cách sử dụng AdaBoost.
1. Kết hợp các phương pháp phân lớp (classifiers) được cấu trúc theo mô hình phân tầng (cascade structure) để phát hiện vùng trong ảnh có mặt người.

## Phần 2. Trích chọn đặc trưng  của ảnh

Chúng ta biểu diễn một đối tượng ảnh dưới dạng tập các đặc trưng (features) thay vì biểu diễn ảnh dưới dạng một mảng các pixel trong hệ thống phát hiện đối tượng, vì cách biểu diễn ảnh dưới dạng tập các đặc trưng cho hiệu năng cao hơn.

Đặc trưng của ảnh được sử dụng trong hệ thống này là 3 loại đặc trưng Haar:

- Đặc trưng **two-rectangle feature**
- Đặc trưng **three-rectangle feature**
- Đặc trưng **four-rectangle feature**

Một số thể hiện cụ thể của 3 đặc trưng trên trong OpenCV:

![haar_features](./images/haar_features.jpg)

Một ảnh cho trước có thể có nhiều đối tượng đặc trưng. Một đối tượng đặc trưng có các thuộc tính sau:

- Loại đặc trưng (two/three hay four-rectangle feature)
- Các tọa độ đỉnh của đối tượng đặc trưng.
- Kích thước của đối tượng đặc trưng.
- Giá trị của đối tượng đặc trưng ( được tính bằng tổng giá trị sáng của các pixel thuộc các vùng trắng trừ đi tổng giá trị sáng của các pixel thuộc vùng đen).

Ví dụ, xét một đối tượng đặc trưng **two-rectangle feature** **x**:
![haar-feature-example](./images/haar-feature-example.png)

Các thuộc tính của đối tượng đặc trưng **x** trong ví dụ này là:

- Loại đặc trưng: **two-rectangle feature**
- Tọa độ đỉnh:
  - Điểm đầu: [2,1]
  - Điểm cuối: [3,2]
- Kích thước: 2x2
- Giá trị: value = (6+3) - (5+1) = 3

Với 3 loại đặc trưng trên, một bức ảnh 24x24 có thể có tới trên 180 000 đối tượng đặc trưng cho 3 loại đặc trưng. Với con số đối tượng đặc trưng lớn như vậy, việc tính giá trị của đối tượng đặc trưng theo phương pháp thông thường tốn một khối lượng tính toán lớn: Với một đối tượng **two-rectangle feature** có kích thước 8x8 chúng ta sẽ mất khoảng 31+31 phép cộng và 1 phép trừ, tổng cộng 63 phép toán.

Để giảm thiểu khối lượng tính toán cần thiết để tính ra giá trị của các đối tượng đặc trưng, chúng ta sử dụng phương pháp **Integral Image**

### 2.1. Integral Image

Theo phương pháp phương pháp Integral Image, giá trị tích phân của một điểm có tọa độ (x,y) trong ảnh là:


## Tài liệu tham khảo

1. [https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)
1. [http://users.utcluj.ro/~tmarita/HCI/C7-8-extra/Face-detect/violaJones_CVPR2001.pdf](http://users.utcluj.ro/~tmarita/HCI/C7-8-extra/Face-detect/violaJones_CVPR2001.pdf)
1. Rapid Object Detection using a Boosted Cascade of Simple Features
