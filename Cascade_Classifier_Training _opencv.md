# Cascade Classifier Training

Để thực hiện việc huấn luyện một hàm phân lớp cho Viola–Jones object detection framework, chúng ta cần thực hiện 2 công việc: Chuẩn bị dữ liệu training và chạy chương trình training

Các công cụ liên quan tới việc chuẩn bị dữ liệu training:

- **opencv\_createsamples** is used to prepare a training dataset of positive and test samples. opencv\_createsamples produces dataset of positive samples in a format that is supported by both opencv\_haartraining and opencv\_traincascade applications. The output is a file with \*.vec extension, it is a binary format which contains images.

**opencv\_createsamples** được sử dụng để chuẩn bị dữ liệu training cho các ví dụ học chứa đối tượng (positive samples) cũng như các ví dụ trong tập tests. Output của lệnh này là một file đuôi **\*.vec**, đây là một file binary chứa các ảnh trong tập ví dụ train chứa đối tượng.

An opencv\_createsamples utility provides functionality for dataset generating, writing and viewing. The term dataset is used here for both training set and test set.

## 1. Quá trình chuẩn bị dữ liệu

Để huấn luyện hàm phân lớp cascade classifier, chúng ta cần 2 tập ví dụ học: Tập ví dụ học chứa các hình ảnh có đối tượng - positive samples và tập ví dụ học chứa các hình ảnh không có đối tượng - negative samples - non-object images. Tập positive samples  được tạo ra từ công cụ **opencv\_createsamples** dưới dạng một file vector chứa các positive samples, còn tập negative samples chúng ta cần tự chuẩn bị.

### 1.1 Negative samples

Để sử dụng các ví dụ negative samples trong quá trình huấn luyện, chúng ta cần phải có một file txt ghi lại danh sách các tệp ảnh là Negative sample, và một folder chứa tất cả các ảnh được liệt kê trong file txt này. Ví dụ:

An example of description file:

Directory structure:

```bash
    /img
      img1.jpg
      img2.jpg
    bg.txt
```

File **bg.txt**:

```bash
    img/img1.jpg
    img/img2.jpg
```

### 1.2 Positive Samples

Tập các Positive samples có thể tạo ra chỉ cần bằng một hình ảnh, nhưng cũng có thể cần tới vài trăm - hàng ngàn hình ảnh như trong trường hợp nhận diện mặt người. Trong trường hợp tập Positive Samples chỉ có một ảnh, thì các sample biến thể sẽ được tạo ra bằng cách biến đổi ảnh gốc đầu tiên theo các thao tác như xoay ngẫu nhiên, đổi màu, thay đổi cường độ sáng ... Và trong trường hợp này chúng ta cần truyền thêm tham số vào câu lệnh opencv\_createsamples để điều khiển việc tạo các sample biến thể từ sample gốc.

Các tham số có mà chúng ta có thể truyền vào câu lệnh tạo file vector chứa các positive samples - câu lệnh **opencv\_createsamples** là:

```bash
~/GitHubRepositories/MMDB_20162$ opencv\_createsamples
Usage: opencv\_createsamples

- [-info <collection_file_name>]
- [-img <image_file_name>]
- [-vec <vec_file_name>]
- [-bg <background_file_name>]
- [-num <number_of_samples = 1000>]
- [-bgcolor <background_color = 0>]
- [-inv] [-randinv] [-bgthresh <background_color_threshold = 80>]
- [-maxidev <max_intensity_deviation = 40>]
- [-maxxangle <max_x_rotation_angle = 1.100000>]
- [-maxyangle <max_y_rotation_angle = 1.100000>]
- [-maxzangle <max_z_rotation_angle = 0.500000>]
- [-show [<scale = 4.000000>]]
- [-w <sample_width = 24>]
- [-h <sample_height = 24>]
- [-pngoutput]
```

```bash
Command line arguments:

    -vec <vec_file_name>

        Name of the output file containing the positive samples for training.

    -img <image_file_name>

        Source object image (e.g., a company logo).

    -bg <background_file_name>

        Background description file; contains a list of images which are used as a background for randomly distorted versions of the object.

    -num <number_of_samples>

        Number of positive samples to generate.

    -bgcolor <background_color>

        Background color (currently grayscale images are assumed); the background color denotes the transparent color. Since there might be compression artifacts, the amount of color tolerance can be specified by -bgthresh. All pixels withing bgcolor-bgthresh and bgcolor+bgthresh range are interpreted as transparent.

    -bgthresh <background_color_threshold>

    -inv

        If specified, colors will be inverted.

    -randinv

        If specified, colors will be inverted randomly.

    -maxidev <max_intensity_deviation>

        Maximal intensity deviation of pixels in foreground samples.

    -maxxangle <max_x_rotation_angle>

    -maxyangle <max_y_rotation_angle>

    -maxzangle <max_z_rotation_angle>

        Maximum rotation angles must be given in radians.

    -show

        Useful debugging option. If specified, each sample will be shown. Pressing Esc will continue the samples creation process without.

    -w <sample_width>

        Width (in pixels) of the output samples.

    -h <sample_height>

        Height (in pixels) of the output samples.

    -pngoutput

        With this option switched on opencv_createsamples tool generates a collection of PNG samples and a number of associated annotation files, instead of a single vec file.

```

### 1.2.1 Tạo file vector chứa các Positive Samples từ tập ảnh chứa các đối tượng - Converting the marked-up collection of samples into a vec format

Chúng ta có thể tạo ra file vector chứa tập các Positive Samples từ một tập các ảnh có chứa các đối tượng được đặt trong thư mục gọi là **source positive images**. Lúc này, chúng ta cần có một file **info.txt** chứa danh sách thông tin về các bức ảnh nằm trong source positive images.

Thông tin về một bức ảnh chứa trong file **info.txt** bao gồm:

- tên/đường dẫn của bức ảnh
- Số lượng đối tượng có trong bức ảnh đó (một bức ảnh có thể có nhiều hơn một đối tượng, ví dụ như ảnh chụp tập thể có n đối tượng là n gương mặt của n người).
- Vị trí của từng đối tượng trong bức ảnh, được xác định bởi 4 thông số:
   - Tọa độ x
   - Tọa độ y
   - Chiều rộng - width
   - Chiều dài - heigh

Ví dụ:

```bash


Directory structure:

    /img
      img1.jpg
      img2.jpg
    info.dat

File info.dat:

    img/img1.jpg  1  140 100 45 45
    img/img2.jpg  2  100 200 50 50   50 30 25 25

```

Các thông số của ví dụ trên nói lên:

- Bức ảnh 1 có đường dẫn tương đối là  img/img1.jpg, chứa một đối tượng ở tọa độ (140,100), kích thước là 45 x 45 pixel
- Bức ảnh 2 có đường dẫn tương đối là  img/img2.jpg, chứa hai đối tượng: Đối tượng đầu tiên ở tọa độ (100,200), kích thước là 50 x 50 pixel, đối tượng thứ hai ở tọa độ (50,30), kích thước là 25 x 25 pixel

Khi tạo file từ ảnh chứa đối tượng, chúng ta cần phải chỉ định cho câu lệnh opencv\_createsamples vị trí của file **info.txt** thông qua tham số **-info** các thông số khác mà chúng ta cần quan tâm khi sử dụng cách này là:

```bash
Command line arguments:

    -vec <vec_file_name> : đường dẫn của file vector output

        Name of the output file containing the positive samples for training.
    -num <number_of_samples>: Số lượng sample có trong file vector output

        Number of positive samples to generate.
    -show

        Useful debugging option. If specified, each sample will be shown. Pressing Esc will continue the samples creation process without.

    -w <sample_width> chiều rộng của output sample

        Width (in pixels) of the output samples.

    -h <sample_height> chiều cao của output sample

        Height (in pixels) of the output samples.
```

Sau khi đã sử dụng câu lệnh opencv\_createsamples để tạo file vector đầu vào và chuẩn bị tập Negative samples, chúng ta tiến hành công việc huấn luyện hàm phân lớp.

## 2. Quá trình huấn luyện hàm phân lớp - Cascade Training

Để huấn luyện hàm phân lớp, chúng ta cần có dữ liệu đầu vào như đã chuẩn bị ở phần trước, sau đó chúng ta sử dụng câu lệnh **opencv\_traincascade** để tiến hành huấn luyện và tạo hàm phân lớp. Hàm phân lớp được câu lệnh **opencv\_traincascade** được lưu vào một file xml, và được sử dụng trong quá trình phát hiện đối tượng sau đó:

Các tham số truyền vào câu lệnh **opencv\_traincascade** khi chúng ta huấn luyện hàm phân lớp là:

```md
Common arguments:

    -data <cascade_dir_name> vị trí lưu file chứa hàm phân lớp sau huấn luyện

    -vec <vec_file_name> file vector đầu vào mà câu lệnh opencv\_createsamples tạo ra vec-file with positive samples (created by opencv_createsamples utility).

    -numPos <number_of_positive_samples> Số lượng sample ở lớp 1 mà chúng ta sử dụng để huấn luyện hàm phân lớp

    -numNeg <number_of_negative_samples> Số lượng sample ở lớp 0 mà chúng ta sử dụng để huấn luyện hàm phân lớp

    -numStages <number_of_stages> Số tầng của cascade classifier

    -precalcValBufSize <precalculated_vals_buffer_size_in_Mb>: Lượng ram mà chúng ta sử dụng để huấn luyện hàm phân lớp
    -precalcIdxBufSize <precalculated_idxs_buffer_size_in_Mb>: Lượng ram mà chúng ta sử dụng để huấn luyện hàm phân lớp

Cascade parameters:

    -stageType <BOOST(default)> Phương pháp tạo các hàm phân lớp tại các stage

    -featureType<{HAAR(default), LBP}> Loại đặc trưng mà chúng ta sử dụng để nhận diện đối tượng. Type of features: HAAR - Haar-like features, LBP - local binary patterns.
    -w <sampleWidth>    -h <sampleHeight> : Kích thước chính xác của các sample trong file vector mà chúng ta sử dụng để huấn luyện - Hai thông số này phải trùng với hai thông số tương tự mà chúng ta truyền vào câu lệnh opencv\_createsampless
Boosted classifer parameters:

    -bt <{DAB, RAB, LB, GAB(default)}> Loại boosted classifiers mà chúng ta sử dụng để tạo classifier ở từng stage: DAB - Discrete AdaBoost, RAB - Real AdaBoost, LB - LogitBoost, GAB - Gentle AdaBoost.

    -minHitRate <min_hit_rate> Tỉ lệ tối thiểu các ví dụ học lớp 1 mà classifier của từng tầng phải nhận dạng được. Ví dụ nếu -minHitRate = 0.99 và số lượng positive samples là 10 000 thì hàm classifier của từng tầng phải nhận dạng đúng ít nhất 10 000 *0.99 = 9900 samples trong tập positive samples vào lớp 1. Tỉ lệ hit rate trên toàn bộ n stages của hàm phân lớp cuối cùng là min_hit_rate ^ n

        Minimal desired hit rate for each stage of the classifier. Overall hit rate may be estimated as (min_hit_rate ^ number_of_stages) [Viola2004].
        Hit rate is defined as the ratio of the number of objects detected in the test image to that of the total objects present. While training a minimum required hit rate is specified, this also depicts the quality of training. MHRate is sometimes also referred as sensitivity.

    -maxFalseAlarmRate <max_false_alarm_rate>

        tỉ lệ % số lượng negative samples tối đa (các anhrh không chứa đối tượng) có thể bị classfier của một stage phân lớp nhầm thành lớp 1 positive class. Tỉ lệ phân lớp nhầm ảnh lớp 0 thành lớp 1 trên toàn bộ n stage là max_false_alarm_rate ^ n

        Maximal desired false alarm rate for each stage of the classifier. Overall false alarm rate may be estimated as (max_false_alarm_rate ^ number_of_stages) [Viola2004].

        Lưu ý rằng: the training terminates, when one of Max_false_alarm_rate or Min_hit_rate is reached. Khi Min_hit_rate - quá trình training không đạt yêu cầu. Khi Max_false_alarm_rate reach - tỉ lệ lỗi giảm xuống, quá trình training đạt yêu cầu và nhảy sang stage tiếp theo hoặc kết thúc và in ra hàm phân lớp cuối cùng - nếu như đang ở stage cuối.

    -maxDepth <max_depth_of_weak_tree>

        Maximal depth of a weak tree. A decent choice is 1, that is case of stumps.

    -maxWeakCount <max_weak_tree_count>

        Maximal count of weak trees for every cascade stage. The boosted classifier (stage) will have so many weak trees (<=maxWeakCount), as needed to achieve the given -maxFalseAlarmRate.

Haar-like feature parameters:

    -mode <BASIC (default) | CORE | ALL>

        Thiết lập các loại đặc trưng sẽ được sử dụng. Nếu giá trị là BASIC thì chỉ có các đặc trưng đơn giản hình chữ nhật được sử dụng. Nếu giá trị là ALL thì mọi loại đặc trưng, kể cả các đặc trưng nghiêng 45 độ như hình thoi cũng được sử dụng.
        Selects the type of Haar features set used in training. BASIC use only upright features, while ALL uses the full set of upright and 45 degree rotated feature set. See [Rainer2002] for more details.
```

Sau khi câu lệnh **opencv\_traincascade** thực hiện xong, hàm phân lớp sẽ được lưu vào một file có định dạng .xml tại đường dẫn được chỉ định bởi tham số **-data** parameter. Đến đây là quá trình huấn luyện hàm phân lớp đã hoàn tất.
