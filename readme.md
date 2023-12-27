# Machine Learning Project

Trong nhiệm vụ này, chúng tôi thực hiện 10 phương pháp khác nhau.
10 phương pháp này, chúng tôi kết hợp mô hình ngôn ngữ lớn với mô hình Resnet50 và VGG16.
Chúng tôi, sử dụng data augmentation và đạt được kết quả tốt nhất.


Để chạy phiên bản tốt nhất của chúng tôi.

Đầu tiên cài các môi trường cần sử dụng.

```
pip install requirement.txt
```

Tiếp theo, chia data với câu lệnh dưới đây.
Chúng tôi lấy 20% sample trong tập train làm tập valid.

```
bash script/split_data.sh
```

Sau đó, để thực hiện bước data_augmentation:

```
bash script/data_aug.sh
```

Chúng tôi chỉ thực hiện data augmentation với các sample có nhãn không chứa <font color="green"> Comedy </font> và <font color="green"> Drama </font>

Để thực hiện huấn luyện mô hình ta chạy lệnh:

```
bash script/train.sh
```

Để test model:

```
bash script/test.sh
```

Để thực hiện dự đoán một ảnh:

```
python demo.py --model path_to_model --path_image path_to_image --title title_of_image
```

