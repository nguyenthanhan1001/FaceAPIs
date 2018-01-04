# FaceAPIs
Face APIs based on Bachelor Thesis

# Webserver:
- Cài đặt hệ điều hành Ubuntu 14.04 LTS x64
- Cài đặt python==2.7.13
- Cài đặt Django==1.8.7
- Giải nén file webserver.zip trong thư mục SOURCE vào /var/www
- Vận hành: /var/www/webserver:
	python ./manager.py runserver

# Face Detection - SSD300
- Cài đặt hệ điều hành Ubuntu 14.04 LTS x64
- Cài đặt python==2.7.13
- Cài đặt Django==1.8.7
- Giải nén file wsfd-ssd.zip trong thư mục SOURCE
- Sao chép nội dung thư mục wsfd-ssd vào /var/www
- Download file "model.tar.gz" theo link: https://drive.google.com/open?id=0B3NQa8ihCo_SZTNmb2NMaTVnaG8
Sau đó chép vào thư mục /var/www/SSDFace với tên "model"
- Vào thư mục /var/www/mmhci:
	python ./manager.py runserver

# Face Recognition - VGG16
- Cài đặt hệ điều hành Ubuntu 14.04 LTS x64
- Cài đặt python==2.7.13
- Cài đặt Django==1.8.7
- Giải nén file wsfr-vgg-ssd.zip trong thư mục SOURCE
- Sao chép nội dung thư mục wsfr-vgg-ssd vào /var/www
- Download file "vggface.npy" theo link: https://drive.google.com/open?id=0B3NQa8ihCo_SZEdfMlM3YnB3ZnM.
Sau đó chép vào thư mục /var/www/vgg16 với tên "vggface.npy"
- Download file "model-tf-50.tar.gz" theo link: https://drive.google.com/open?id=0B3NQa8ihCo_SWGxGb05UdmtRemc
Sau đó chép vào thư mục /var/www/vgg16 với tên "model-tf-50"
- Vào thư mục /var/www/mmhci:
python ./manager.py runserver

# Thư mục moduls:
- vggface: face recognition, fc7-extract
- SSDFace: face detection
- compose_tracklets: compose from recognised meta
- retrain_classification: retrain for new #uid