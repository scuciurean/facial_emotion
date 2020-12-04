Python version: 3.8.5

## Clone the repo
```
git clone -b custom_database https://github.com/cuciureansergiu/facial_emotion.git
cd facial_emotion
```

## Install libs
```
python -m pip install -r requirements.txt
```
## Copy database
```
facial_emotion
.
├── data
│   └── lena.png
├── database                    <-- the custom database
│   ├── a
│   ├── d
│   ├── f
│   ├── h
│   ├── n
│   ├── sa
│   └── su
├── detect.py
├── hog.py
├── imfilters.py
├── LICENSE
├── README.md
├── requirements.txt
├── trained_model.svm
├── train_emotion.py
├── train_face.py
└── util.py
```

## Train model
```
python train_emotion.py database 0.7
```

## Output
The model will be saved as [database_name].svm and the confusion matrix in [database_name]_report.txt
