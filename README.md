environment.yml 里提供了我们进行实验的环境信息

安装detectron2

pip install 'git+https://mirror.ghproxy.com/https://github.com/facebookresearch/detectron2.git'

安装ViLT

cd ViLT

python setup.py install

cd ..

自行安装pytorch

pip install -r requirements.txt

按照models_dir.txt和dataset_dir.txt中的格式配置预训练模型以及数据集

运行python PMPL_main.py
