FROM abalajiaus/oct_ca:latest

RUN pip install fastai
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install mlflow
RUN pip install cython
RUN pip install opencv-python
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
RUN apt-get install -y libgtk2.0-dev
RUN pip install pycocotools 
