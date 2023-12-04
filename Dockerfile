#base image
FROM python:3.10
LABEL org.opencontainers.image.source https://github.com/serengil/deepface
# -----------------------------------
# create required folder
RUN mkdir -p /app/deepface && mkdir -p root/.deepface/weights/
# -----------------------------------
# Copy required files from repo into image
COPY ./requirements.txt /app/
COPY ./setup.py /app/
COPY ./README.md /app/

# -----------------------------------
# update image os
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 wget -y

WORKDIR /root/.deepface/weights/
RUN wget https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx && \
    wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx && \
    wget https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5 && \
    wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
# -----------------------------------
# switch to application directory
WORKDIR /app

# -----------------------------------
# if you will use gpu, then you should install tensorflow-gpu package
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org onnxruntime==1.15.1
# -----------------------------------
# install deepface from pypi release (might be out-of-the-date)
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org deepface
# -----------------------------------
# install deepface from source code (always up-to-date)
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .
# -----------------------------------
# some packages are optional in deepface. activate if your task depends on one.
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org cmake==3.24.1.1
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org dlib==19.20.0
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org lightgbm==2.3.1
# -----------------------------------

COPY ./deepface /app/deepface
COPY ./api/*.py /app/
# environment variables
ENV PYTHONUNBUFFERED=1
ARG WORKERS=4
ARG SNOWFACE_PORT=5000
ENV WORKERS=${WORKERS}
ENV SNOWFACE_PORT=${SNOWFACE_PORT}
ENV PROMETHEUS_MULTIPROC_DIR=/prometheus-metrics
ENV PROMETHEUS_DISABLE_CREATED_SERIES=True
RUN mkdir -p $PROMETHEUS_MULTIPROC_DIR
# -----------------------------------
# run the app (re-configure port if necessary)
EXPOSE $SNOWFACE_PORT
CMD ["bash","-c","gunicorn --workers=$WORKERS --threads=2 --worker-class=gthread --config=/app/gunicorn.conf.py --timeout=180 --keep-alive=30 --bind=0.0.0.0:$SNOWFACE_PORT 'app:create_app()'"]
