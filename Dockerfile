FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7 as base

# ADD .git to image to allow for commit hash retrieval
ADD . /src

WORKDIR /src

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install -y ffmpeg

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/sinzlab/propose.git
RUN pip install --no-cache-dir -e .

WORKDIR /src