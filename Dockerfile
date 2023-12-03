FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7 as base

# ADD .git to image to allow for commit hash retrieval
ADD . /src

WORKDIR /src

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir git+https://github.com/sinzlab/propose.git
RUN pip install --no-cache-dir -e .

WORKDIR /src