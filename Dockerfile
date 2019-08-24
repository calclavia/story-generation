FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

RUN apt update && apt install -y rsync

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN git clone https://github.com/nvidia/apex && \
    cd apex && \
    python setup.py install --cuda_ext --cpp_ext && \
    rm -rf /apex

CMD "/bin/bash"