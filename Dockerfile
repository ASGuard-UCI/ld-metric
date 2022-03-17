FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 



RUN conda create -n ld-metric python=3.8 -y
SHELL ["conda", "run", "-n", "ld-metric", "/bin/bash", "-c"]
RUN conda install jupyter
RUN conda install pytorch==1.6 torchvision -c pytorch
RUN conda install -c conda-forge opencv
RUN conda install numba
RUN conda install -c conda-forge pyopencl pocl intel-opencl-rt intel-compute-runtime beignet ocl-icd-system

RUN conda init bash
RUN echo "conda activate ld-metric" >> ~/.bashrc

RUN apt install -y ffmpeg libsm6 libxext6 git 


RUN git clone https://github.com/commaai/openpilot -b v0.6.6

WORKDIR /openpilot/phonelibs
RUN apt install -y autoconf libtool
RUN sh install_capnp.sh

WORKDIR /openpilot/selfdrive/messaging
RUN apt install -y clang
RUN conda install cython
RUN PYTHONPATH=$PYTHONPATH:/openpilot/ make

WORKDIR /openpilot/selfdrive/controls/lib/lateral_mpc
RUN find ./ -name "*.o" | xargs rm -rf
RUN make all

WORKDIR /openpilot/cereal/
RUN make

WORKDIR /openpilot/selfdrive/boardd
RUN PYTHONPATH=$PYTHONPATH:/openpilot/ make boardd_api_impl.so

WORKDIR /openpilot/common/kalman
RUN PYTHONPATH=$PYTHONPATH:/openpilot/ make simple_kalman_impl.so

WORKDIR /openpilot/selfdrive/can
RUN pip install jinja2
RUN PYTHONPATH=$PYTHONPATH:/openpilot/ make -j3
RUN PYTHONPATH=$PYTHONPATH:/openpilot/ make -j3 packer_impl.so

WORKDIR /openpilot/common/kalman
RUN PYTHONPATH=$PYTHONPATH:/openpilot/ make simple_kalman_impl.so

COPY data/tools /openpilot/tools

RUN echo "export PYTHONPATH=$PYTHONPATH:/openpilot" >> ~/.bashrc

COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

