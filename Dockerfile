FROM pytorch/conda-builder:cuda12.1-ed16f28f02923da52c23c8d74924881b0be02fdc
RUN pip install numpy
RUN yum install -y libglvnd-glx

WORKDIR /PriVision
COPY . /PriVision
CMD ["python", "predict.py"]

