FROM nvidia/cuda:12.1.0-base-ubuntu22.04 
CMD nvidia-smi

FROM python:3.9-slim

WORKDIR /app

COPY . .
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r req.txt
RUN pip3 install torch=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
RUN pip3 install transformers==4.38.1
RUN pip3 install tokenizers==0.15.0
