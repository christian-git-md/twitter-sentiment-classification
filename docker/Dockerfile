FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
RUN pip install 'transformers[torch]' \
 fastapi \
 uvicorn \
 python-multipart \
 scikit-learn \
 emoji \
 datasets \
 huggingface_hub \
 evaluate \
 matplotlib \
 jupyterlab
RUN apt-get update
RUN apt-get install -y \
 git-lfs \
 curl
COPY download_models.py .
RUN python3 download_models.py
COPY entrypoint.sh /usr/src/app/entrypoint.sh
RUN chmod +x /usr/src/app/entrypoint.sh
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]