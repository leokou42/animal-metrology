FROM python:3.12-slim

ARG INSTALL_DEPTH_PRO=false

ENV TZ=Asia/Taipei

WORKDIR /app

# System deps for OpenCV headless + pycocotools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git tzdata libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Depth Pro to a separate cache dir (not affected by volume mount)
RUN if [ "$INSTALL_DEPTH_PRO" = "true" ]; then \
        pip install --no-cache-dir git+https://github.com/apple/ml-depth-pro.git && \
        mkdir -p /opt/depth_pro && \
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download('apple/DepthPro', 'depth_pro.pt', local_dir='/opt/depth_pro/')"; \
    fi

COPY . .

RUN mkdir -p /app/outputs /app/weights

EXPOSE 8000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
