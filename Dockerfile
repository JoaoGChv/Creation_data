# ─── Base ────────────────────────────────────────────────────────────────────
# Ubuntu 22.04 (jammy) — repositório oficial do RealSense SDK está disponível
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# ─── Dependências do sistema ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg2 \
        lsb-release \
        ca-certificates \
        usbutils \
        python3 \
        python3-pip \
        # OpenCV com suporte a GUI (GTK)
        python3-opencv \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ─── SDK Intel RealSense (userspace — kernel module fica no HOST) ─────────────
# O librealsense2-dkms (módulo do kernel) NÃO pode ser instalado dentro do
# container. Ele deve estar instalado no host. O container usa apenas a
# biblioteca userspace via USB passthrough.
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
        | gpg --dearmor -o /usr/share/keyrings/librealsense.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/librealsense.gpg] \
        https://librealsense.intel.com/Debian/apt-repo jammy main" \
        > /etc/apt/sources.list.d/librealsense.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        librealsense2 \
        librealsense2-utils \
        librealsense2-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── Dependências Python ──────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ─── Código ───────────────────────────────────────────────────────────────────
COPY realsense_capture.py .
COPY Realsense_probe.py   .

# Diretório de saída (montado como volume em runtime)
RUN mkdir -p /app/output
VOLUME ["/app/output"]

# ─── Variável para GUI via X11 (opcional) ────────────────────────────────────
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0

# ─── Entrypoint ───────────────────────────────────────────────────────────────
ENTRYPOINT ["python3", "realsense_capture.py"]
CMD []
