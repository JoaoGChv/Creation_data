# ─── Base ────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# ─── Dependências do sistema ──────────────────────────────────────────────────
# Não usamos o repositório apt da Intel: o wheel pyrealsense2 do PyPI já
# embute a biblioteca librealsense2 compilada, sem necessidade de GPG/apt.
# O módulo de kernel (librealsense2-dkms) deve estar instalado no HOST.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        # USB access
        libusb-1.0-0 \
        udev \
        # OpenCV GUI (GTK3 + X11)
        libgl1 \
        libglib2.0-0 \
        libgtk-3-0 \
        libgdk-pixbuf2.0-0 \
        libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# ─── Dependências Python ──────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
# pyrealsense2 do PyPI inclui librealsense2 compilada — não precisa do apt Intel
RUN pip3 install --no-cache-dir -r requirements.txt

# ─── Código ───────────────────────────────────────────────────────────────────
COPY realsense_capture.py .
COPY Realsense_probe.py   .

RUN mkdir -p /app/output
VOLUME ["/app/output"]

# ─── GUI via X11 (passthrough do host) ───────────────────────────────────────
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0

# ─── Entrypoint ───────────────────────────────────────────────────────────────
ENTRYPOINT ["python3", "realsense_capture.py"]
CMD []
