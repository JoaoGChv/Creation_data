#!/usr/bin/env bash
# docker-run.sh — Atalhos para rodar o container de captura RealSense D455
#
# Uso:
#   ./docker-run.sh build                      # constrói a imagem
#   ./docker-run.sh capture                    # captura direta (ESPAÇO=salvar)
#   ./docker-run.sh bag [arquivo.bag]          # grava sessão em .bag
#   ./docker-run.sh extract arquivo.bag        # extrai frames de um .bag
#   ./docker-run.sh probe                      # diagnóstico de configurações suportadas
#
# REQUISITO NO HOST:
#   sudo apt install librealsense2-dkms        # módulo do kernel (apenas no host)

set -euo pipefail

IMAGE="realsense-d455"
OUTPUT_DIR="$(pwd)/output"

# Flags comuns a todos os modos ao vivo (acesso USB + display X11)
DOCKER_FLAGS=(
    --rm
    --privileged
    -v /dev/bus/usb:/dev/bus/usb
    -v "${OUTPUT_DIR}:/app/output"
    # GUI via X11 (comente as duas linhas abaixo se não quiser janela de preview)
    -e DISPLAY="${DISPLAY:-:0}"
    -v /tmp/.X11-unix:/tmp/.X11-unix
)

# Permite conexão X11 do container (executa no host, não dentro do Docker)
allow_x11() {
    if command -v xhost &>/dev/null; then
        xhost +local:docker 2>/dev/null || true
    fi
}

case "${1:-help}" in
    build)
        echo "[BUILD] Construindo imagem ${IMAGE}..."
        docker build -t "${IMAGE}" .
        echo "[BUILD] Concluído."
        ;;

    capture)
        allow_x11
        echo "[RUN] Captura direta — ESPAÇO=salvar  Q=sair"
        echo "      Saída em: ${OUTPUT_DIR}"
        docker run "${DOCKER_FLAGS[@]}" "${IMAGE}"
        ;;

    bag)
        allow_x11
        BAG_ARG="${2:-}"
        echo "[RUN] Gravação .bag — Q=parar"
        if [[ -n "${BAG_ARG}" ]]; then
            docker run "${DOCKER_FLAGS[@]}" \
                -v "$(pwd):/app/bags" \
                "${IMAGE}" --bag "/app/bags/${BAG_ARG}"
        else
            # Nome automático; salva no diretório atual
            docker run "${DOCKER_FLAGS[@]}" \
                -v "$(pwd):/app/bags" \
                "${IMAGE}" --bag
        fi
        ;;

    extract)
        BAG_FILE="${2:?Uso: $0 extract arquivo.bag}"
        ABS_BAG="$(realpath "${BAG_FILE}")"
        echo "[RUN] Extraindo frames de: ${ABS_BAG}"
        echo "      Saída em: ${OUTPUT_DIR}"
        docker run --rm \
            -v "${ABS_BAG}:/app/input.bag:ro" \
            -v "${OUTPUT_DIR}:/app/output" \
            "${IMAGE}" --extract /app/input.bag
        ;;

    probe)
        allow_x11
        echo "[RUN] Probe de configurações suportadas..."
        docker run "${DOCKER_FLAGS[@]}" \
            --entrypoint python3 \
            "${IMAGE}" Realsense_probe.py
        ;;

    help|*)
        echo "Uso: $0 {build|capture|bag [arquivo]|extract arquivo.bag|probe}"
        ;;
esac
