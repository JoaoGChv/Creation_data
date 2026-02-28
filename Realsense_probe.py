#!/usr/bin/env python3
"""
realsense_probe.py
==================
Testa combinacoes de FPS/resolucao suportadas pela D415 e identifica
a melhor config para USB 2.1. Execute ANTES do script principal.

Uso:
    python realsense_probe.py
"""

import time
import sys

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERRO] pip install pyrealsense2")
    sys.exit(1)


def probe_config(width, height, fps, fmt_color=rs.format.bgr8, fmt_depth=rs.format.z16):
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, width, height, fmt_color, fps)
    config.enable_stream(rs.stream.depth, width, height, fmt_depth, fps)
    try:
        profile = pipeline.start(config)
        # Tenta capturar 3 frames reais com timeout generoso
        ok = 0
        for _ in range(3):
            try:
                f = pipeline.wait_for_frames(timeout_ms=6000)
                if f.get_color_frame() and f.get_depth_frame():
                    ok += 1
            except RuntimeError:
                pass
        pipeline.stop()
        return ok > 0
    except RuntimeError:
        try:
            pipeline.stop()
        except Exception:
            pass
        return False


def main():
    print("=== RealSense D415 — Probe de configuracoes suportadas ===\n")

    # Hardware reset primeiro
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("[ERRO] Nenhum dispositivo encontrado.")
        sys.exit(1)

    dev = devs[0]
    print(f"Dispositivo: {dev.get_info(rs.camera_info.name)}")
    print(f"Serial     : {dev.get_info(rs.camera_info.serial_number)}")
    print(f"USB type   : {dev.get_info(rs.camera_info.usb_type_descriptor)}")
    print(f"Firmware   : {dev.get_info(rs.camera_info.firmware_version)}")
    print()

    print("Resetando hardware...")
    dev.hardware_reset()
    time.sleep(5)

    # Combinacoes a testar — do mais lento para o mais rapido
    candidates = [
        (640, 480, 6),
        (640, 480, 15),
        (640, 480, 30),
        (424, 240, 6),
        (424, 240, 15),
        (424, 240, 30),
        (320, 240, 6),
        (320, 240, 30),
    ]

    print(f"{'Resolucao':<16} {'FPS':<6} {'Funciona?'}")
    print("-" * 36)
    working = []
    for w, h, fps in candidates:
        result = probe_config(w, h, fps)
        status = "✓  SIM" if result else "✗  nao"
        print(f"  {w}x{h:<10}  {fps:<6} {status}")
        if result:
            working.append((w, h, fps))
        time.sleep(1)  # pausa entre testes para o barramento descansar

    print()
    if working:
        best = working[0]
        print(f"[OK] Configuracoes que funcionam: {working}")
        print(f"[OK] Recomendada para USB 2.1: WIDTH={best[0]}, HEIGHT={best[1]}, FPS={best[2]}")
        print()
        print("Use esses valores no realsense_d415_capture.py:")
        print(f"  WIDTH, HEIGHT = {best[0]}, {best[1]}")
        print(f"  FPS           = {best[2]}")
    else:
        print("[ERRO] Nenhuma configuracao funcionou.")
        print("Verifique:")
        print("  - Cabo USB (use cabo USB 3.0 mesmo em porta 2.1)")
        print("  - Porta USB (tente outra porta diretamente na placa-mae)")
        print("  - sudo dmesg | grep -i usb | tail -20")


if __name__ == "__main__":
    main()
