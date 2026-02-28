#!/usr/bin/env python3
"""
realsense_d415_capture.py
=========================
Captura RGB-D resiliente para Intel RealSense D415 em USB 2.1 / Ubuntu 24.04.

Estrutura de saida compativel com HorizonGS:
    output/
    ├── rgb/   frame_00001.jpg  ...
    └── depth/ frame_00001.npy  ...

Tecla ESPACO -> salva par (RGB + Depth)
Tecla Q / ESC -> encerra

Dependencias:
    pip install pyrealsense2 numpy opencv-python-headless
"""

import os
import sys
import time
import errno
import threading
import traceback

os.environ["QT_QPA_PLATFORM"]          = "xcb"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_LOGGING_RULES"]         = "*.debug=false;qt.qpa.*=false"

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERRO] pyrealsense2 nao encontrado.  pip install pyrealsense2")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
WIDTH, HEIGHT   = 640, 480
FPS             = 30            # 6 FPS e mais estavel em USB 2.1; troque para 15 se quiser
FRAME_TIMEOUT   = 2000         # ms — tempo maximo que wait_for_frames espera por frame
OUTPUT_DIR      = "output"
JPEG_QUALITY    = 95
WINDOW_NAME     = "D415 | ESPACO=salvar  Q=sair"

# ---------------------------------------------------------------------------
# Buffer compartilhado entre threads
# ---------------------------------------------------------------------------
_lock            = threading.Lock()
_latest_color    = None   # np.ndarray BGR uint8
_latest_depth    = None   # np.ndarray float32 metros
_capture_running = False
_frames_received = 0      # contador para diagnostico


def _capture_thread(pipeline, align, depth_scale):
    global _latest_color, _latest_depth, _capture_running, _frames_received
    consecutive_timeouts = 0

    print("[THREAD] Captura iniciada.")
    while _capture_running:
        try:
            frameset = pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT)
            aligned  = align.process(frameset)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img    = np.asanyarray(color_frame.get_data()).copy()
            depth_raw    = np.asanyarray(depth_frame.get_data())
            depth_meters = (depth_raw * depth_scale).astype(np.float32)

            with _lock:
                _latest_color    = color_img
                _latest_depth    = depth_meters
                _frames_received += 1

            consecutive_timeouts = 0

        except RuntimeError as e:
            msg = str(e)
            # "Frame didn't arrive within N" e o timeout normal do SDK
            if "arrive" in msg or "timeout" in msg.lower():
                consecutive_timeouts += 1
                # So imprime apos varias tentativas para nao poluir terminal
                if consecutive_timeouts % 5 == 1:
                    print(f"[THREAD] Aguardando frame (timeout #{consecutive_timeouts}) — normal em USB 2.1")
                if consecutive_timeouts > 30:
                    print("[THREAD] AVISO: Muitos timeouts. Verifique cabo/porta USB.")
                    consecutive_timeouts = 0
            else:
                print(f"[THREAD] Erro de captura: {msg}")
                time.sleep(0.2)

        except Exception as e:
            print(f"[THREAD] Excecao: {e}")
            time.sleep(0.2)

    print("[THREAD] Captura encerrada.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_dirs():
    for sub in ("rgb", "depth"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def frame_filename(counter, ext):
    return f"frame_{counter:05d}.{ext}"


def print_intrinsics(profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    print("\n--- Intrinsecos da camera (D415) ---")
    print(f"  Resolucao   : {intr.width} x {intr.height}")
    print(f"  fx          : {intr.fx:.4f}")
    print(f"  fy          : {intr.fy:.4f}")
    print(f"  ppx (cx)    : {intr.ppx:.4f}")
    print(f"  ppy (cy)    : {intr.ppy:.4f}")
    print(f"  Distorcao   : {intr.model}")
    print(f"  Coeficientes: {[round(c, 6) for c in intr.coeffs]}")
    print("------------------------------------\n")


def hardware_reset_device():
    print("[INIT] Procurando dispositivos RealSense...")
    ctx     = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("Nenhum dispositivo RealSense encontrado.")
    for dev in devices:
        name   = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"[INIT] Hardware reset: {name} (S/N {serial})")
        dev.hardware_reset()
    # Tempo generoso para re-enumeracao em USB 2.1
    print("[INIT] Aguardando re-enumeracao (5 s)...")
    time.sleep(5)
    print("[INIT] Reset concluido.\n")


def build_pipeline():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)
    print(f"[INIT] Iniciando pipeline {WIDTH}x{HEIGHT} @ {FPS} FPS ...")
    profile = pipeline.start(config)
    # Pausa pos-start: da tempo ao firmware para estabilizar o stream
    print("[INIT] Aguardando estabilizacao do firmware (2 s)...")
    time.sleep(2)
    print("[INIT] Pipeline pronto.\n")
    return pipeline, profile


def get_depth_scale(profile):
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"[INFO] Depth scale: {scale:.6f} m/unit")
    return scale


def warmup(pipeline, align, seconds=3):
    """
    Descarta frames pelo tempo especificado usando wait_for_frames.
    Usar segundos em vez de contagem e mais robusto para FPS baixo.
    Apos o warmup, pausa para o buffer interno do pipeline se reestabelecer.
    """
    print(f"[INIT] Aquecendo camera por {seconds} s ...")
    deadline = time.time() + seconds
    count = 0
    while time.time() < deadline:
        try:
            fs = pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT)
            align.process(fs)
            count += 1
        except RuntimeError:
            pass
    print(f"[INIT] Warmup concluido ({count} frames). Pausa de 1 s para reestabelecer buffer...")
    time.sleep(1)   # <-- pausa critica: deixa o buffer do pipeline se encher novamente
    print("[INIT] Camera pronta.\n")


def colorize_depth(depth_meters):
    MAX_DIST = 5.0
    norm = np.clip(depth_meters / MAX_DIST, 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def save_frame(counter, color_img, depth_meters):
    rgb_path   = os.path.join(OUTPUT_DIR, "rgb",   frame_filename(counter, "jpg"))
    depth_path = os.path.join(OUTPUT_DIR, "depth", frame_filename(counter, "npy"))

    ok, buf = cv2.imencode(".jpg", color_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        with open(rgb_path, "wb") as f:
            f.write(buf.tobytes())
    else:
        print(f"[ERRO] Falha ao codificar JPEG #{counter}")
        return

    np.save(depth_path, depth_meters)
    print(f"[SAVE] #{counter:05d}  {rgb_path}  |  {depth_path}")


def handle_bus_error(exc):
    msg = str(exc).lower()
    if "device or resource busy" in msg or getattr(exc, "errno", None) == errno.EBUSY:
        print("\n[ERRO] Dispositivo ocupado (errno 16 -- EBUSY).")
        print("       Libere com:  sudo fuser -k /dev/video*")
    elif any(k in msg for k in ("usb", "bus", "transport")):
        print("\n[ERRO] Erro de barramento USB.")
        print("       1. Reconecte o cabo USB fisicamente.")
        print("       2. Use porta USB 3.x direta (sem hub passivo).")
        print("       3. sudo dmesg | tail -20")
    elif any(k in msg for k in ("no device", "not found")):
        print("\n[ERRO] Nenhum dispositivo RealSense encontrado.")
        print("       rs-enumerate-devices")
    else:
        print(f"\n[ERRO] {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def run():
    global _capture_running

    make_output_dirs()

    try:
        hardware_reset_device()
    except RuntimeError as exc:
        handle_bus_error(exc)
        return

    try:
        pipeline, profile = build_pipeline()
    except RuntimeError as exc:
        handle_bus_error(exc)
        return

    print_intrinsics(profile)
    depth_scale = get_depth_scale(profile)
    align       = rs.align(rs.stream.color)

    # Warmup com pausa de reestabelecimento de buffer
    warmup(pipeline, align, seconds=3)

    # Inicia thread de captura
    _capture_running = True
    t = threading.Thread(
        target=_capture_thread,
        args=(pipeline, align, depth_scale),
        daemon=True,
        name="RealSenseCapture"
    )
    t.start()

    # Aguarda primeiro frame (timeout generoso para USB 2.1 a 6 FPS)
    print("[INIT] Aguardando primeiro frame ...")
    deadline = time.time() + 15.0
    while time.time() < deadline:
        with _lock:
            if _latest_color is not None:
                break
        time.sleep(0.1)
    else:
        print("[ERRO] Nenhum frame recebido em 15 s.")
        print("       Dicas:")
        print("         - Troque FPS = 6 no codigo se estiver usando 15")
        print("         - Reconecte o cabo USB em porta diferente")
        print("         - Execute:  rs-enumerate-devices")
        _capture_running = False
        t.join(timeout=3)
        pipeline.stop()
        return

    print(f"[INIT] Primeiro frame recebido! Iniciando preview.\n")

    frame_counter = 0

    # Detecta se OpenCV tem suporte a GUI (opencv-python vs opencv-python-headless)
    gui_available = False
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
        gui_available = True
        print("[INFO] GUI detectada (OpenCV com GTK/Qt).")
    except cv2.error:
        print("[INFO] OpenCV sem suporte a GUI (headless). Modo terminal ativado.")
        print("       Para ativar preview visual:")
        print("         pip uninstall opencv-python-headless")
        print("         pip install opencv-python")
        print()

    if gui_available:
        print(f"[RUN] ESPACO = salvar frame  |  Q / ESC = sair\n")
        _run_gui_loop(frame_counter)
    else:
        print(f"[RUN] ENTER = salvar frame  |  Ctrl+C = sair\n")
        _run_terminal_loop(frame_counter)

    _capture_running = False
    t.join(timeout=3)
    pipeline.stop()
    if gui_available:
        cv2.destroyAllWindows()
    print(f"[FIN] Sessao encerrada.")


def _run_gui_loop(frame_counter):
    """Loop principal com preview visual OpenCV."""
    try:
        while True:
            with _lock:
                if _latest_color is not None:
                    color_snap = _latest_color.copy()
                    depth_snap = _latest_depth.copy()
                else:
                    color_snap = None

            if color_snap is not None:
                depth_vis = colorize_depth(depth_snap)
                canvas    = np.hstack([color_snap, depth_vis])
                label     = f"Salvos: {frame_counter}  |  ESPACO=salvar  Q=sair"
                cv2.putText(canvas, label, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print("\n[RUN] Encerrando.")
                break

            if key == ord(' '):
                with _lock:
                    if _latest_color is None:
                        print("[WARN] Sem frame disponivel.")
                        continue
                    snap_c = _latest_color.copy()
                    snap_d = _latest_depth.copy()
                frame_counter += 1
                save_frame(frame_counter, snap_c, snap_d)

    except KeyboardInterrupt:
        print("\n[RUN] Interrompido (Ctrl+C).")


def _run_terminal_loop(frame_counter):
    """
    Loop headless: captura via input() no terminal.
    ENTER = salvar frame atual | 'q' + ENTER = sair
    Util quando opencv-python-headless esta instalado.
    """
    import select

    print("  Digite ENTER para salvar | 'q' + ENTER para sair")
    print("  (frames sendo capturados continuamente em background)\n")

    try:
        while True:
            # input() nao bloqueante via select (Linux)
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'q':
                    print("\n[RUN] Encerrando.")
                    break
                else:
                    # Qualquer tecla (incluindo ENTER vazio) salva o frame
                    with _lock:
                        if _latest_color is None:
                            print("[WARN] Sem frame disponivel.")
                            continue
                        snap_c = _latest_color.copy()
                        snap_d = _latest_depth.copy()
                    frame_counter += 1
                    save_frame(frame_counter, snap_c, snap_d)

    except KeyboardInterrupt:
        print("\n[RUN] Interrompido (Ctrl+C).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run()
    except RuntimeError as e:
        handle_bus_error(e)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        traceback.print_exc()
        sys.exit(2)
