#!/usr/bin/env python3
"""
realsense_capture.py
====================
Captura RGB-D para Intel RealSense D455 (D415 também suportada).

Modos de uso:
  python realsense_capture.py                    → captura direta (ESPAÇO=salvar)
  python realsense_capture.py --bag [arquivo]    → grava sessão completa em .bag
  python realsense_capture.py --extract arq.bag  → extrai todos os frames de um .bag

Saída compatível com HorizonGS:
    output/
    ├── rgb/    frame_00001.jpg
    ├── depth/  frame_00001.npy   ← uint16 bruto z16 (sem colorização)
    └── meta.json                 ← depth_scale e intrínsecos da câmera
"""

import os
import sys
import time
import errno
import threading
import traceback
import json
import argparse
from datetime import datetime

os.environ["QT_QPA_PLATFORM"]          = "xcb"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_LOGGING_RULES"]         = "*.debug=false;qt.qpa.*=false"

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERRO] pyrealsense2 não encontrado.  pip install pyrealsense2")
    sys.exit(1)

# ─── Configurações ────────────────────────────────────────────────────────────
WIDTH, HEIGHT       = 848, 480      # resolução nativa da D455 (D415: 640x480)
FPS                 = 30
FRAME_TIMEOUT       = 2000          # ms
OUTPUT_DIR          = "output"
JPEG_QUALITY        = 95
WINDOW_NAME         = "D455 | ESPAÇO=salvar  Q=sair"

# Otimizações de preview para CPUs fracas
PREVIEW_WAITKEY_MS  = 100
PREVIEW_SCALE       = 0.5
DEPTH_VIS_EVERY_N   = 3

# ─── Buffer compartilhado entre threads ───────────────────────────────────────
_lock            = threading.Lock()
_latest_color    = None   # np.ndarray BGR uint8
_latest_depth    = None   # np.ndarray uint16 — valores brutos z16 SEM colorização
_capture_running = False
_frames_received = 0


# ─── Helpers gerais ───────────────────────────────────────────────────────────

def make_output_dirs():
    for sub in ("rgb", "depth"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def frame_filename(counter, ext):
    return f"frame_{counter:05d}.{ext}"


def save_meta(profile, depth_scale):
    """Salva depth_scale e intrínsecos em meta.json."""
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    meta = {
        "depth_scale_m_per_unit": depth_scale,
        "width":  intr.width,
        "height": intr.height,
        "fx":     intr.fx,
        "fy":     intr.fy,
        "cx":     intr.ppx,
        "cy":     intr.ppy,
        "distortion_model":  str(intr.model),
        "distortion_coeffs": [round(c, 8) for c in intr.coeffs],
    }
    path = os.path.join(OUTPUT_DIR, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] Intrínsecos salvos em {path}")
    return meta


def print_intrinsics(profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    print("\n--- Intrínsecos da câmera (D455) ---")
    print(f"  Resolução   : {intr.width} x {intr.height}")
    print(f"  fx          : {intr.fx:.4f}")
    print(f"  fy          : {intr.fy:.4f}")
    print(f"  ppx (cx)    : {intr.ppx:.4f}")
    print(f"  ppy (cy)    : {intr.ppy:.4f}")
    print(f"  Distorção   : {intr.model}")
    print(f"  Coeficientes: {[round(c, 6) for c in intr.coeffs]}")
    print("------------------------------------\n")


def get_depth_scale(profile):
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"[INFO] Depth scale: {scale:.6f} m/unit  (profundidade_m = pixel * {scale:.6f})")
    return scale


def colorize_depth_preview(depth_u16):
    """Coloriza SOMENTE para preview visual. O arquivo salvo é sempre uint16 bruto."""
    MAX_DIST_MM = 5000
    norm = np.clip(depth_u16.astype(np.float32) / MAX_DIST_MM, 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def save_frame(counter, color_bgr, depth_u16):
    """
    Salva um par RGB+Depth.
    depth_u16: np.ndarray uint16 — valores z16 brutos (milímetros × depth_scale).
    Não há colorização: o array é o dado direto do sensor.
    """
    rgb_path   = os.path.join(OUTPUT_DIR, "rgb",   frame_filename(counter, "jpg"))
    depth_path = os.path.join(OUTPUT_DIR, "depth", frame_filename(counter, "npy"))

    ok, buf = cv2.imencode(".jpg", color_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        with open(rgb_path, "wb") as f:
            f.write(buf.tobytes())
    else:
        print(f"[ERRO] Falha ao codificar JPEG #{counter}")
        return

    # Salva depth como uint16 bruto — sem nenhuma transformação ou colorização
    np.save(depth_path, depth_u16)
    print(f"[SAVE] #{counter:05d}  {rgb_path}  |  {depth_path}")


def handle_bus_error(exc):
    msg = str(exc).lower()
    if "device or resource busy" in msg or getattr(exc, "errno", None) == errno.EBUSY:
        print("\n[ERRO] Dispositivo ocupado (EBUSY).")
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
    print("[INIT] Aguardando re-enumeração (5 s)...")
    time.sleep(5)
    print("[INIT] Reset concluído.\n")


def build_pipeline(bag_path=None):
    """
    Constrói o pipeline.
    - bag_path=None → câmera ao vivo
    - bag_path=str  → gravação para arquivo .bag (streams brutos rgb + z16)
    """
    pipeline = rs.pipeline()
    config   = rs.config()

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)

    if bag_path:
        config.enable_record_to_file(bag_path)
        print(f"[BAG] Gravando em: {bag_path}")

    print(f"[INIT] Iniciando pipeline {WIDTH}x{HEIGHT} @ {FPS} FPS ...")
    profile = pipeline.start(config)
    print("[INIT] Aguardando estabilização do firmware (2 s)...")
    time.sleep(2)
    print("[INIT] Pipeline pronto.\n")
    return pipeline, profile


def warmup(pipeline, align, seconds=3):
    print(f"[INIT] Aquecendo câmera por {seconds} s ...")
    deadline = time.time() + seconds
    count = 0
    while time.time() < deadline:
        try:
            fs = pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT)
            align.process(fs)
            count += 1
        except RuntimeError:
            pass
    print(f"[INIT] Warmup concluído ({count} frames). Pausa de 1 s...")
    time.sleep(1)
    print("[INIT] Câmera pronta.\n")


# ─── Thread de captura (modos ao vivo e gravação .bag) ────────────────────────

def _capture_thread(pipeline, align):
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

            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            # Depth bruto: uint16, unidades nativas do sensor (z16)
            depth_u16 = np.asanyarray(depth_frame.get_data()).copy()

            with _lock:
                _latest_color    = color_bgr
                _latest_depth    = depth_u16
                _frames_received += 1

            consecutive_timeouts = 0

        except RuntimeError as e:
            msg = str(e)
            if "arrive" in msg or "timeout" in msg.lower():
                consecutive_timeouts += 1
                if consecutive_timeouts % 5 == 1:
                    print(f"[THREAD] Aguardando frame (timeout #{consecutive_timeouts})")
                if consecutive_timeouts > 30:
                    print("[THREAD] AVISO: Muitos timeouts. Verifique cabo/porta USB.")
                    consecutive_timeouts = 0
            else:
                print(f"[THREAD] Erro: {msg}")
                time.sleep(0.2)
        except Exception as e:
            print(f"[THREAD] Exceção: {e}")
            time.sleep(0.2)

    print("[THREAD] Captura encerrada.")


# ─── Modo 1: Captura direta ───────────────────────────────────────────────────

def run_direct():
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
    warmup(pipeline, align, seconds=3)
    save_meta(profile, depth_scale)

    _capture_running = True
    t = threading.Thread(target=_capture_thread, args=(pipeline, align), daemon=True)
    t.start()

    print("[INIT] Aguardando primeiro frame ...")
    deadline = time.time() + 15.0
    while time.time() < deadline:
        with _lock:
            if _latest_color is not None:
                break
        time.sleep(0.1)
    else:
        print("[ERRO] Nenhum frame recebido em 15 s.")
        _capture_running = False
        t.join(timeout=3)
        pipeline.stop()
        return

    print("[INIT] Primeiro frame recebido!\n")

    frame_counter = 0
    gui_available = False
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
        gui_available = True
        print("[INFO] GUI disponível.")
    except cv2.error:
        print("[INFO] OpenCV headless — modo terminal.")

    if gui_available:
        frame_counter = _run_gui_loop(frame_counter)
    else:
        frame_counter = _run_terminal_loop(frame_counter)

    _capture_running = False
    t.join(timeout=3)
    pipeline.stop()
    if gui_available:
        cv2.destroyAllWindows()
    print(f"[FIN] Sessão encerrada. {frame_counter} frame(s) salvos.")


def _run_gui_loop(frame_counter):
    _cached_depth_vis = None
    try:
        while True:
            with _lock:
                frames_now = _frames_received
                if _latest_color is not None:
                    color_snap = _latest_color.copy()
                    depth_snap = _latest_depth.copy()
                else:
                    color_snap = None

            if color_snap is not None:
                if _cached_depth_vis is None or (frames_now % DEPTH_VIS_EVERY_N == 0):
                    _cached_depth_vis = colorize_depth_preview(depth_snap)

                canvas = np.hstack([color_snap, _cached_depth_vis])
                if PREVIEW_SCALE != 1.0:
                    canvas = cv2.resize(canvas, (0, 0),
                                        fx=PREVIEW_SCALE, fy=PREVIEW_SCALE,
                                        interpolation=cv2.INTER_NEAREST)

                label = f"Salvos: {frame_counter}  |  ESPACO=salvar  Q=sair"
                cv2.putText(canvas, label, (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(PREVIEW_WAITKEY_MS) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print("\n[RUN] Encerrando.")
                break

            if key == ord(' '):
                with _lock:
                    if _latest_color is None:
                        print("[WARN] Sem frame disponível.")
                        continue
                    snap_c = _latest_color.copy()
                    snap_d = _latest_depth.copy()
                frame_counter += 1
                save_frame(frame_counter, snap_c, snap_d)

    except KeyboardInterrupt:
        print("\n[RUN] Interrompido (Ctrl+C).")

    return frame_counter


def _run_terminal_loop(frame_counter):
    import select
    print("  ENTER = salvar frame atual | 'q' + ENTER = sair\n")
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'q':
                    print("\n[RUN] Encerrando.")
                    break
                else:
                    with _lock:
                        if _latest_color is None:
                            print("[WARN] Sem frame disponível.")
                            continue
                        snap_c = _latest_color.copy()
                        snap_d = _latest_depth.copy()
                    frame_counter += 1
                    save_frame(frame_counter, snap_c, snap_d)
    except KeyboardInterrupt:
        print("\n[RUN] Interrompido (Ctrl+C).")
    return frame_counter


# ─── Modo 2: Gravação .bag ────────────────────────────────────────────────────

def run_record_bag(bag_path):
    """
    Grava RGB + depth z16 bruto diretamente num arquivo .bag.
    Nenhum filtro de colorização é aplicado — o .bag contém dados brutos.
    """
    global _capture_running

    try:
        hardware_reset_device()
    except RuntimeError as exc:
        handle_bus_error(exc)
        return

    try:
        pipeline, profile = build_pipeline(bag_path=bag_path)
    except RuntimeError as exc:
        handle_bus_error(exc)
        return

    print_intrinsics(profile)
    depth_scale = get_depth_scale(profile)
    align       = rs.align(rs.stream.color)
    warmup(pipeline, align, seconds=3)

    # Salva meta mesmo no modo bag para referência futura
    make_output_dirs()
    save_meta(profile, depth_scale)

    _capture_running = True
    t = threading.Thread(target=_capture_thread, args=(pipeline, align), daemon=True)
    t.start()

    print("[BAG] Gravando... Pressione Q ou Ctrl+C para parar.\n")

    gui_available = False
    try:
        cv2.namedWindow(WINDOW_NAME + " [GRAVANDO BAG]", cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
        gui_available = True
    except cv2.error:
        pass

    _cached_depth_vis = None
    try:
        while True:
            if gui_available:
                with _lock:
                    frames_now = _frames_received
                    color_snap = _latest_color.copy() if _latest_color is not None else None
                    depth_snap = _latest_depth.copy() if _latest_depth is not None else None

                if color_snap is not None:
                    if _cached_depth_vis is None or (frames_now % DEPTH_VIS_EVERY_N == 0):
                        _cached_depth_vis = colorize_depth_preview(depth_snap)
                    canvas = np.hstack([color_snap, _cached_depth_vis])
                    if PREVIEW_SCALE != 1.0:
                        canvas = cv2.resize(canvas, (0, 0),
                                            fx=PREVIEW_SCALE, fy=PREVIEW_SCALE,
                                            interpolation=cv2.INTER_NEAREST)
                    with _lock:
                        label = f"[REC] Frames: {_frames_received}  |  Q=parar"
                    cv2.putText(canvas, label, (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imshow(WINDOW_NAME + " [GRAVANDO BAG]", canvas)

                key = cv2.waitKey(PREVIEW_WAITKEY_MS) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
            else:
                time.sleep(0.5)
                with _lock:
                    n = _frames_received
                print(f"\r[BAG] {n} frames gravados... (Ctrl+C para parar)", end="", flush=True)

    except KeyboardInterrupt:
        print()

    _capture_running = False
    t.join(timeout=3)
    pipeline.stop()
    if gui_available:
        cv2.destroyAllWindows()

    with _lock:
        total = _frames_received
    print(f"\n[FIN] Gravação encerrada. {total} frames gravados em {bag_path}")


# ─── Modo 3: Extração de .bag ─────────────────────────────────────────────────

def run_extract_bag(bag_path):
    """
    Lê um arquivo .bag e extrai todos os frames como:
      - rgb/frame_NNNNN.jpg   → imagem colorida
      - depth/frame_NNNNN.npy → uint16 bruto z16 (sem colorização)
    """
    if not os.path.isfile(bag_path):
        print(f"[ERRO] Arquivo não encontrado: {bag_path}")
        sys.exit(1)

    make_output_dirs()

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    print(f"[EXTRACT] Abrindo: {bag_path}")
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        print(f"[ERRO] Falha ao abrir .bag: {exc}")
        sys.exit(1)

    # Configura playback para não repetir e tentar manter velocidade real
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    depth_scale = get_depth_scale(profile)
    save_meta(profile, depth_scale)

    align         = rs.align(rs.stream.color)
    frame_counter = 0

    print("[EXTRACT] Extraindo frames... (Ctrl+C para cancelar)\n")
    try:
        while True:
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                # Fim do arquivo
                break

            aligned     = align.process(frameset)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            depth_u16 = np.asanyarray(depth_frame.get_data()).copy()  # uint16 bruto

            frame_counter += 1
            save_frame(frame_counter, color_bgr, depth_u16)

    except KeyboardInterrupt:
        print("\n[EXTRACT] Cancelado pelo usuário.")

    pipeline.stop()
    print(f"\n[FIN] {frame_counter} frame(s) extraídos de {bag_path}")
    print(f"      depth salvo como uint16 bruto (multiplique por depth_scale para obter metros)")
    print(f"      depth_scale = {depth_scale:.6f}  (ver output/meta.json)")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Captura RGB-D para RealSense D455",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python realsense_capture.py                       # captura direta\n"
            "  python realsense_capture.py --bag                 # grava sessao.bag\n"
            "  python realsense_capture.py --bag minha.bag       # grava minha.bag\n"
            "  python realsense_capture.py --extract sessao.bag  # extrai frames\n"
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--bag",
        nargs="?",
        const="",
        metavar="ARQUIVO",
        help="Grava sessão em .bag (padrão: sessao_YYYYMMDD_HHMMSS.bag)",
    )
    group.add_argument(
        "--extract",
        metavar="ARQUIVO.bag",
        help="Extrai todos os frames de um .bag existente",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"Diretório de saída (padrão: {OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output

    try:
        if args.extract is not None:
            run_extract_bag(args.extract)

        elif args.bag is not None:
            if args.bag == "":
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                bag_path = f"sessao_{ts}.bag"
            else:
                bag_path = args.bag
            run_record_bag(bag_path)

        else:
            run_direct()

    except RuntimeError as e:
        handle_bus_error(e)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        traceback.print_exc()
        sys.exit(2)
