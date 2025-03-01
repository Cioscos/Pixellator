import queue
import cv2
import numpy as np
import argparse
import time
import logging
import threading
from PIL import Image
from multiprocessing import Process, Queue, Event
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl
from ascii_conv import frame_to_ascii_cython  # Modulo Cython compilato

# Configura il logging
logging.basicConfig(
    filename="ascii_video.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode='w'
)

def extract_frames(video_path, frame_queue, fps, stop_event):
    """
    Estrae i frame dal video e li inserisce nella coda frame_queue.
    Quando il video finisce o l'utente interrompe, inserisce None per segnalare la fine.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        return
    frame_time = 1 / fps
    try:
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
            time.sleep(frame_time)
    except KeyboardInterrupt:
        print("\n[!] Interruzione nel processo di estrazione frame.")
    finally:
        cap.release()
        frame_queue.put(None)  # Segnala la fine

def process_frames(input_queue, output_queue, stop_event, new_width):
    """
    Processo separato per convertire i frame in ASCII usando la funzione Cython.
    Misura il tempo di conversione per ciascun frame e lo logga.
    """
    frame_idx = 0
    while not stop_event.is_set():
        try:
            frame = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if frame is None:
            break
        conv_start = time.time()
        ascii_art = frame_to_ascii_cython(frame, new_width)
        conv_end = time.time()
        conversion_time_ms = (conv_end - conv_start) * 1000
        logging.info(f"[PROCESS] Frame {frame_idx} - Conversione: {conversion_time_ms:.2f} ms")
        output_queue.put(ascii_art)
        frame_idx += 1

def update_frames(output_queue, app, control, stop_event, log_fps, log_perf):
    """
    Legge i frame ASCII dalla coda output_queue e aggiorna lo schermo con prompt_toolkit.
    Misura il tempo di aggiornamento (stampa) e logga i valori (in ms) e gli FPS.
    """
    frame_count = 0
    fps_count = 0
    fps_start = time.time()
    while not stop_event.is_set():
        try:
            ascii_frame = output_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        print_start = time.time()
        control.text = ANSI(ascii_frame)
        app.invalidate()
        print_end = time.time()
        printing_time_ms = (print_end - print_start) * 1000

        if log_perf:
            logging.info(f"[UPDATE] Frame {frame_count} - Stampa: {printing_time_ms:.2f} ms")
        frame_count += 1
        fps_count += 1

        now = time.time()
        if now - fps_start >= 1.0:
            if log_fps:
                logging.info(f"[UPDATE] FPS effettivi: {fps_count}")
            fps_count = 0
            fps_start = now

def main():
    parser = argparse.ArgumentParser(
        description="Real-time ASCII video using multiprocessing.Process and prompt_toolkit with Cython conversion."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of actual FPS")
    parser.add_argument("--log_performance", action="store_true", help="Enable logging of performance metrics")
    args = parser.parse_args()

    # Code per il passaggio dei dati tra i processi
    frame_queue = Queue(maxsize=args.fps)      # Coda di input per i frame raw
    ascii_queue = Queue(maxsize=args.fps)      # Coda di output per i frame ASCII
    stop_event = Event()                       # Evento per la terminazione pulita

    # Avvia il processo di estrazione dei frame
    extractor_process = Process(target=extract_frames, args=(args.video_path, frame_queue, args.fps, stop_event))
    extractor_process.start()

    # Avvia il processo di conversione frame -> ASCII
    processor_process = Process(target=process_frames, args=(frame_queue, ascii_queue, stop_event, args.width))
    processor_process.start()

    # Definisci i key bindings per uscire (premi "q" o Ctrl+C)
    kb = KeyBindings()
    @kb.add("q")
    def exit_(event):
        stop_event.set()
        event.app.exit()
    @kb.add("c-c")
    def exit_ctrl_c(event):
        stop_event.set()
        event.app.exit()

    # Configura l'interfaccia di prompt_toolkit
    control = FormattedTextControl(text=ANSI("Loading..."))
    root_container = Window(content=control)
    layout = Layout(root_container)
    app = Application(layout=layout, key_bindings=kb, full_screen=True)

    # Avvia il thread per aggiornare lo schermo
    updater_thread = threading.Thread(
        target=update_frames,
        args=(ascii_queue, app, control, stop_event, args.log_fps, args.log_performance),
        daemon=True
    )
    updater_thread.start()

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        updater_thread.join(timeout=1)
        extractor_process.terminate()
        processor_process.terminate()
        extractor_process.join()
        processor_process.join()
        frame_queue.close()
        ascii_queue.close()
        frame_queue.cancel_join_thread()
        ascii_queue.cancel_join_thread()
        print("[✔] Terminazione completata.")

if __name__ == '__main__':
    main()
