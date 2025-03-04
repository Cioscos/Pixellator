import queue
from typing import List, Tuple

import cv2
import numpy as np
import argparse
import time
import logging
import threading
import sys
import shutil
import curses
from numba import jit
from multiprocessing import Process, Queue, Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool  # Pool basato su thread (utile su Windows)

# Configurazione del logging per salvare i tempi (in ms) di conversione e rendering.
logging.basicConfig(
    filename="ascii_video.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode='w'
)

# Stringa dei caratteri ASCII ordinati per densità (dal più chiaro al più scuro).
ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"
# Precompone una Look-Up Table (LUT) per mappare un valore di luminosità (0-255) in un carattere ASCII.
ASCII_LUT = np.array(list(ASCII_CHARS))


def frame_to_ascii(frame_data):
    """
    Converte un frame video in una stringa ASCII a colori usando operazioni vettorializzate.
    Utilizza sequenze ANSI per colorare ciascun carattere.

    Parametri:
        frame_data (tuple): (frame, new_width) dove:
            - frame: array NumPy contenente il frame (in formato BGR).
            - new_width (int): larghezza desiderata dell'output ASCII.

    Ritorna:
        str: stringa contenente il frame convertito in ASCII con escape ANSI.
    """
    frame, new_width = frame_data
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)

    # Ridimensiona il frame (operazione eseguita in C)
    resized = cv2.resize(frame, (new_width, new_height))
    # Converte il frame da BGR a RGB
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Calcola la luminosità come media dei canali RGB
    brightness = np.mean(rgb_frame, axis=2).astype(np.uint8)
    # Mappa la luminosità in un indice per la LUT
    char_indices = (brightness.astype(np.uint16) * (len(ASCII_CHARS) - 1) // 255).astype(np.uint8)
    ascii_chars = ASCII_LUT[char_indices]

    # Converte ciascun canale RGB in stringa
    r_str = np.char.mod("%d", rgb_frame[:, :, 0])
    g_str = np.char.mod("%d", rgb_frame[:, :, 1])
    b_str = np.char.mod("%d", rgb_frame[:, :, 2])

    # Costruisce la stringa ANSI per ciascun pixel:
    # Formato: "\033[38;2;{r};{g};{b}m{char}\033[0m"
    prefix = "\033[38;2;"
    mid = "m"
    suffix = "\033[0m"
    ansi = np.char.add(prefix, r_str)
    ansi = np.char.add(ansi, ";")
    ansi = np.char.add(ansi, g_str)
    ansi = np.char.add(ansi, ";")
    ansi = np.char.add(ansi, b_str)
    ansi = np.char.add(ansi, mid)
    ansi = np.char.add(ansi, ascii_chars)
    ansi = np.char.add(ansi, suffix)

    # Unisce le righe per ottenere la stringa finale
    ascii_str = "\n".join("".join(row) for row in ansi)
    return ascii_str


def frame_to_ascii_curses(frame_data):
    """
    Converte un frame video in una stringa ASCII senza sequenze ANSI, adatto al rendering con curses.
    Questa funzione non aggiunge informazioni sul colore, semplificando la visualizzazione in curses.

    Parametri:
        frame_data (tuple): (frame, new_width) dove:
            - frame: array NumPy contenente il frame (in formato BGR).
            - new_width (int): larghezza desiderata dell'output ASCII.

    Ritorna:
        str: stringa contenente il frame convertito in ASCII.
    """
    frame, new_width = frame_data
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)

    resized = cv2.resize(frame, (new_width, new_height))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    brightness = np.mean(rgb_frame, axis=2).astype(np.uint8)
    char_indices = (brightness.astype(np.uint16) * (len(ASCII_CHARS) - 1) // 255).astype(np.uint8)
    ascii_chars = ASCII_LUT[char_indices]

    # Costruisce la stringa ASCII senza escape di colore
    ascii_str = "\n".join("".join(row) for row in ascii_chars)
    return ascii_str


def frame_to_ascii_curses_color(frame_data) -> List[List[Tuple[str, int]]]:
    """
    Converte un frame video in una rappresentazione ASCII colorata adatta per il rendering
    con la libreria curses. Per ogni cella restituisce una tupla (carattere, color_index) dove
    color_index è l'indice del colore in xterm 256 colori.

    Parametri:
        frame_data (tuple): (frame, new_width) dove:
            - frame: array NumPy contenente il frame (in formato BGR).
            - new_width (int): larghezza desiderata dell'output ASCII.

    Ritorna:
        List[List[Tuple[str, int]]]: Rappresentazione 2D del frame, dove ogni elemento
                                     è una tupla contenente il carattere ASCII e il colore.
    """
    frame, new_width = frame_data
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    # Assicura che l'altezza non sia zero
    new_height = max(int(aspect_ratio * new_width * 0.5), 1)

    # Ridimensiona il frame
    resized = cv2.resize(frame, (new_width, new_height))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Calcola la luminosità per mappare in carattere ASCII
    brightness = np.mean(rgb_frame, axis=2).astype(np.uint8)
    char_indices = (brightness.astype(np.uint16) * (len(ASCII_CHARS) - 1) // 255).astype(np.uint8)
    ascii_chars = np.array(list(ASCII_CHARS))[char_indices]

    # Converti i canali in un tipo a 16 bit per evitare overflow
    r_channel = rgb_frame[:, :, 0].astype(np.uint16)
    g_channel = rgb_frame[:, :, 1].astype(np.uint16)
    b_channel = rgb_frame[:, :, 2].astype(np.uint16)

    # Calcola l'indice colore in xterm 256 colori.
    # Mappa ogni canale (0-255) in un range 0-5.
    r6 = (r_channel * 6) // 256
    g6 = (g_channel * 6) // 256
    b6 = (b_channel * 6) // 256
    color_indices = 16 + 36 * r6 + 6 * g6 + b6

    # Costruisci la rappresentazione 2D: per ogni cella una tupla (carattere, color_index)
    frame_ascii = []
    for i in range(new_height):
        row = []
        for j in range(new_width):
            row.append((str(ascii_chars[i, j]), int(color_indices[i, j])))
        frame_ascii.append(row)

    return frame_ascii


def extract_frames(video_path, raw_queue, fps):
    """
    Estrae i frame da un video e li inserisce nella coda raw_queue.

    Parametri:
        video_path (str): percorso del file video.
        raw_queue (Queue): coda per passare i frame grezzi.
        fps (int): numero di frame al secondo da estrarre.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        return
    frame_time = 1 / fps
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_queue.put(frame)
            time.sleep(frame_time)
    except KeyboardInterrupt:
        print("\n[!] Interruzione nell'estrazione dei frame.")
    finally:
        cap.release()
        raw_queue.put(None)  # Segnala la fine dello stream


def convert_frames(raw_queue, ascii_queue, pool, batch_size, new_width, stop_event, conversion_function):
    """
    Legge i frame grezzi dalla raw_queue e li elabora in batch con un pool parallelo.
    Questa versione accumula quanti più frame sono disponibili fino a un massimo di batch_size
    senza aspettare inutilmente, riducendo così i tempi di attesa.

    Parametri:
        raw_queue (Queue): coda dei frame grezzi.
        ascii_queue (Queue): coda dei frame convertiti in ASCII insieme al tempo di conversione.
        pool (Pool): pool di processi o thread per la conversione.
        batch_size (int): numero massimo di frame da elaborare insieme.
        new_width (int): larghezza desiderata per l'output ASCII.
        stop_event (threading.Event): evento per terminare il ciclo.
        conversion_function (callable): funzione da utilizzare per convertire un frame in ASCII.
    """
    while not stop_event.is_set():
        batch = []
        # Accumula quanti più frame sono disponibili fino a un massimo di batch_size
        for _ in range(batch_size):
            try:
                frame = raw_queue.get(timeout=0.01)
            except queue.Empty:
                break
            if frame is None:
                stop_event.set()
                break
            batch.append((frame, new_width))

        if not batch:
            continue

        conv_start = time.perf_counter()
        try:
            ascii_frames = pool.map(conversion_function, batch, chunksize=len(batch))
        except Exception as e:
            print("Errore durante la conversione dei frame:", e)
            break
        conv_end = time.perf_counter()
        conversion_time_ms = ((conv_end - conv_start) * 1000) / len(batch)

        # Invia nella coda per ogni frame una tupla (ascii_frame, conversion_time_ms)
        for af in ascii_frames:
            ascii_queue.put((af, conversion_time_ms))


@jit(nopython=True)
def fast_diff_lines(new_lines: List[str], old_lines: List[str], max_lines: int) -> List[int]:
    """
    Computazione accelerata delle differenze tra righe mediante Numba JIT.

    Parametri:
        new_lines (List[str]): Lista delle nuove righe.
        old_lines (List[str]): Lista delle righe precedenti.
        max_lines (int): Numero massimo di righe da confrontare.

    Ritorna:
        List[int]: Indici delle righe che risultano differenti.
    """
    diff_indices = []  # Lista per salvare gli indici delle righe modificate
    n_new = len(new_lines)
    n_old = len(old_lines)
    for i in range(max_lines):
        # Se entrambi gli indici sono fuori dai limiti, passa al prossimo ciclo
        if i >= n_new and i >= n_old:
            continue

        new_line = new_lines[i] if i < n_new else ""
        old_line = old_lines[i] if i < n_old else ""

        # Se le lunghezze sono differenti, la riga va aggiornata
        if len(new_line) != len(old_line):
            diff_indices.append(i)
            continue

        # Confronto carattere per carattere
        for j in range(len(new_line)):
            if j >= len(old_line) or new_line[j] != old_line[j]:
                diff_indices.append(i)
                break

    return diff_indices


def render_frames_sys_partial(ascii_queue, stop_event, log_fps=False, log_performance=False):
    """
    Legge i frame ASCII dalla ascii_queue e aggiorna parzialmente il display
    del terminale utilizzando sys.stdout.write con calcolo accelerato delle differenze.
    Vengono riscritte solo le righe modificate.

    Parametri:
        ascii_queue (Queue): coda dei frame ASCII, contenente tuple (ascii_frame, conversion_time_ms).
        stop_event (threading.Event): evento per terminare il ciclo.
        log_fps (bool): se True, logga il numero di aggiornamenti al secondo.
        log_performance (bool): se True, logga il tempo impiegato nel rendering.
    """
    prev_frame_lines = None
    prev_terminal_size = shutil.get_terminal_size()
    frame_counter = 0
    fps_count = 0
    fps_start = time.perf_counter()

    # Sequenze di escape precompilate
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    CLEAR_SCREEN = "\033[2J\033[H"
    GOTO_FORMAT = "\033[{};1H"

    output_buffer = []

    # Nasconde il cursore
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        while not stop_event.is_set():
            try:
                ascii_frame, conversion_time_ms = ascii_queue.get(timeout=0.01)
            except queue.Empty:
                #time.sleep(0.001)
                continue

            if log_performance:
                rendering_start = time.perf_counter()

            frame_lines = ascii_frame.split("\n")
            current_terminal_size = shutil.get_terminal_size()
            terminal_resized = current_terminal_size != prev_terminal_size

            if terminal_resized:
                sys.stdout.write(CLEAR_SCREEN)
                sys.stdout.flush()
                prev_frame_lines = None
                prev_terminal_size = current_terminal_size

            output_buffer.clear()

            if prev_frame_lines is None:
                output_buffer.append(ascii_frame)
            else:
                max_lines = max(len(frame_lines), len(prev_frame_lines))
                diff_indices = fast_diff_lines(frame_lines, prev_frame_lines, max_lines)
                for i in diff_indices:
                    if i < len(frame_lines):
                        new_line = frame_lines[i]
                        if i < len(prev_frame_lines):
                            old_line = prev_frame_lines[i]
                            if len(new_line) < len(old_line):
                                new_line += " " * (len(old_line) - len(new_line))
                        output_buffer.append(f"{GOTO_FORMAT.format(i + 1)}{new_line}")

            if output_buffer:
                sys.stdout.write("".join(output_buffer))
                sys.stdout.flush()

            prev_frame_lines = frame_lines

            if log_performance:
                rendering_end = time.perf_counter()
                total_rendering_time_ms = (rendering_end - rendering_start) * 1000
                logging.info(
                    f"Frame {frame_counter} - Conversion: {conversion_time_ms:.2f} ms, "
                    f"Total Rendering: {total_rendering_time_ms:.2f} ms, "
                    f"Changed lines: {len(output_buffer)}"
                )

            frame_counter += 1
            fps_count += 1

            now = time.perf_counter()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                if log_fps:
                    logging.info(f"[LOG] FPS display (sys partial): {fps_count / elapsed:.1f}")
                fps_count = 0
                fps_start = now
    finally:
        try:
            term_height = shutil.get_terminal_size().lines
        except Exception:
            term_height = 25
        sys.stdout.write("\n" * term_height)
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


def render_frames_curses_color(ascii_queue, stop_event, log_fps=False, log_performance=False):
    """
    Renderizza i frame ASCII colorati utilizzando la libreria curses.
    Per ogni riga, vengono aggiornati in blocco i segmenti contigui che presentano differenze rispetto al frame precedente,
    riducendo il numero di chiamate a curses.addstr e migliorando le performance.

    Parametri:
        ascii_queue (Queue): coda contenente tuple (ascii_frame, conversion_time_ms) dove
                             ascii_frame è una rappresentazione 2D del frame come lista di liste
                             di tuple (carattere, color_index).
        stop_event (threading.Event): evento usato per terminare il ciclo di rendering.
        log_fps (bool): se True, logga il numero di aggiornamenti al secondo.
        log_performance (bool): se True, logga il tempo impiegato per il rendering di ogni frame.
    """

    def curses_loop(stdscr):
        # Inizializza il supporto colori in curses
        curses.start_color()
        curses.use_default_colors()
        stdscr.nodelay(True)
        curses.curs_set(0)  # Nasconde il cursore
        frame_counter = 0
        fps_count = 0
        fps_start = time.perf_counter()
        prev_frame = []  # Rappresentazione 2D del frame precedente
        max_y, max_x = stdscr.getmaxyx()

        # Cache per le coppie colore: mappa color_index -> pair_number
        color_pair_cache = {}
        next_pair_number = 1  # I numeri di coppia iniziano da 1

        def get_color_attr(color_index):
            """
            Restituisce l'attributo curses per un dato indice di colore xterm 256.
            Se non esiste, inizializza una nuova coppia colore. In caso il terminale non supporti
            256 colori, viene applicato un mapping semplice.

            Parametri:
                color_index (int): Indice del colore in xterm 256.

            Ritorna:
                int: Attributo curses per il colore.
            """
            nonlocal next_pair_number
            # Se il terminale supporta meno di 256 colori, mappa l'indice
            mapped_color = color_index
            if curses.COLORS < 256:
                mapped_color = color_index % curses.COLORS

            if color_index in color_pair_cache:
                return curses.color_pair(color_pair_cache[color_index])
            else:
                if next_pair_number >= curses.COLOR_PAIRS:
                    return curses.A_NORMAL
                try:
                    curses.init_pair(next_pair_number, mapped_color, -1)
                    color_pair_cache[color_index] = next_pair_number
                    pair_attr = curses.color_pair(next_pair_number)
                    next_pair_number += 1
                    return pair_attr
                except curses.error:
                    return curses.A_NORMAL

        while not stop_event.is_set():
            try:
                ascii_frame, conversion_time_ms = ascii_queue.get(timeout=0.005)
            except queue.Empty:
                # Controlla eventuale input per uscita
                if stdscr.getch() == ord('q'):
                    stop_event.set()
                continue

            if log_performance:
                rendering_start = time.perf_counter()

            # ascii_frame è una lista di liste di tuple (carattere, color_index)
            new_frame = ascii_frame

            curr_max_y, curr_max_x = stdscr.getmaxyx()
            if curr_max_y != max_y or curr_max_x != max_x:
                stdscr.clear()
                max_y, max_x = curr_max_y, curr_max_x
                prev_frame = [[('', -1)] * max_x for _ in range(max_y)]

            # Aggiorna per riga in maniera "aggregata"
            for i in range(min(len(new_frame), max_y)):
                new_row = new_frame[i]
                old_row = prev_frame[i] if i < len(prev_frame) else [('', -1)] * len(new_row)
                j = 0
                while j < min(len(new_row), max_x):
                    # Se la cella non è cambiata, salta
                    if new_row[j] == old_row[j]:
                        j += 1
                        continue
                    # Inizio di un segmento modificato
                    start = j
                    current_color = new_row[j][1]
                    current_attr = get_color_attr(current_color)
                    segment_chars = [new_row[j][0]]
                    j += 1
                    # Raggruppa i caratteri contigui che hanno lo stesso attributo e sono modificati
                    while j < min(len(new_row), max_x):
                        # Se la cella è invariata o il colore differisce, interrompi il segmento
                        if new_row[j] == old_row[j]:
                            break
                        next_attr = get_color_attr(new_row[j][1])
                        if next_attr != current_attr:
                            break
                        segment_chars.append(new_row[j][0])
                        j += 1
                    segment_str = "".join(segment_chars)
                    try:
                        stdscr.addstr(i, start, segment_str, current_attr)
                    except curses.error:
                        pass

            stdscr.refresh()

            # Aggiorna il frame precedente (copia profonda)
            prev_frame = [row[:] for row in new_frame]

            if log_performance:
                rendering_end = time.perf_counter()
                total_rendering_time_ms = (rendering_end - rendering_start) * 1000
                logging.info(
                    f"Frame {frame_counter} - Conversion: {conversion_time_ms:.2f} ms, "
                    f"Total Rendering: {total_rendering_time_ms:.2f} ms"
                )

            frame_counter += 1
            fps_count += 1

            now = time.perf_counter()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                if log_fps:
                    logging.info(f"[LOG] FPS display (curses color): {fps_count / elapsed:.1f}")
                fps_count = 0
                fps_start = now

            if stdscr.getch() == ord('q'):
                stop_event.set()

    curses.wrapper(curses_loop)


def render_frames_curses(ascii_queue, stop_event, log_fps=False, log_performance=False):
    """
    Renderizza i frame ASCII utilizzando la libreria curses con aggiornamento parziale
    a livello di caratteri. Aggiorna solamente i caratteri che risultano differenti
    rispetto al frame precedente, riducendo il carico di aggiornamento dello schermo
    e migliorando le performance.

    Parametri:
        ascii_queue (Queue): Coda contenente tuple (ascii_frame, conversion_time_ms) da renderizzare.
        stop_event (threading.Event): Evento usato per terminare il ciclo di rendering.
        log_fps (bool): Se True, logga il numero di aggiornamenti al secondo.
        log_performance (bool): Se True, logga il tempo impiegato per il rendering di ogni frame.
    """

    def curses_loop(stdscr):
        curses.curs_set(0)  # Nasconde il cursore
        stdscr.nodelay(True)  # Input non bloccante
        frame_counter = 0
        fps_count = 0
        fps_start = time.perf_counter()  # Usa un timer ad alta risoluzione
        prev_frame_lines = []  # Salva il frame precedente per il confronto
        max_y, max_x = stdscr.getmaxyx()

        # Pre-allocazione delle stringhe di spazi per la pulizia
        blank_spaces = {}  # Cache di stringhe di spazi

        while not stop_event.is_set():
            # Poll dell'evento di stop prima di attendere sulla coda
            if stop_event.is_set():
                break

            try:
                # Riduzione del timeout per una risposta più veloce ai segnali di stop
                ascii_frame, conversion_time_ms = ascii_queue.get(timeout=0.005)
            except queue.Empty:
                # Verifica dell'input senza dormire
                key = stdscr.getch()
                if key == ord('q'):
                    stop_event.set()
                continue

            if log_performance:
                rendering_start = time.perf_counter()  # Timer ad alta risoluzione

            frame_lines = ascii_frame.split("\n")
            curr_max_y, curr_max_x = stdscr.getmaxyx()
            # Aggiorna le dimensioni dello schermo solo se cambiate
            if curr_max_y != max_y or curr_max_x != max_x:
                max_y, max_x = curr_max_y, curr_max_x

            num_lines = min(max(len(frame_lines), len(prev_frame_lines)), max_y)

            # Aggiornamento parziale: riga per riga
            for i in range(num_lines):
                new_line = frame_lines[i] if i < len(frame_lines) else ""
                old_line = prev_frame_lines[i] if i < len(prev_frame_lines) else ""

                # Ottimizzazione: salta l'intera riga se identica
                if new_line == old_line:
                    continue

                max_width = max_x - 1  # Calcola una sola volta

                # Ottimizzazione: se le righe sono completamente diverse, riscrivi tutta la riga
                if len(new_line) <= max_width and abs(len(new_line) - len(old_line)) > len(new_line) // 2:
                    try:
                        stdscr.move(i, 0)
                        stdscr.clrtoeol()
                        if new_line:
                            stdscr.addstr(i, 0, new_line)
                        continue
                    except curses.error:
                        pass

                # Altrimenti, confronto carattere per carattere per aggiornamenti parziali
                min_len = min(len(new_line), len(old_line))

                # Trova la prima differenza per ottimizzare
                start_diff = 0
                while start_diff < min_len and new_line[start_diff] == old_line[start_diff]:
                    start_diff += 1

                # Trova l'ultima differenza per ottimizzare ulteriormente
                if start_diff < min_len:
                    end_diff = min_len - 1
                    while end_diff > start_diff and new_line[end_diff] == old_line[end_diff]:
                        end_diff -= 1

                    # Aggiorna solo i caratteri che cambiano nel mezzo
                    if start_diff <= end_diff and end_diff < max_width:
                        try:
                            segment = new_line[start_diff:end_diff + 1]
                            stdscr.addstr(i, start_diff, segment)
                        except curses.error:
                            pass

                # Se la nuova riga è più lunga, aggiorna il tratto in eccesso
                if len(new_line) > len(old_line):
                    try:
                        segment_start = max(len(old_line), start_diff)
                        if segment_start < len(new_line) and segment_start < max_width:
                            stdscr.addstr(i, segment_start, new_line[segment_start:min(max_width, len(new_line))])
                    except curses.error:
                        pass

                # Se la vecchia riga era più lunga, cancella i caratteri in eccesso
                elif len(old_line) > len(new_line):
                    try:
                        clear_length = min(max_width - len(new_line), len(old_line) - len(new_line))
                        if clear_length > 0:
                            # Usa cache per le stringhe di spazi
                            if clear_length not in blank_spaces:
                                blank_spaces[clear_length] = " " * clear_length
                            stdscr.addstr(i, len(new_line), blank_spaces[clear_length])
                    except curses.error:
                        pass

            # Aggiorna solo alla fine del disegno completo
            stdscr.refresh()

            # Usa copia profonda solo se necessario
            prev_frame_lines = frame_lines.copy() if frame_lines else []

            if log_performance:
                rendering_end = time.perf_counter()  # Timer ad alta risoluzione
                total_rendering_time_ms = (rendering_end - rendering_start) * 1000
                logging.info(
                    f"Frame {frame_counter} - Conversion: {conversion_time_ms:.2f} ms, "
                    f"Total Rendering: {total_rendering_time_ms:.2f} ms"
                )

            frame_counter += 1
            fps_count += 1

            now = time.perf_counter()  # Timer ad alta risoluzione
            elapsed = now - fps_start
            if elapsed >= 1.0:
                if log_fps:
                    logging.info(f"[LOG] FPS display (curses): {fps_count / elapsed:.1f}")
                fps_count = 0
                fps_start = now

            # Controllo input spostato alla fine del ciclo
            key = stdscr.getch()
            if key == ord('q'):
                stop_event.set()

    curses.wrapper(curses_loop)


def generate_calibration_frame(width, height):
    """
    Genera un frame ASCII di calibrazione tutto bianco, con un bordo e una croce centrale.

    Parametri:
        width (int): Larghezza dell'output ASCII.
        height (int): Altezza dell'output ASCII.

    Ritorna:
        str: Stringa ASCII con il frame bianco, bordi e croce centrale.
    """
    # Caratteri
    BORDER_CHAR = "#"
    CROSS_CHAR = "+"
    WHITE_CHAR = "█"  # Blocchi pieni per simulare un frame bianco

    # Crea una matrice di caratteri bianchi
    ascii_frame = [[WHITE_CHAR] * width for _ in range(height)]

    # Disegna i bordi del rettangolo
    for x in range(width):
        ascii_frame[0][x] = BORDER_CHAR  # Riga superiore
        ascii_frame[-1][x] = BORDER_CHAR  # Riga inferiore

    for y in range(height):
        ascii_frame[y][0] = BORDER_CHAR  # Colonna sinistra
        ascii_frame[y][-1] = BORDER_CHAR  # Colonna destra

    # Disegna la croce centrale
    center_y, center_x = height // 2, width // 2
    for x in range(width):
        ascii_frame[center_y][x] = CROSS_CHAR  # Linea orizzontale
    for y in range(height):
        ascii_frame[y][center_x] = CROSS_CHAR  # Linea verticale

    # Converti la matrice in una stringa
    ascii_string = "\n".join("".join(row) for row in ascii_frame)
    return ascii_string


def render_calibration_frame(width, height):
    """
    Mostra un frame di calibrazione nel terminale e attende che l'utente prema ENTER.
    Dopo l'input, il terminale viene svuotato e il buffer di scorrimento viene rimosso.

    Parametri:
        width (int): Larghezza dell'output ASCII.
        height (int): Altezza dell'output ASCII.
    """
    # Genera il frame di calibrazione
    calibration_frame = generate_calibration_frame(width, height)

    # Escape sequences per il terminale
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    CLEAR_SCREEN = "\033[2J\033[H"
    RESET_TERMINAL = "\033c"  # Reset completo del terminale

    # Pulisce lo schermo e nasconde il cursore
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write(calibration_frame + "\n")
    sys.stdout.write("\n[INFO] Regola la dimensione del terminale e premi ENTER per iniziare...\n")
    sys.stdout.flush()

    # Attendi l'input dell'utente
    input()

    # Resetta completamente il terminale (rimuove lo scrollback buffer)
    sys.stdout.write(RESET_TERMINAL)
    sys.stdout.flush()

    # Ripristina il cursore
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.flush()


def main():
    """
    Funzione principale che configura il parsing degli argomenti, crea le code e i thread/processi
    necessari alla pipeline (estrazione, conversione e rendering) e avvia il rendering in tempo reale.
    In base alle flag --use_curses e --curses_color, il rendering avviene tramite curses in modalità colorata o monocromatica,
    oppure con il metodo standard (sys).
    """
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(
        description="Real-time ASCII video using a parallel pipeline with separate conversion and rendering."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for extraction (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of display FPS")
    parser.add_argument("--log_performance", action="store_true", help="Enable logging of conversion and rendering performance")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing frames (default: 1)")
    parser.add_argument("--use_threads", action="store_true", help="Use thread pool instead of multiprocessing pool (utile su Windows)")
    parser.add_argument("--use_curses", action="store_true", help="Use curses library for rendering instead of sys-based partial rendering")
    parser.add_argument("--curses_color", action="store_true", help="In curses mode, use colored rendering")
    args = parser.parse_args()

    # Apertura video per ottenere le dimensioni iniziali
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Errore: impossibile leggere il primo frame.")
        return

    height, width_frame = frame.shape[:2]
    aspect_ratio = height / width_frame
    new_height = int(aspect_ratio * args.width * 0.45)  # Altezza adattata

    render_calibration_frame(args.width, new_height)

    raw_queue = Queue(maxsize=args.fps * 3)
    ascii_queue = Queue(maxsize=args.fps * 3)

    extractor_process = Process(target=extract_frames, args=(args.video_path, raw_queue, args.fps))
    extractor_process.start()

    if args.use_threads:
        pool = ThreadPool(processes=cpu_count())
    else:
        pool = Pool(processes=cpu_count())

    stop_event = threading.Event()

    # Se viene attivata la modalità curses, scegli il rendering a colori o monocromatico
    if args.use_curses:
        if args.curses_color:
            conversion_function = frame_to_ascii_curses_color
            renderer_function = render_frames_curses_color
        else:
            conversion_function = frame_to_ascii_curses
            renderer_function = render_frames_curses
    else:
        conversion_function = frame_to_ascii
        renderer_function = render_frames_sys_partial

    converter_thread = threading.Thread(
        target=convert_frames,
        args=(raw_queue, ascii_queue, pool, args.batch_size, args.width, stop_event, conversion_function),
        daemon=True
    )
    converter_thread.start()

    renderer_thread = threading.Thread(
        target=renderer_function,
        args=(ascii_queue, stop_event, args.log_fps, args.log_performance),
        daemon=True
    )
    renderer_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    stop_event.set()
    renderer_thread.join(timeout=1)
    converter_thread.join(timeout=1)
    pool.close()
    pool.join()
    extractor_process.terminate()
    extractor_process.join()
    raw_queue.close()
    raw_queue.cancel_join_thread()
    ascii_queue.close()
    ascii_queue.cancel_join_thread()
    print("[✔] Terminazione completata.")

if __name__ == '__main__':
    main()
