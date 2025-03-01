# cython: boundscheck=False, wraparound=False, language_level=3

import numpy as np
cimport numpy as np
from PIL import Image
import cv2
import cython

# Costante contenente i caratteri ASCII
ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"
cdef int num_ascii = len(ASCII_CHARS)

def convert_pixels_to_ascii(np.ndarray[np.uint8_t, ndim=3] pixels):
    """
    Converte un array NumPy di dimensione (h, w, 3) (in RGB) in una stringa ASCII
    con codici ANSI per il colore.
    """
    cdef int h = <int>pixels.shape[0]
    cdef int w = <int>pixels.shape[1]
    cdef int i, j, index
    cdef double brightness
    cdef unsigned char r, g, b
    cdef list out_lines = []
    cdef list row_chars
    for i in range(h):
        row_chars = []
        for j in range(w):
            r = pixels[i, j, 0]
            g = pixels[i, j, 1]
            b = pixels[i, j, 2]
            brightness = (r + g + b) / 3.0
            index = <int>(brightness * (num_ascii - 1) / 255.0)
            if index < 0:
                index = 0
            elif index >= num_ascii:
                index = num_ascii - 1
            # Crea la stringa con codici ANSI a 24-bit per il colore
            row_chars.append(f"\033[38;2;{r};{g};{b}m{ASCII_CHARS[index]}\033[0m")
        out_lines.append("".join(row_chars))
    return "\n".join(out_lines)

def frame_to_ascii_cython(object frame, int new_width):
    """
    Converte un frame (immagine BGR da OpenCV) in ASCII art colorato.
    1. Converte il frame da BGR a RGB
    2. Ridimensiona l'immagine in base a new_width mantenendo l'aspect ratio
    3. Converte l'immagine in un array NumPy e chiama convert_pixels_to_ascii
    """
    cdef object image
    cdef int width, height
    cdef float aspect_ratio
    cdef int new_height
    # Converti da BGR a RGB e crea un oggetto PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)
    image = image.resize((new_width, new_height))
    # Converte l'immagine in un array NumPy
    cdef np.ndarray[np.uint8_t, ndim=3] pixels = np.array(image)
    return convert_pixels_to_ascii(pixels)
