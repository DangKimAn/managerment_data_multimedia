"""
Spatial Prediction Filters for image compression (PNG standard).
Vectorized using NumPy for high-performance processing.
Written in simple A2/B1 English.
"""

from enum import IntEnum
import numpy as np


class FilterType(IntEnum):
    """Available filter types for scanlines"""
    NONE = 0      # No filter (Direct data)
    SUB = 1       # Difference from left pixel (Pixel - Left)
    UP = 2        # Difference from above pixel (Pixel - Above)
    AVERAGE = 3   # Difference from average: Pixel - (Left + Above) // 2
    PAETH = 4     # Paeth linear predictor (PNG standard)
    ADAPTIVE = 5  # Pick the best filter for each scanline automatically


def paeth_predictor(a: int, b: int, c: int) -> int:
    """Standard scalar Paeth predictor from 3 neighbor pixels"""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    else:
        return c


def paeth_predictor_vectorized(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorized Paeth predictor using NumPy matrices"""
    a_i = a.astype(np.int32)
    b_i = b.astype(np.int32)
    c_i = c.astype(np.int32)
    p = a_i + b_i - c_i
    pa = np.abs(p - a_i)
    pb = np.abs(p - b_i)
    pc = np.abs(p - c_i)
    pred = np.where((pa <= pb) & (pa <= pc), a_i, np.where(pb <= pc, b_i, c_i))
    return pred.astype(np.uint8)


def filter_scanlines(raw_bytes: bytes, width: int, height: int, bpp: int,
                     filter_mode: FilterType) -> bytes:
    """
    Apply spatial prediction filters to each row of pixels (scanline) using NumPy.
    Each scanline starts with 1 byte indicating which filter was used.
    """
    stride = width * bpp
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, stride))

    left = np.zeros_like(arr)
    left[:, bpp:] = arr[:, :-bpp]

    above = np.zeros_like(arr)
    above[1:, :] = arr[:-1, :]

    up_left = np.zeros_like(arr)
    up_left[1:, bpp:] = arr[:-1, :-bpp]

    output = bytearray()

    if filter_mode == FilterType.NONE:
        f_all = arr
    elif filter_mode == FilterType.SUB:
        f_all = ((arr.astype(np.int16) - left.astype(np.int16)) & 0xFF).astype(np.uint8)
    elif filter_mode == FilterType.UP:
        f_all = ((arr.astype(np.int16) - above.astype(np.int16)) & 0xFF).astype(np.uint8)
    elif filter_mode == FilterType.AVERAGE:
        avg_pred = ((left.astype(np.uint16) + above.astype(np.uint16)) // 2).astype(np.uint8)
        f_all = ((arr.astype(np.int16) - avg_pred.astype(np.int16)) & 0xFF).astype(np.uint8)
    elif filter_mode == FilterType.PAETH:
        paeth_pred = paeth_predictor_vectorized(left, above, up_left)
        f_all = ((arr.astype(np.int16) - paeth_pred.astype(np.int16)) & 0xFF).astype(np.uint8)
    elif filter_mode == FilterType.ADAPTIVE:
        f0 = arr
        f1 = ((arr.astype(np.int16) - left.astype(np.int16)) & 0xFF).astype(np.uint8)
        f2 = ((arr.astype(np.int16) - above.astype(np.int16)) & 0xFF).astype(np.uint8)
        avg_pred = ((left.astype(np.uint16) + above.astype(np.uint16)) // 2).astype(np.uint8)
        f3 = ((arr.astype(np.int16) - avg_pred.astype(np.int16)) & 0xFF).astype(np.uint8)
        paeth_pred = paeth_predictor_vectorized(left, above, up_left)
        f4 = ((arr.astype(np.int16) - paeth_pred.astype(np.int16)) & 0xFF).astype(np.uint8)

        all_filters = [f0, f1, f2, f3, f4]

        for y in range(height):
            best_t = 0
            best_score = float('inf')
            best_line = None
            for t_idx, f_mat in enumerate(all_filters):
                line = f_mat[y]
                line_u16 = line.astype(np.uint16)
                score = np.minimum(line_u16, 256 - line_u16).sum()
                if score < best_score:
                    best_score = score
                    best_t = t_idx
                    best_line = line
            output.append(best_t)
            output.extend(best_line.tobytes())
        return bytes(output)
    else:
        raise ValueError(f"Unsupported filter_mode: {filter_mode}")

    for y in range(height):
        output.append(int(filter_mode))
        output.extend(f_all[y].tobytes())

    return bytes(output)


def unfilter_scanlines(filtered_bytes: bytes, width: int, height: int, bpp: int) -> bytes:
    """Restore (unfilter) the original pixel bytes from filtered scanlines."""
    stride = width * bpp
    line_len = stride + 1
    output = bytearray(height * stride)
    prev_line = np.zeros(stride, dtype=np.uint8)

    for y in range(height):
        offset = y * line_len
        ftype = FilterType(filtered_bytes[offset])
        scanline = np.frombuffer(filtered_bytes[offset + 1 : offset + line_len], dtype=np.uint8)
        curr_unfiltered = np.zeros(stride, dtype=np.uint8)

        if ftype == FilterType.NONE:
            curr_unfiltered = scanline

        elif ftype == FilterType.SUB:
            scan_2d = scanline.reshape((-1, bpp)).astype(np.uint16)
            unfilt_2d = np.cumsum(scan_2d, axis=0) & 0xFF
            curr_unfiltered = unfilt_2d.astype(np.uint8).ravel()

        elif ftype == FilterType.UP:
            curr_unfiltered = (scanline.astype(np.uint16) + prev_line.astype(np.uint16)).astype(np.uint8)

        elif ftype == FilterType.AVERAGE:
            for i in range(stride):
                left = int(curr_unfiltered[i - bpp]) if i >= bpp else 0
                above = int(prev_line[i])
                pred = (left + above) // 2
                curr_unfiltered[i] = (int(scanline[i]) + pred) & 0xFF

        elif ftype == FilterType.PAETH:
            for i in range(stride):
                left = int(curr_unfiltered[i - bpp]) if i >= bpp else 0
                above = int(prev_line[i])
                up_left = int(prev_line[i - bpp]) if i >= bpp else 0
                pred = paeth_predictor(left, above, up_left)
                curr_unfiltered[i] = (int(scanline[i]) + pred) & 0xFF
        else:
            raise ValueError(f"Invalid filter type in stream: {ftype}")

        output[y * stride : (y + 1) * stride] = curr_unfiltered.tobytes()
        prev_line = curr_unfiltered

    return bytes(output)
