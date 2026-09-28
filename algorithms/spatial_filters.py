"""
Spatial Prediction Filters for image compression (PNG standard).
Reduces difference between neighbor pixels so DEFLATE can compress more easily.
Written in simple A2/B1 English.
"""

from enum import IntEnum


class FilterType(IntEnum):
    """Available filter types for scanlines"""
    NONE = 0      # No filter (Direct data)
    SUB = 1       # Difference from left pixel (Pixel - Left)
    UP = 2        # Difference from above pixel (Pixel - Above)
    AVERAGE = 3   # Difference from average: Pixel - (Left + Above) // 2
    PAETH = 4     # Paeth linear predictor (PNG standard)
    ADAPTIVE = 5  # Pick the best filter for each scanline automatically


def paeth_predictor(a: int, b: int, c: int) -> int:
    """
    Standard Paeth predictor from 3 neighbor pixels:
    - a: left pixel
    - b: above pixel
    - c: upper-left pixel
    """
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


def filter_scanlines(raw_bytes: bytes, width: int, height: int, bpp: int,
                     filter_mode: FilterType) -> bytes:
    """
    Apply spatial prediction filters to each row of pixels (scanline).
    Each scanline starts with 1 byte that indicates which filter was used.
    
    Args:
        raw_bytes: 1D byte array of all image pixels.
        width: Image width.
        height: Image height.
        bpp: Bytes per pixel (e.g. 3 for RGB, 1 for Grayscale).
        filter_mode: The FilterType to use.
    Returns:
        bytes: Filtered byte stream ready for entropy/dictionary compression.
    """
    stride = width * bpp
    output = bytearray()
    prev_line = bytearray(stride)

    for y in range(height):
        curr_line = raw_bytes[y * stride : (y + 1) * stride]

        if filter_mode == FilterType.NONE:
            chosen_type = FilterType.NONE
            filtered_line = bytearray(curr_line)

        elif filter_mode == FilterType.SUB:
            chosen_type = FilterType.SUB
            filtered_line = bytearray(stride)
            filtered_line[:bpp] = curr_line[:bpp]
            for i in range(bpp, stride):
                filtered_line[i] = (curr_line[i] - curr_line[i - bpp]) & 0xFF

        elif filter_mode == FilterType.UP:
            chosen_type = FilterType.UP
            filtered_line = bytearray(stride)
            for i in range(stride):
                filtered_line[i] = (curr_line[i] - prev_line[i]) & 0xFF

        elif filter_mode == FilterType.AVERAGE:
            chosen_type = FilterType.AVERAGE
            filtered_line = bytearray(stride)
            for i in range(stride):
                left = curr_line[i - bpp] if i >= bpp else 0
                above = prev_line[i]
                pred = (left + above) // 2
                filtered_line[i] = (curr_line[i] - pred) & 0xFF

        elif filter_mode == FilterType.PAETH:
            chosen_type = FilterType.PAETH
            filtered_line = bytearray(stride)
            for i in range(stride):
                left = curr_line[i - bpp] if i >= bpp else 0
                above = prev_line[i]
                up_left = prev_line[i - bpp] if i >= bpp else 0
                pred = paeth_predictor(left, above, up_left)
                filtered_line[i] = (curr_line[i] - pred) & 0xFF

        elif filter_mode == FilterType.ADAPTIVE:
            # Test all 5 filters and select the one with the smallest sum of differences
            # Filter 0: None
            f0 = curr_line
            s0 = sum((256 - b) if b > 128 else b for b in f0)

            # Filter 1: Sub
            f1 = bytearray(stride)
            f1[:bpp] = curr_line[:bpp]
            for i in range(bpp, stride):
                f1[i] = (curr_line[i] - curr_line[i - bpp]) & 0xFF
            s1 = sum((256 - b) if b > 128 else b for b in f1)

            # Filter 2: Up
            f2 = bytearray(stride)
            for i in range(stride):
                f2[i] = (curr_line[i] - prev_line[i]) & 0xFF
            s2 = sum((256 - b) if b > 128 else b for b in f2)

            # Filter 3: Average
            f3 = bytearray(stride)
            for i in range(stride):
                left = curr_line[i - bpp] if i >= bpp else 0
                above = prev_line[i]
                f3[i] = (curr_line[i] - ((left + above) // 2)) & 0xFF
            s3 = sum((256 - b) if b > 128 else b for b in f3)

            # Filter 4: Paeth
            f4 = bytearray(stride)
            for i in range(stride):
                left = curr_line[i - bpp] if i >= bpp else 0
                above = prev_line[i]
                up_left = prev_line[i - bpp] if i >= bpp else 0
                pred = paeth_predictor(left, above, up_left)
                f4[i] = (curr_line[i] - pred) & 0xFF
            s4 = sum((256 - b) if b > 128 else b for b in f4)

            # Sort by error score and pick the lowest
            candidates = [
                (s0, FilterType.NONE, f0),
                (s1, FilterType.SUB, f1),
                (s2, FilterType.UP, f2),
                (s3, FilterType.AVERAGE, f3),
                (s4, FilterType.PAETH, f4)
            ]
            candidates.sort(key=lambda item: item[0])
            chosen_type = candidates[0][1]
            filtered_line = candidates[0][2]
        else:
            raise ValueError(f"Unsupported filter_mode: {filter_mode}")

        output.append(int(chosen_type))
        output.extend(filtered_line)
        prev_line = bytearray(curr_line)

    return bytes(output)


def unfilter_scanlines(filtered_bytes: bytes, width: int, height: int, bpp: int) -> bytes:
    """
    Restore (unfilter) the original pixel bytes from filtered scanlines.
    
    Args:
        filtered_bytes: Byte stream containing filter headers and filtered pixels.
        width: Image width.
        height: Image height.
        bpp: Bytes per pixel.
    Returns:
        bytes: 1D byte array of original pixels.
    """
    stride = width * bpp
    line_len = stride + 1
    output = bytearray(height * stride)
    prev_line = bytearray(stride)

    for y in range(height):
        offset = y * line_len
        ftype = FilterType(filtered_bytes[offset])
        scanline = bytearray(filtered_bytes[offset + 1 : offset + line_len])
        curr_unfiltered = bytearray(stride)

        if ftype == FilterType.NONE:
            curr_unfiltered = scanline

        elif ftype == FilterType.SUB:
            for i in range(bpp):
                curr_unfiltered[i] = scanline[i]
            for i in range(bpp, stride):
                curr_unfiltered[i] = (scanline[i] + curr_unfiltered[i - bpp]) & 0xFF

        elif ftype == FilterType.UP:
            for i in range(stride):
                curr_unfiltered[i] = (scanline[i] + prev_line[i]) & 0xFF

        elif ftype == FilterType.AVERAGE:
            for i in range(stride):
                left = curr_unfiltered[i - bpp] if i >= bpp else 0
                above = prev_line[i]
                pred = (left + above) // 2
                curr_unfiltered[i] = (scanline[i] + pred) & 0xFF

        elif ftype == FilterType.PAETH:
            for i in range(stride):
                left = curr_unfiltered[i - bpp] if i >= bpp else 0
                above = prev_line[i]
                up_left = prev_line[i - bpp] if i >= bpp else 0
                pred = paeth_predictor(left, above, up_left)
                curr_unfiltered[i] = (scanline[i] + pred) & 0xFF
        else:
            raise ValueError(f"Invalid filter type in stream: {ftype}")

        output[y * stride : (y + 1) * stride] = curr_unfiltered
        prev_line = curr_unfiltered

    return bytes(output)
