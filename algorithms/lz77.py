"""
LZ77 Compression and Decompression Algorithm (Pure Python).
Uses a sliding window to find repeated byte sequences and replaces them with (offset, length, next_byte) tokens.
Written in simple A2/B1 English.
"""

import struct
from typing import List, Tuple


def lz77_compress(data: bytes, window_size: int = 4096, lookahead_size: int = 32) -> List[Tuple[int, int, int]]:
    """
    Compress byte data using the LZ77 sliding window algorithm.
    
    Args:
        data: Input bytes to compress.
        window_size: Max distance to look back in history (default 4096 bytes).
        lookahead_size: Max length of a match (default 32 bytes).
    Returns:
        list of tuples: Each token is (offset, length, next_byte).
            - offset: How many bytes to jump back.
            - length: How many bytes to copy from that position.
            - next_byte: The next literal byte after the match.
    """
    i = 0
    n = len(data)
    tokens: List[Tuple[int, int, int]] = []
    
    # Hash table storing past 3-byte positions for fast search
    pos_table = {}

    while i < n:
        best_offset = 0
        best_length = 0

        window_start = max(0, i - window_size)
        prefix = data[i : i + 3]

        # Check if the next 3 bytes were seen in the sliding window
        if len(prefix) == 3 and prefix in pos_table:
            for cand_pos in reversed(pos_table[prefix]):
                if cand_pos < window_start:
                    break
                
                # Count matching bytes
                match_len = 0
                max_len = min(lookahead_size, n - i - 1)
                while match_len < max_len and data[cand_pos + match_len] == data[i + match_len]:
                    match_len += 1

                if match_len > best_length:
                    best_length = match_len
                    best_offset = i - cand_pos
                    if match_len >= lookahead_size:
                        break

        if best_length >= 3:
            next_byte = data[i + best_length]
            tokens.append((best_offset, best_length, next_byte))

            # Add new positions to the hash table
            for step in range(best_length + 1):
                if i + step + 3 <= n:
                    p = data[i + step : i + step + 3]
                    if p not in pos_table:
                        pos_table[p] = []
                    pos_table[p].append(i + step)

            i += best_length + 1
        else:
            # Literal byte (no match found)
            tokens.append((0, 0, data[i]))
            if i + 3 <= n:
                if prefix not in pos_table:
                    pos_table[prefix] = []
                pos_table[prefix].append(i)
            i += 1

    return tokens


def lz77_decompress(tokens: List[Tuple[int, int, int]]) -> bytes:
    """
    Decompress a list of LZ77 tokens back into the original bytes.
    
    Args:
        tokens: List of (offset, length, next_byte) tuples.
    Returns:
        bytes: Decompressed original data.
    """
    output = bytearray()

    for offset, length, next_byte in tokens:
        if length > 0:
            start = len(output) - offset
            for k in range(length):
                output.append(output[start + k])
        output.append(next_byte)

    return bytes(output)


def lz77_encode(data: bytes, window_size: int = 4096, lookahead_size: int = 32) -> bytes:
    """
    Convenience function: Run LZ77 compression and serialize tokens into a compact binary format.
    Each token is packed as:
        - offset: 2 bytes (uint16)
        - length: 1 byte (uint8)
        - next_byte: 1 byte (uint8)
    """
    tokens = lz77_compress(data, window_size=window_size, lookahead_size=lookahead_size)
    stream = bytearray()
    
    # Store token count (4 bytes uint32)
    stream.extend(struct.pack(">I", len(tokens)))
    
    # Store each token (4 bytes per token)
    for offset, length, next_byte in tokens:
        stream.extend(struct.pack(">HBB", offset, length, next_byte))
        
    return bytes(stream)


def lz77_decode(compressed_data: bytes) -> bytes:
    """
    Convenience function: Deserialize binary stream into tokens and decompress.
    """
    if len(compressed_data) < 4:
        raise ValueError("Invalid LZ77 stream: Too short.")
        
    token_count = struct.unpack(">I", compressed_data[:4])[0]
    expected_len = 4 + token_count * 4
    if len(compressed_data) < expected_len:
        raise ValueError("Invalid LZ77 stream: Incomplete token data.")

    tokens = []
    offset = 4
    for _ in range(token_count):
        off, length, next_b = struct.unpack(">HBB", compressed_data[offset : offset + 4])
        tokens.append((off, length, next_b))
        offset += 4

    return lz77_decompress(tokens)
