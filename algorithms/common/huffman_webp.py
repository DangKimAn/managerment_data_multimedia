"""
algorithms/common/huffman_webp.py
Canonical Huffman Coding module tailored for WebP Lossless (VP8L).

Key WebP Lossless characteristics implemented here:
1. Canonical Huffman: Codes are assigned based on code lengths (RFC 1951 / VP8L).
   Only code lengths are stored, not full frequency trees, saving header space.
2. LSB-First Bit Ordering: In WebP Lossless (VP8L), bits are packed and read
   in Least-Significant-Bit first order (bit 0 to bit 7).
3. Fast Prefix-Tree Decoding: Reconstructs exact Canonical codes deterministically.

Can be used standalone or as the entropy coding stage for WebP Lossless.
Written in simple A2/B1 English.
"""

import heapq
import struct
from collections import Counter
from typing import Dict, List, Optional, Tuple


MAGIC_WPHU = b"WPHU"  # WebP Huffman stream identifier
MAX_CODE_LENGTH = 15  # VP8L maximum Huffman code length is 15 bits


# ==============================================================================
# LSB-FIRST BITSTREAM WRITER AND READER (WebP VP8L Convention)
# ==============================================================================

class LsbBitWriter:
    """
    Bitstream writer packing bits in LSB-first order (WebP VP8L standard).
    Bit 0 of each value occupies the least-significant available bit in the byte.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.bit_buf = 0
        self.bit_count = 0

    def write_bit(self, bit: int) -> None:
        """Write a single bit (0 or 1)."""
        self.bit_buf |= (bit & 1) << self.bit_count
        self.bit_count += 1
        if self.bit_count == 8:
            self.buffer.append(self.bit_buf)
            self.bit_buf = 0
            self.bit_count = 0

    def write_bits(self, value: int, num_bits: int) -> None:
        """
        Write `num_bits` from `value` to stream in LSB-first order.
        """
        self.bit_buf |= (value & ((1 << num_bits) - 1)) << self.bit_count
        self.bit_count += num_bits
        while self.bit_count >= 8:
            self.buffer.append(self.bit_buf & 0xFF)
            self.bit_buf >>= 8
            self.bit_count -= 8

    def flush(self) -> bytes:
        """Flush remaining buffered bits (padded with 0s) and return bytes."""
        if self.bit_count > 0:
            self.buffer.append(self.bit_buf & 0xFF)
            self.bit_buf = 0
            self.bit_count = 0
        return bytes(self.buffer)


class LsbBitReader:
    """
    Bitstream reader unpacking bits in LSB-first order (WebP VP8L standard).
    """

    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_buf = 0
        self.bit_count = 0

    def read_bit(self) -> int:
        """Read a single bit (0 or 1). Returns 0 on EOF."""
        while self.bit_count < 1:
            if self.byte_pos < len(self.data):
                self.bit_buf |= self.data[self.byte_pos] << self.bit_count
                self.byte_pos += 1
                self.bit_count += 8
            else:
                return 0
        bit = self.bit_buf & 1
        self.bit_buf >>= 1
        self.bit_count -= 1
        return bit

    def read_bits(self, num_bits: int) -> int:
        """Read `num_bits` from stream in LSB-first order."""
        while self.bit_count < num_bits:
            if self.byte_pos < len(self.data):
                self.bit_buf |= self.data[self.byte_pos] << self.bit_count
                self.byte_pos += 1
                self.bit_count += 8
            else:
                break
        val = self.bit_buf & ((1 << num_bits) - 1)
        self.bit_buf >>= num_bits
        self.bit_count = max(0, self.bit_count - num_bits)
        return val


# ==============================================================================
# CANONICAL HUFFMAN CODE GENERATION
# ==============================================================================

class _HuffNode:
    """Helper node for building initial Huffman tree to measure code lengths."""
    def __init__(self, freq: int, symbol: Optional[int] = None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other: "_HuffNode") -> bool:
        return self.freq < other.freq


def compute_code_lengths(frequencies: Dict[int, int], max_bits: int = MAX_CODE_LENGTH) -> Dict[int, int]:
    """
    Compute optimal Huffman code lengths (bit depths) for given symbol frequencies.
    Enforces maximum code length of `max_bits` (default 15 bits for WebP VP8L).

    Args:
        frequencies: Mapping from symbol (0..255) to occurrence count.
        max_bits: Maximum code length allowed (<= 15).
    Returns:
        dict: Mapping from symbol to code length in bits.
    """
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        sym = next(iter(frequencies.keys()))
        return {sym: 1}

    # Build standard Huffman tree with priority queue
    heap: List[_HuffNode] = []
    for sym, freq in frequencies.items():
        heapq.heappush(heap, _HuffNode(freq, symbol=sym))

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        parent = _HuffNode(n1.freq + n2.freq, left=n1, right=n2)
        heapq.heappush(heap, parent)

    root = heap[0]

    # Measure depth of each leaf
    lengths: Dict[int, int] = {}

    def _traverse(node: _HuffNode, depth: int):
        if node.symbol is not None:
            lengths[node.symbol] = max(1, depth)
            return
        if node.left:
            _traverse(node.left, depth + 1)
        if node.right:
            _traverse(node.right, depth + 1)

    _traverse(root, 0)

    # Cap code lengths at max_bits if necessary
    for sym in lengths:
        if lengths[sym] > max_bits:
            lengths[sym] = max_bits

    return lengths


def build_canonical_codes(
    code_lengths: Dict[int, int],
    max_bits: int = MAX_CODE_LENGTH
) -> Dict[int, Tuple[int, int]]:
    """
    Generate Canonical Huffman codes from symbol code lengths.
    Follows RFC 1951 / VP8L standard canonical code assignment:
    1. Sort symbols by length, then by symbol value.
    2. Codes of length L are consecutive integers.
    3. First code of length L = (first code of length L-1 + count(L-1)) << 1.

    Args:
        code_lengths: Mapping from symbol to bit length.
        max_bits: Maximum bit length.
    Returns:
        dict: Mapping from symbol to (canonical_code_int, bit_length).
    """
    if not code_lengths:
        return {}

    # 1. Count number of symbols for each code length
    bl_count = [0] * (max_bits + 1)
    for sym, length in code_lengths.items():
        if 0 < length <= max_bits:
            bl_count[length] += 1

    # 2. Find starting numerical code for each length
    code = 0
    bl_count[0] = 0
    next_code = [0] * (max_bits + 1)
    for bits in range(1, max_bits + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    # 3. Assign codes to symbols in numerical order
    canonical_codes: Dict[int, Tuple[int, int]] = {}
    for sym in sorted(code_lengths.keys()):
        length = code_lengths[sym]
        if 0 < length <= max_bits:
            canonical_codes[sym] = (next_code[length], length)
            next_code[length] += 1

    return canonical_codes


# ==============================================================================
# HIGH-LEVEL ENCODE / DECODE API
# ==============================================================================

def huffman_webp_encode(data: bytes) -> bytes:
    """
    Encode raw byte data using Canonical Huffman with LSB-first bitstream.

    Stream Layout:
        - Magic: b"WPHU" (4 bytes)
        - Original uncompressed data length: uint32 (4 bytes)
        - Symbol count N: uint16 (2 bytes)
        - For each of N symbols:
            - Symbol (uint8, 1 byte)
            - Code length (uint8, 1 byte)
        - Packed bitstream (LSB-first)

    Args:
        data: Raw uncompressed bytes.
    Returns:
        bytes: Encoded payload.
    """
    if not data:
        return MAGIC_WPHU + struct.pack("<IH", 0, 0)

    # 1. Calculate symbol frequencies and compute canonical code lengths
    frequencies = dict(Counter(data))
    code_lengths = compute_code_lengths(frequencies)
    canonical_codes = build_canonical_codes(code_lengths)

    # 2. Pack header
    header = bytearray(MAGIC_WPHU)
    orig_len = len(data)
    num_symbols = len(code_lengths)
    header.extend(struct.pack("<IH", orig_len, num_symbols))

    # Store (symbol, length) pairs
    for sym in sorted(code_lengths.keys()):
        header.append(sym)
        header.append(code_lengths[sym])

    # 3. Encode data bits using LSB bit writer
    writer = LsbBitWriter()
    for b in data:
        code_int, length = canonical_codes[b]
        # In VP8L canonical code, bits are written sequentially
        # from MSB of canonical code to LSB, or LSB to MSB.
        # Writing MSB-to-LSB into LsbBitWriter ensures prefix matching during read_bit.
        for i in range(length - 1, -1, -1):
            bit = (code_int >> i) & 1
            writer.write_bit(bit)

    bitstream = writer.flush()
    return bytes(header) + bitstream


def huffman_webp_decode(encoded_data: bytes) -> bytes:
    """
    Decode payload encoded by `huffman_webp_encode`.

    Args:
        encoded_data: Encoded byte stream.
    Returns:
        bytes: Reconstructed uncompressed bytes.
    Raises:
        ValueError: If header or data is corrupted.
    """
    if len(encoded_data) < 10:
        raise ValueError("Encoded data too short.")

    # 1. Validate magic
    magic = encoded_data[:4]
    if magic != MAGIC_WPHU:
        raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC_WPHU!r}")

    # 2. Read original length and symbol count
    orig_len, num_symbols = struct.unpack("<IH", encoded_data[4:10])
    if orig_len == 0 or num_symbols == 0:
        return b""

    # 3. Read code lengths
    offset = 10
    code_lengths: Dict[int, int] = {}
    for _ in range(num_symbols):
        sym = encoded_data[offset]
        length = encoded_data[offset + 1]
        code_lengths[sym] = length
        offset += 2

    canonical_codes = build_canonical_codes(code_lengths)

    # 4. Build fast decoding prefix tree
    # Each node is [child0, child1, symbol_if_leaf]
    tree_root: dict = {}
    for sym, (code_int, length) in canonical_codes.items():
        curr = tree_root
        for i in range(length - 1, -1, -1):
            bit = (code_int >> i) & 1
            if bit not in curr:
                curr[bit] = {}
            curr = curr[bit]
        curr["val"] = sym

    # 5. Read bitstream and decode symbols
    bitstream_data = encoded_data[offset:]
    reader = LsbBitReader(bitstream_data)

    out = bytearray(orig_len)
    curr_node = tree_root
    decoded_count = 0

    while decoded_count < orig_len:
        bit = reader.read_bit()
        if bit not in curr_node:
            raise ValueError(f"Corrupted bitstream at symbol {decoded_count}/{orig_len}")
        curr_node = curr_node[bit]
        if "val" in curr_node:
            out[decoded_count] = curr_node["val"]
            decoded_count += 1
            curr_node = tree_root

    return bytes(out)
