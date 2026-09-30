"""
Huffman Coding Algorithm (Pure Python).
An entropy coding technique that gives shorter binary codes to bytes that appear more frequently.
Written in simple A2/B1 English.
"""

import heapq
import struct
from collections import Counter
from typing import Dict, Optional, Tuple


class HuffmanNode:
    """A node inside the Huffman binary tree"""
    def __init__(self, byte_val: Optional[int], freq: int):
        self.byte_val = byte_val  # Byte value (0-255) or None for internal nodes
        self.freq = freq          # How many times this byte appears
        self.left: Optional[HuffmanNode] = None
        self.right: Optional[HuffmanNode] = None

    def __lt__(self, other: "HuffmanNode") -> bool:
        # Comparison used by min-heap priority queue
        return self.freq < other.freq


def build_huffman_tree(data: bytes) -> Optional[HuffmanNode]:
    """Build a Huffman binary tree from byte frequencies"""
    if not data:
        return None

    # Count frequency of each byte
    freq_map = Counter(data)

    # Initialize min-heap with leaf nodes
    heap = [HuffmanNode(byte_val=b, freq=f) for b, f in freq_map.items()]
    heapq.heapify(heap)

    # If all bytes are identical, create a parent node
    if len(heap) == 1:
        single = heapq.heappop(heap)
        parent = HuffmanNode(byte_val=None, freq=single.freq)
        parent.left = single
        return parent

    # Combine two smallest nodes until only 1 root node remains
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)

        merged = HuffmanNode(byte_val=None, freq=node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]


def build_codes(root: Optional[HuffmanNode]) -> Dict[int, str]:
    """Traverse Huffman tree and generate prefix bit codes (e.g. 97: '010')"""
    codes = {}

    def _traverse(node: Optional[HuffmanNode], current_code: str):
        if node is None:
            return
        if node.byte_val is not None:
            # Leaf node reached: assign bit string
            codes[node.byte_val] = current_code if current_code else "0"
            return
        _traverse(node.left, current_code + "0")
        _traverse(node.right, current_code + "1")

    _traverse(root, "")
    return codes


def huffman_encode(data: bytes) -> bytes:
    """
    Encode input data into a self-contained binary Huffman bitstream.
    
    Format:
        - Original byte length: 4 bytes (uint32)
        - Table entry count: 2 bytes (uint16)
        - For each symbol:
            - Symbol byte: 1 byte
            - Frequency count: 4 bytes (uint32)
        - Packed bitstream bytes
    """
    if not data:
        return b""

    # 1. Build tree and codebook
    tree = build_huffman_tree(data)
    codes = build_codes(tree)

    # 2. Encode bytes into bits
    bit_chunks = [codes[b] for b in data]
    bit_string = "".join(bit_chunks)

    # 3. Pack bits into bytearray
    bit_len = len(bit_string)
    packed_bytes = bytearray()
    for idx in range(0, bit_len, 8):
        byte_bits = bit_string[idx : idx + 8]
        # Pad with 0s if last byte is incomplete
        if len(byte_bits) < 8:
            byte_bits = byte_bits.ljust(8, "0")
        packed_bytes.append(int(byte_bits, 2))

    # 4. Serialize header (original length + frequency table)
    freq_map = Counter(data)
    header = bytearray()
    header.extend(struct.pack(">II", len(data), len(freq_map)))
    for sym, count in freq_map.items():
        header.extend(struct.pack(">BI", sym, count))

    return bytes(header) + bytes(packed_bytes)


def huffman_decode(compressed_data: bytes) -> bytes:
    """
    Decode a self-contained Huffman binary stream back into original bytes.
    """
    if not compressed_data:
        return b""

    # 1. Read header
    orig_len, table_size = struct.unpack(">II", compressed_data[:8])
    offset = 8

    # 2. Rebuild frequency map
    freq_map = {}
    for _ in range(table_size):
        sym, count = struct.unpack(">BI", compressed_data[offset : offset + 5])
        freq_map[sym] = count
        offset += 5

    # 3. Rebuild Huffman tree from frequency map
    heap = [HuffmanNode(byte_val=b, freq=f) for b, f in freq_map.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        single = heapq.heappop(heap)
        root = HuffmanNode(byte_val=None, freq=single.freq)
        root.left = single
    else:
        while len(heap) > 1:
            n1 = heapq.heappop(heap)
            n2 = heapq.heappop(heap)
            m = HuffmanNode(byte_val=None, freq=n1.freq + n2.freq)
            m.left = n1
            m.right = n2
            heapq.heappush(heap, m)
        root = heap[0]

    # 4. Read bitstream and traverse tree
    bitstream = compressed_data[offset:]
    decoded = bytearray()
    curr_node = root

    for byte in bitstream:
        # Check bits from MSB to LSB
        for bit_pos in range(7, -1, -1):
            bit = (byte >> bit_pos) & 1
            curr_node = curr_node.left if bit == 0 else curr_node.right

            if curr_node.byte_val is not None:
                decoded.append(curr_node.byte_val)
                curr_node = root
                if len(decoded) == orig_len:
                    return bytes(decoded)

    return bytes(decoded)
