<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> 164a03b44130c429459dffce95f5bcc4cd9a2e03
"""
Bitstream I/O for quantized CLIP embeddings.

The format is intentionally simple: a magic header, the compressed length,
and a zstd-compressed stream of bytes representing the quantized vector.
"""
<<<<<<< HEAD
=======
>>>>>>> origin/master
>>>>>>> 164a03b44130c429459dffce95f5bcc4cd9a2e03

import struct
import numpy as np
import zstandard as zstd
from pathlib import Path

<<<<<<< HEAD

MAGIC = b"CLPF"
=======
<<<<<<< HEAD
MAGIC = b'CLPF'
>>>>>>> 164a03b44130c429459dffce95f5bcc4cd9a2e03
VERSION = 1


def write_bitstream(q_bytes: bytes, dim: int, out_path: Path) -> None:
    comp = zstd.ZstdCompressor(level=22).compress(q_bytes)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(comp)))
        f.write(comp)


def read_bitstream(in_path: Path) -> np.ndarray:
    with open(in_path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, "Bad magic"
        ln = struct.unpack("<I", f.read(4))[0]
        comp = f.read(ln)
    raw = zstd.ZstdDecompressor().decompress(comp)
    q = np.frombuffer(raw, dtype=np.uint8)
<<<<<<< HEAD
    return q
=======
    return q
=======

MAGIC = b"CLPF"
VERSION = 1


def write_bitstream(q_bytes: bytes, dim: int, out_path: Path) -> None:
    comp = zstd.ZstdCompressor(level=22).compress(q_bytes)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(comp)))
        f.write(comp)


def read_bitstream(in_path: Path) -> np.ndarray:
    with open(in_path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, "Bad magic"
        ln = struct.unpack("<I", f.read(4))[0]
        comp = f.read(ln)
    raw = zstd.ZstdDecompressor().decompress(comp)
    q = np.frombuffer(raw, dtype=np.uint8)
    return q
>>>>>>> origin/master
>>>>>>> 164a03b44130c429459dffce95f5bcc4cd9a2e03
