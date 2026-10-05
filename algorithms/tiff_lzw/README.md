# TIFF-LZW Image Compression

This package encodes image pixels into a standards-compliant TIFF container
using lossless LZW compression. The TIFF Compression tag is `259`, and the LZW
value stored by the codec is `5`.

## API

```python
from algorithms import TiffLzwCompressor

codec = TiffLzwCompressor()
compressed = codec.compress("input.bmp")
restored = codec.decompress(compressed)
restored.save("restored.bmp")
```

The codec accepts file paths, Pillow images, and NumPy arrays. Input is
normalized to `L`, `RGB`, or `RGBA` so that pixel comparisons are consistent
across all benchmark methods.

`compress()` and `decompress()` work in memory. Use `compress_file()` and
`decompress_file()` when files need to be written to disk.
