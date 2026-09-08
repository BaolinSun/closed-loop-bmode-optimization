# -*- coding: utf-8 -*-
"""极简 DICOM Part-10 读取器：只取我们需要的几个标签和像素数据。"""
import struct
import numpy as np

VR_WITH_LONG_LENGTH = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"}


def read_dicom(path):
    data = open(path, "rb").read()
    if data[128:132] != b"DICM":
        raise ValueError("Not a DICOM Part-10 file")
    pos = 132
    tags, pixel = {}, None
    while pos + 8 <= len(data):
        group, element = struct.unpack_from("<HH", data, pos)
        vr = data[pos + 4:pos + 6]
        if vr.isalpha() and vr.isupper():
            if vr in VR_WITH_LONG_LENGTH:
                length = struct.unpack_from("<I", data, pos + 8)[0]
                head = 12
            else:
                length = struct.unpack_from("<H", data, pos + 6)[0]
                head = 8
        else:                                   # implicit VR
            length = struct.unpack_from("<I", data, pos + 4)[0]
            vr, head = b"UN", 8
        start = pos + head
        if (group, element) == (0x7FE0, 0x0010):
            pixel = (start, length, vr)
            if length == 0xFFFFFFFF:            # encapsulated / undefined length
                pixel = (start, len(data) - start, vr)
            break
        value = data[start:start + length]
        if vr in (b"US", b"SS") and length >= 2:
            value = struct.unpack_from("<H", value, 0)[0]
        elif vr in (b"UL", b"SL") and length >= 4:
            value = struct.unpack_from("<I", value, 0)[0]
        else:
            try:
                value = value.decode("latin-1").strip("\x00 ")
            except Exception:
                pass
        tags[(group, element)] = value
        pos = start + (length if length != 0xFFFFFFFF else 0)
    return data, tags, pixel


NAMES = {
    (0x0008, 0x0060): "Modality",
    (0x0028, 0x0002): "SamplesPerPixel",
    (0x0028, 0x0004): "PhotometricInterpretation",
    (0x0028, 0x0008): "NumberOfFrames",
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
    (0x0028, 0x0100): "BitsAllocated",
    (0x0028, 0x0101): "BitsStored",
    (0x0002, 0x0010): "TransferSyntaxUID",
    (0x0018, 0x6011): "SequenceOfUltrasoundRegions",
    (0x0028, 0x0006): "PlanarConfiguration",
}
