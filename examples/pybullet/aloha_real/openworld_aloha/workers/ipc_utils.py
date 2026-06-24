import pickle
import struct


def _read_exact(stream, length):
    chunks = []
    remaining = length
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(stream, obj):
    data = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


def recv_msg(stream):
    header = _read_exact(stream, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    payload = _read_exact(stream, length)
    if payload is None:
        return None
    return pickle.loads(payload)
