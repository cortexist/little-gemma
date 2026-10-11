"""Model-free transport gate: binary prefixes before newline, duplex, EOF drain."""
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


def receive(peer, size):
    data = bytearray()
    while len(data) < size:
        part = peer.recv(size-len(data))
        assert part, 'Unexpected EOF'
        data.extend(part)
    return bytes(data)


with tempfile.TemporaryDirectory(prefix='cat-') as root:
    path = str(Path(root)/'s')
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(path)
        server.listen(1)
        server.settimeout(3)
        proc = subprocess.Popen([sys.argv[1], path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            with server.accept()[0] as peer:
                peer.settimeout(3)
                prefix = b'\1T\0\0\0\0\5\0\0\0hello'
                proc.stdin.write(prefix)
                proc.stdin.flush()
                assert receive(peer, len(prefix)) == prefix, 'Prefix must arrive before turn closure'
                reply = b'early\x00\x1a\r\nreply'
                peer.sendall(reply)
                assert proc.stdout.read(len(reply)) == reply, 'Reply must flow while input is still open'
                payload = bytes(range(256))*40
                proc.stdin.write(payload)
                proc.stdin.close()
                assert receive(peer, len(payload)) == payload, 'Preserve binary and partial reads'
                assert peer.recv(1) == b'', 'Input EOF must half-close'
                peer.sendall(b'final reply')
            assert proc.stdout.read() == b'final reply', 'Drain output after stdin EOF'
            assert proc.wait(timeout=3) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
print('socket_cat binary prefix / duplex / EOF gate passed')
