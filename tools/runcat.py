import sys
import socket
import threading
import argparse

# FIX: Manually define AF_UNIX for Windows environments where Python hides it
if not hasattr(socket, "AF_UNIX"):
    socket.AF_UNIX = 1  # 1 is the constant value for AF_UNIX in Windows Winsock

def pipe_stdin_to_socket(sock):
    """Reads binary data from stdin and pushes it into the UNIX socket."""
    try:
        while True:
            data = sys.stdin.buffer.read(4096)
            if not data:
                break  # EOF reached on stdin
            sock.sendall(data)
    except (ConnectionResetError, BrokenPipeError):
        pass
    except Exception as e:
        print(f"\nError writing to socket: {e}", file=sys.stderr)
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

def main():
    parser = argparse.ArgumentParser(description="Cross-platform minimalist socat clone for UNIX domain sockets.")
    parser.add_argument("socket_path", help="Path to the UNIX domain socket file.")
    args = parser.parse_args()

    # Establish the connection to the socket using the injected or native family
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client_socket.connect(args.socket_path)
    except Exception as e:
        print(f"Error connecting to {args.socket_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Start background thread to pipe stdin -> socket
    stdin_thread = threading.Thread(target=pipe_stdin_to_socket, args=(client_socket,), daemon=True)
    stdin_thread.start()

    # Read socket -> stdout on the main thread
    try:
        while True:
            response = client_socket.recv(4096)
            if not response:
                break  # Server closed connection
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()
    except (ConnectionResetError, BrokenPipeError):
        pass
    except Exception as e:
        print(f"\nError reading from socket: {e}", file=sys.stderr)
    finally:
        client_socket.close()

if __name__ == "__main__":
    main()
