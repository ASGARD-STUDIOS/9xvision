"""Where live perception goes: OSC (TouchDesigner, Max, Ableton, Resolume...), WebSocket (browsers, games, agents)
and JSON lines (scripts, logs).

The WebSocket server listens on 127.0.0.1 unless told otherwise. Browsers may connect only from the server's own
monitor page or an origin passed with allow_origins (cross-site pages are refused, so a random web page cannot
read the camera's signals), and the Host header must be local (no DNS rebinding). Camera frames and control
commands also need the session token."""
from __future__ import annotations

import asyncio
import json
import re
import socket
import struct
import sys
import threading
from importlib import resources
from typing import Callable

# ---------------------------------------------------------------- OSC


def _pad(b: bytes) -> bytes:
    return b + b"\0" * (4 - len(b) % 4)


def osc_message(address: str, *args) -> bytes:
    tags, data = ",", b""
    for a in args:
        if isinstance(a, bool):
            tags += "T" if a else "F"
        elif isinstance(a, int):
            tags += "i"
            data += struct.pack(">i", a)
        elif isinstance(a, float):
            tags += "f"
            data += struct.pack(">f", a)
        else:
            tags += "s"
            data += _pad(str(a).encode())
    return _pad(address.encode()) + _pad(tags.encode()) + data


def osc_bundle(messages: list[bytes]) -> bytes:
    out = _pad(b"#bundle") + struct.pack(">Q", 1)          # timetag 1 = immediately
    for m in messages:
        out += struct.pack(">i", len(m)) + m
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def osc_addresses(msg: dict, prefix: str = "/9xv") -> list[bytes]:
    """A state becomes one message per number (/9xv/affect/arousal 0.62, /9xv/face/blend/jawOpen 0.1, ...);
    an event becomes /9xv/event/<kind> with its fields as arguments."""
    out: list[bytes] = []
    if msg.get("type") == "event":
        args = [v for k, v in msg.items() if k not in ("type", "kind", "t") and isinstance(v, (int, float, str, bool))]
        return [osc_message(f"{prefix}/event/{msg['kind']}", *args)]

    def walk(path: str, v) -> None:
        if isinstance(v, bool):
            out.append(osc_message(path, 1.0 if v else 0.0))
        elif isinstance(v, (int, float)):
            out.append(osc_message(path, float(v)))
        elif isinstance(v, str):
            out.append(osc_message(path, v))
        elif isinstance(v, dict):
            for k, x in v.items():
                if k not in ("components", "palette"):
                    walk(f"{path}/{_slug(k) if path.endswith('/watch') else k}", x)
        elif isinstance(v, list) and v and isinstance(v[0], dict) and "hand" in v[0]:
            for h in v:
                walk(f"{path}/{h['hand'].lower()}", {k: x for k, x in h.items() if k != "hand"})
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            out.append(osc_message(path, *[float(x) for x in v]))
    for key in ("video", "audio", "face", "hands", "body", "affect", "watch"):
        if msg.get(key) is not None:
            walk(f"{prefix}/{key}", msg[key])
    # presence flags, so a receiver can drop values that are only stale (nothing is sent for an absent face/hand)
    out.append(osc_message(f"{prefix}/face/present", 1.0 if msg.get("face") else 0.0))
    if msg.get("hands") is not None:
        seen = {h.get("hand", "").lower() for h in msg["hands"]}
        for side in ("left", "right"):
            out.append(osc_message(f"{prefix}/hands/{side}/present", 1.0 if side in seen else 0.0))
    return out


def parse_target(target: str) -> tuple[str, int]:
    """"host:port" or just "port" (this machine) -> (host, port), with a readable error."""
    host, _, port = target.rpartition(":")
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(f"OSC target {target!r}: use HOST:PORT or PORT, e.g. 127.0.0.1:9000")
    host = host.strip("[]") or "127.0.0.1"
    try:
        socket.getaddrinfo(host, int(port), type=socket.SOCK_DGRAM)
    except socket.gaierror:
        raise ValueError(f"OSC target {target!r}: unknown host {host!r}") from None
    return host, int(port)


class OscOut:
    def __init__(self, target: str):
        self.addr = parse_target(target)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg: dict) -> None:
        msgs = osc_addresses(msg)
        # keep each datagram well under typical UDP limits
        for k in range(0, len(msgs), 40):
            try:
                self.sock.sendto(osc_bundle(msgs[k:k + 40]), self.addr)
            except OSError:
                pass


# ---------------------------------------------------------------- JSON lines


class JsonlOut:
    def __init__(self, target: str):
        self.f = sys.stdout if target == "-" else open(target, "a", buffering=1)

    def send(self, msg: dict) -> None:
        self.f.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self.f.flush()


# ---------------------------------------------------------------- WebSocket + monitor page


class WsServer:
    """Broadcasts every message to connected clients and answers {"cmd": ...} requests through `control`."""

    def __init__(self, host: str, port: int, token: str, control: Callable[[dict], dict],
                 frame: Callable[[], bytes | None], allow_origins: list[str] | None = None):
        self.host, self.port, self.token = host, port, token
        self.control, self.frame = control, frame
        self.allow = set(allow_origins or [])
        self.clients: set = set()
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.error: Exception | None = None
        threading.Thread(target=self._run, daemon=True).start()
        self.ready.wait(10)
        if self.error:
            raise self.error

    # -- security
    def _local_host(self, req) -> bool:
        h = (req.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return h in ("127.0.0.1", "localhost", "::1") or (self.host not in ("127.0.0.1", "localhost") and h == self.host) \
            or self.host == "0.0.0.0"

    def _origin_ok(self, req) -> bool:
        o = req.headers.get("Origin")
        if o is None:
            return True                       # native clients (TouchDesigner, scripts) send no Origin
        own = {f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}", f"http://{self.host}:{self.port}"}
        return o in own or o in self.allow

    def _has_token(self, path: str) -> bool:
        return bool(re.search(r"[?&]k=" + re.escape(self.token) + r"(&|$)", path))

    def _process(self, conn, req):
        from websockets.datastructures import Headers
        from websockets.http11 import Response
        if not self._local_host(req) or not self._origin_ok(req):
            return Response(403, "Forbidden", Headers([("Content-Type", "text/plain")]), b"origin or host not allowed\n")
        if req.headers.get("Upgrade", "").lower() == "websocket":
            conn.authed = self._has_token(req.path)
            return None
        path = req.path.split("?")[0]
        pages = {"/": "live_monitor.html", "/index.html": "live_monitor.html", "/mirror": "live_mirror.html"}
        if path in pages:
            html = resources.files("ninexvision").joinpath(pages[path]).read_bytes()
            return Response(200, "OK", Headers([("Content-Type", "text/html; charset=utf-8"), ("Cache-Control", "no-store")]), html)
        if path == "/frame.jpg":
            if not self._has_token(req.path):
                return Response(401, "Unauthorized", Headers([("Content-Type", "text/plain")]), b"token needed\n")
            jpg = self.frame()
            if jpg is None:
                return Response(404, "Not Found", Headers([]), b"")
            return Response(200, "OK", Headers([("Content-Type", "image/jpeg"), ("Cache-Control", "no-store")]), jpg)
        return Response(404, "Not Found", Headers([("Content-Type", "text/plain")]), b"not found\n")

    async def _handler(self, conn) -> None:
        self.clients.add(conn)
        try:
            async for raw in conn:
                try:
                    req = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not getattr(conn, "authed", False):
                    await conn.send(json.dumps({"id": req.get("id"), "error": "control needs the session token (?k=...)"}))
                    continue
                try:
                    reply = await self.loop.run_in_executor(None, self.control, req)
                except Exception as e:      # a failed command is an answer, not a dropped connection
                    reply = {"error": f"{type(e).__name__}: {e}"}
                await conn.send(json.dumps({"id": req.get("id"), "type": "reply", **reply}))
        except Exception:
            pass
        finally:
            self.clients.discard(conn)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        from websockets.asyncio.server import serve

        async def main():
            try:
                self.server = await serve(self._handler, self.host, self.port, process_request=self._process,
                                          compression=None, max_queue=4, ping_interval=None)
            except OSError as e:
                self.error = RuntimeError(f"can't listen on {self.host}:{self.port} ({e.strerror}); pick another --port")
                self.ready.set()
                return
            self.ready.set()
            await self.server.serve_forever()
        try:
            self.loop.run_until_complete(main())
        except Exception as e:
            self.error = e
            self.ready.set()

    def send(self, msg: dict) -> None:
        if not self.clients:
            return
        data = json.dumps(msg, separators=(",", ":"))

        def push():
            for c in list(self.clients):
                if getattr(c, "pending", 0) > 8:      # a slow client skips messages instead of piling them up
                    continue
                c.pending = getattr(c, "pending", 0) + 1
                fut = asyncio.ensure_future(c.send(data))
                fut.add_done_callback(lambda f, c=c: setattr(c, "pending", c.pending - 1))
        self.loop.call_soon_threadsafe(push)

    def close(self) -> None:
        if getattr(self, "server", None):
            self.loop.call_soon_threadsafe(self.server.close)
