"""Trust native loopback, or the explicit Docker gateway for local port publishing."""
import ipaddress
import os
from pathlib import Path
import socket
import struct

ROUTE_FILE = Path("/proc/net/route")


def docker_gateway():
    if os.environ.get("GRID_BOT_CONTAINER") != "1" or os.environ.get("GRID_BOT_TRUST_DOCKER_GATEWAY") != "1":
        return None
    try:
        for line in ROUTE_FILE.read_text(encoding="ascii").splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 4 and fields[1] == "00000000" and int(fields[3], 16) & 2:
                address = socket.inet_ntoa(struct.pack("<I", int(fields[2], 16)))
                if ipaddress.ip_address(address).is_private:
                    return address
    except (OSError, ValueError, OverflowError):
        pass
    return None


def trusted_client(host):
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or host == docker_gateway()
