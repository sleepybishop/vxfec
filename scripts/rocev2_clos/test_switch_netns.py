"""Kernel forwarding regression; run with sudo unshare --mount --net --fork."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def run(*args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def helper_source():
    source = Path(__file__).with_name('run_rocev2_clos.sh').read_text()
    return re.search(r'^disable_link_offloads\(\) \{.*?^\}', source, re.M | re.S).group()


def read_stats(path):
    if not path.exists(): return {}
    return {key: int(value) for key, value in (word.split('=') for word in path.read_text().split())}


class Fabric:
    def __init__(self, binary, directory):
        self.directory = directory
        self.anchors = []; self.process = None; self.log = None
        self.stats = directory / 'switch.stats'
        self.binary = binary
        try:
            for index in (0, 1):
                child = subprocess.Popen(['unshare', '--net', 'sleep', '120'])
                self.anchors.append(child)
                for _ in range(200):
                    if os.readlink(f'/proc/{child.pid}/ns/net') != os.readlink('/proc/self/ns/net'): break
                    time.sleep(.005)
                else: raise RuntimeError('network namespace was not created')
                run('ip', 'link', 'add', f'sw{index}', 'type', 'veth', 'peer', 'name', f'host{index}')
                run('ip', 'link', 'set', f'host{index}', 'netns', str(child.pid))
                self.peer(index, 'ip', 'link', 'set', f'host{index}', 'name', 'eth0')
                self.peer(index, 'sysctl', '-qw', 'net.ipv6.conf.all.disable_ipv6=1')
                self.peer(index, 'ip', 'link', 'set', 'lo', 'up')
                self.peer(index, 'ip', 'link', 'set', 'eth0', 'mtu', '9000', 'up')
                self.peer(index, 'ip', 'addr', 'add', f'192.0.2.{index+1}/24', 'dev', 'eth0')
                run('ip', 'link', 'set', f'sw{index}', 'mtu', '9000', 'up')
        except BaseException:
            self.close()
            raise

    def peer(self, index, *args):
        return run('nsenter', '-t', str(self.anchors[index].pid), '-n', *args)

    def disable_offloads(self):
        # Exercise the launcher's implementation, substituting anonymous namespace handles.
        script = '''set -eu
ip() { [[ "$1 $2" == 'netns exec' ]]; nsenter -t "$3" -n "${@:4}"; }
''' + helper_source() + '\n'
        for index in (0, 1):
            script += f'disable_link_offloads {self.anchors[index].pid} eth0\n'
            script += f'disable_link_offloads {os.getpid()} sw{index}\n'
        run('bash', '-c', script)

    def start(self):
        self.log = (self.directory / 'switch.log').open('w')
        self.process = subprocess.Popen([str(self.binary), 'switch', 'sw0', 'sw1', '1000', '4096', '4', '0', str(self.stats)],
                                        stdout=self.log, stderr=self.log)
        for _ in range(400):
            if self.process.poll() is not None: raise RuntimeError('switch exited during startup')
            if read_stats(self.stats).get('valid') == 1: return
            time.sleep(.005)
        raise RuntimeError('missing initial switch statistics')

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate(); self.process.wait(timeout=5)
        if self.log: self.log.close()
        for child in self.anchors:
            child.terminate(); child.wait(timeout=5)
        for index in (0, 1):
            subprocess.run(['ip', 'link', 'del', f'sw{index}'], capture_output=True)

    def transfer(self, protocol, segment_count=32):
        receiver = r'''
import hashlib, socket, struct, sys
protocol, count = sys.argv[1], int(sys.argv[2])
payload = b''.join(bytes([i]) * 1200 for i in range(count))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM if protocol == 'tcp' else socket.SOCK_DGRAM)
s.settimeout(8); s.bind(('192.0.2.2', 38192))
if protocol == 'tcp': s.listen(1)
else: s.setsockopt(socket.IPPROTO_IP, socket.IP_RECVTOS, 1)
print('READY', flush=True)
if protocol == 'tcp':
    c, _ = s.accept(); c.settimeout(8); actual = b''
    while len(actual) < len(payload):
        part = c.recv(65536)
        assert part
        actual += part
    assert actual == payload
else:
    marks = 0
    for i in range(count):
        part, ancillary, _, _ = s.recvmsg(2048, 64)
        assert part == bytes([i]) * 1200, (i, len(part))
        marks += any(level == socket.IPPROTO_IP and kind == socket.IP_TOS and (value[0] & 3) == 3
                     for level, kind, value in ancillary)
    assert marks > 0, 'ECN-marked packets must retain valid checksums'
print('PASS', flush=True)
'''
        recv = subprocess.Popen(['nsenter', '-t', str(self.anchors[1].pid), '-n', sys.executable,
                                 '-u', '-c', receiver, protocol, str(segment_count)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert recv.stdout.readline().strip() == 'READY'
            sender = r'''
import socket, struct, sys
protocol, count = sys.argv[1], int(sys.argv[2])
payload = b''.join(bytes([i]) * 1200 for i in range(count))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM if protocol == 'tcp' else socket.SOCK_DGRAM)
s.settimeout(5)
if protocol == 'tcp':
    s.connect(('192.0.2.2', 38192)); s.sendall(payload)
else:
    s.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 2)
    assert s.sendmsg([payload], [(socket.IPPROTO_UDP, 103, struct.pack('H', 1200))], 0, ('192.0.2.2', 38192)) == len(payload)
'''
            self.peer(0, sys.executable, '-c', sender, protocol, str(segment_count))
            stdout, stderr = recv.communicate(timeout=12)
            assert recv.returncode == 0 and 'PASS' in stdout, (stdout, stderr)
        finally:
            if recv.poll() is None: recv.kill()
            recv.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--switch', type=Path, required=True)
    args = parser.parse_args()
    for kind in ('mnt', 'net'):
        assert os.readlink(f'/proc/self/ns/{kind}') != os.readlink(f'/proc/1/ns/{kind}'), 'use private mount/network namespaces'
    run('sysctl', '-qw', 'net.ipv6.conf.all.disable_ipv6=1')
    with tempfile.TemporaryDirectory(prefix='vxfec-clos-regression-') as temporary:
        root = Path(temporary)
        for count, key in ((32, 'ab_rx_truncated'), (4, 'ab_rx_gso')):
            directory = root / f'gso-{count}'; directory.mkdir()
            fabric = Fabric(args.switch.resolve(), directory)
            try:
                # Preserve veth GSO to deliberately test aggregate rejection.
                for index in (0, 1):
                    fabric.peer(index, 'ethtool', '-K', 'eth0', 'tso', 'on', 'gso', 'on', 'gro', 'on', 'tx-udp-segmentation', 'on')
                    run('ethtool', '-K', f'sw{index}', 'tso', 'on', 'gso', 'on', 'gro', 'on', 'tx-udp-segmentation', 'on')
                fabric.start()
                fabric.peer(0, sys.executable, '-c', f"import socket,struct; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.sendmsg([b'x'*{1200*count}],[(socket.IPPROTO_UDP,103,struct.pack('H',1200))],0,('192.0.2.2',38192))")
                status = fabric.process.wait(timeout=8)
                stats = read_stats(fabric.stats)
                assert status != 0 and stats['valid'] == 0 and stats[key] > 0, (status, stats, (directory/'switch.log').read_text())
                print(f'PASS: GSO {1200*count}-byte aggregate rejected and recorded as {key}', flush=True)
            finally: fabric.close()
        directory = root / 'segmented'; directory.mkdir()
        fabric = Fabric(args.switch.resolve(), directory)
        try:
            fabric.disable_offloads(); fabric.start()
            fabric.transfer('udp', 32)
            fabric.transfer('tcp', 64)
            fabric.process.terminate(); assert fabric.process.wait(timeout=5) == 0
            stats = read_stats(fabric.stats)
            assert stats['valid'] == 1 and stats['tot_invalid_events'] == 0, stats
            assert stats['tot_tx_pkts'] >= 32 and stats['tot_app_ecn_marks'] > 0, stats
            print('PASS: software-segmented large UDP/TCP payloads and ECN checksums; zero observed socket drops or forwarding errors', flush=True)
        finally: fabric.close()


if __name__ == '__main__': main()
