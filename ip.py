# auto_call_esp_final.py
# Corrects Wi-Fi detection: always chooses the active adapter with default gateway.
# Works fully on Windows.

import socket, http.client, concurrent.futures, urllib.parse, subprocess, re, time

port = 80
path = "/esp"
msg = "iamespcam"
timeout_s = 0.8
max_workers = 200

def get_active_ip():
    """Return active Windows adapter IP (with default gateway or named Wi-Fi)."""
    try:
        output = subprocess.check_output("ipconfig", text=True, encoding='utf-8', errors='ignore')
        blocks = output.split("\n\n")
        # prefer block having 'Default Gateway' with a real value
        for block in blocks:
            if "Default Gateway" in block and re.search(r"Default Gateway.+?:\s*\d+\.\d+\.\d+\.\d+", block):
                ip = re.search(r"IPv4[^:]*:\s*([\d\.]+)", block)
                if ip:
                    return ip.group(1)
        # fallback specifically to Wi-Fi section
        for block in blocks:
            if "Wireless LAN adapter Wi-Fi" in block:
                ip = re.search(r"IPv4[^:]*:\s*([\d\.]+)", block)
                if ip:
                    return ip.group(1)
        # final fallback: first private IP not in 169.254
        all_ips = re.findall(r"IPv4[^:]*:\s*([\d\.]+)", output)
        for ip in all_ips:
            if not ip.startswith("169.254."):
                return ip
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None

def call_esp(ip):
    url_path = f"{path}?{urllib.parse.urlencode({'msg':msg})}"
    try:
        conn = http.client.HTTPConnection(ip, port, timeout=timeout_s)
        conn.request("GET", url_path)
        resp = conn.getresponse()
        data = resp.read(200).decode(errors="ignore")
        snippet = data.strip().splitlines()[0] if data else ""
        conn.close()
        return (ip, resp.status, snippet)
    except Exception:
        return (ip, None, None)

def main():
    local_ip = get_active_ip()
    if not local_ip:
        print("❌ Unable to detect active network IP.")
        return
    base = '.'.join(local_ip.split('.')[:3])
    print(f"✅ Active IP: {local_ip}")
    print(f"Scanning {base}.1–254 for {path} on port {port}\n")

    ips = [f"{base}.{i}" for i in range(1,255) if f"{base}.{i}" != local_ip]
    found = []
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(call_esp, ip): ip for ip in ips}
        for fut in concurrent.futures.as_completed(futs):
            ip = futs[fut]
            ip,status,snippet = fut.result()
            if status:
                found.append((ip,status,snippet))
                print(f"{ip} -> {status} {snippet}")

    dur = time.time() - start
    print(f"\nDone. {len(found)} device(s) responded. Time: {dur:.1f}s")

if __name__ == "__main__":
    main()
