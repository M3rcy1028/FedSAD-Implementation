"""
Run CICFlowMeter-4.0 (Docker image `cicflowmeter:4.0`) over the CICIoT2023 PCAPs.

Selection rule (see docs/CICIoT2023_integration_plan.md §4.2):
  - Benign_Final: every pcap (needed for normal statistics).
  - Classes with a single pcap: that pcap.
  - Classes split into many pcaps: only the smallest one (we downsample to a few
    thousand flows per class anyway).

Usage:
  python preprocess/run_cicflowmeter.py --pcap_root CICIoT/PCAP --out_root CICIoT/FLOW --workers 6
"""
import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGE = "cicflowmeter:4.0"
# Concurrent reads from a Windows bind mount are pathologically slow (455 MB: 3 s alone vs 300 s
# with 6 readers), so the selected pcaps are first copied one by one into a Docker volume.
VOLUME = "ciciot_pcap"
JAVA_CMD = ('mkdir -p /o && java -Xmx{xmx} -Djava.library.path=/usr/lib '
            '-cp "/opt/cfm-dist/CICFlowMeter/lib/*" cic.cs.unb.ca.ifm.Cmd "/pcap/{cls}/{pcap}" /o '
            '> /o/cfm.log 2>&1 && '
            # writing flows straight to the Windows bind mount is ~12x slower, so copy at the end
            'cp /o/*_Flow.csv /out/ && grep -a "Packet stats" /o/cfm.log')
DOCKER_ENV = dict(os.environ, MSYS_NO_PATHCONV="1")


def select_pcaps(pcap_root, all_files=False):
    jobs = []
    for cls in sorted(os.listdir(pcap_root)):
        cls_dir = os.path.join(pcap_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        pcaps = [f for f in os.listdir(cls_dir) if f.endswith(".pcap")]
        if not pcaps:
            continue
        if cls == "Benign_Final" or all_files:
            chosen = pcaps
        else:
            chosen = [min(pcaps, key=lambda f: os.path.getsize(os.path.join(cls_dir, f)))]
        jobs += [(cls, f) for f in sorted(chosen)]
    return jobs


def stage_pcaps(pcap_root, jobs):
    """Copy the selected pcaps into the Docker volume sequentially (skips files already staged)."""
    t0 = time.time()
    for i, (cls, pcap) in enumerate(jobs, 1):
        size = os.path.getsize(os.path.join(pcap_root, cls, pcap))
        sh = (f'mkdir -p "/pcap/{cls}" && '
              f'[ "$(stat -c %s "/pcap/{cls}/{pcap}" 2>/dev/null)" = "{size}" ] || '
              f'cp "/in/{cls}/{pcap}" "/pcap/{cls}/{pcap}"')
        cmd = ["docker", "run", "--rm",
               "-v", f"{os.path.abspath(pcap_root)}:/in:ro", "-v", f"{VOLUME}:/pcap",
               "--entrypoint", "sh", IMAGE, "-c", sh]
        subprocess.run(cmd, check=True, env=DOCKER_ENV)
        print(f"[stage {i}/{len(jobs)}] {cls}/{pcap} ({size / 1e9:.2f} GB, {time.time() - t0:.0f}s)", flush=True)


def run_one(out_root, cls, pcap, xmx):
    out_dir = os.path.join(out_root, cls)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{pcap}_Flow.csv")
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        return cls, pcap, "skip (exists)", 0.0

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{VOLUME}:/pcap:ro",
        "-v", f"{os.path.abspath(out_dir)}:/out",
        "--entrypoint", "sh", IMAGE, "-c", JAVA_CMD.format(xmx=xmx, cls=cls, pcap=pcap),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=DOCKER_ENV)
    status = res.stdout.strip() or res.stderr.strip()[-300:]
    if res.returncode != 0:
        status = f"FAILED rc={res.returncode}: {status}"
    return cls, pcap, status, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap_root", default="CICIoT/PCAP")
    ap.add_argument("--out_root", default="CICIoT/FLOW")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--xmx", default="4g")
    ap.add_argument("--all_files", action="store_true", help="process every pcap of every class")
    args = ap.parse_args()

    jobs = select_pcaps(args.pcap_root, args.all_files)
    total_gb = sum(os.path.getsize(os.path.join(args.pcap_root, c, f)) for c, f in jobs) / 1e9
    print(f"[INFO] {len(jobs)} pcaps selected, {total_gb:.1f} GB", flush=True)

    # largest first so the long jobs don't end up running alone at the tail
    jobs.sort(key=lambda j: -os.path.getsize(os.path.join(args.pcap_root, *j)))
    stage_pcaps(args.pcap_root, jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args.out_root, c, f, args.xmx) for c, f in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            cls, pcap, status, sec = fut.result()
            print(f"[{i}/{len(jobs)}] {cls}/{pcap} ({sec:.0f}s) {status}", flush=True)


if __name__ == "__main__":
    main()
