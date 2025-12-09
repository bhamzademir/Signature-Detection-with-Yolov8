#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from typing import Optional
import requests

def read_ids(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                yield s

def main():
    ap = argparse.ArgumentParser(description="IDs -> DMS GET -> JSONL")
    ap.add_argument("--base", default="http://10.208.12.24:9007",
                    help="Base URL (http/https dahil)")
    ap.add_argument("--path", default="/dms-document/get-file-byId/{id}",
                    help="Endpoint yolu; {id} yer tutucu")
    ap.add_argument("--ids", required=True, help="ID listesi dosyası (satır başına bir ID)")
    ap.add_argument("--out", default="responses.jsonl", help="Çıkış JSONL dosyası")
    ap.add_argument("--token", default=os.getenv("X_AUTH_TOKEN"),
                    help="x-auth-token değeri (ENV: X_AUTH_TOKEN)")
    ap.add_argument("--xff", default="10.208.12.24", help="x-forwarded-for (opsiyonel)")
    ap.add_argument("--timeout", type=int, default=120, help="Timeout (sn)")
    ap.add_argument("--retry", type=int, default=2, help="Başarısızlıkta tekrar sayısı")
    ap.add_argument("--sleep", type=float, default=0.0, help="İstekler arası bekleme (sn)")
    args = ap.parse_args()

    if not args.token:
        print("HATA: --token verilmedi ve X_AUTH_TOKEN ortam değişkeni yok.", file=sys.stderr)
        sys.exit(1)

    sess = requests.Session()
    sess.headers.update({
        "x-auth-token": args.token,
        "x-forwarded-for": args.xff
    })

    total = 0
    ok = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for id_ in read_ids(args.ids):
            total += 1
            url = args.base.rstrip("/") + args.path.format(id=id_)
            tries = args.retry + 1

            for attempt in range(1, tries + 1):
                try:
                    print(f"[{total}] GET {url} (deneme {attempt}/{tries})")
                    r = sess.get(url, timeout=args.timeout, allow_redirects=True)
                    status = r.status_code
                    # JSON parse etmeye çalış
                    try:
                        data = r.json()
                    except Exception:
                        data = {"_raw": r.text}

                    # JSONL'e her zaman yaz: id, http status ve payload
                    rec = {"id": id_, "status": status, "payload": data}
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    w.flush()

                    if status == 200:
                        print(f"    -> OK (status=200)")
                        ok += 1
                    else:
                        print(f"    -> WARN (HTTP {status})")
                    break
                except requests.RequestException as e:
                    print(f"    -> HATA: {e}")
                    if attempt == tries:
                        # Yine de JSONL'e hata kaydı bırak
                        rec = {"id": id_, "status": None, "error": str(e)}
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        w.flush()
                time.sleep(1.0 if attempt < tries else 0.0)

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"\nBitti. Başarılı istek sayısı (200): {ok}/{total}")
    print(f"JSONL çıktı: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
