#!/usr/bin/env python3
import os
import sys
import re
import json
import base64
import argparse
from typing import Optional

def sniff_mime_from_b64(b64: str) -> str:
    # Baş prefikse göre hızlı tür tahmini
    if b64.startswith("JVBER"): return "application/pdf"    # %PDF
    if b64.startswith("/9j/"):  return "image/jpeg"
    if b64.startswith("iVBOR"): return "image/png"
    if b64.startswith("R0lGOD"):return "image/gif"
    if b64.startswith("UEsDB"): return "application/zip"
    return "application/octet-stream"

def filename_for(id_: str, mime: str) -> str:
    if mime == "application/pdf":
        return f"file_{id_}.pdf"
    if mime.startswith("image/"):
        return f"file_{id_}.{mime.split('/')[-1]}"
    if mime == "application/zip":
        return f"file_{id_}.zip"
    return f"file_{id_}.bin"

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield lineno, json.loads(s)
            except json.JSONDecodeError as e:
                yield lineno, {"_malformed": s, "_error": str(e)}

def main():
    ap = argparse.ArgumentParser(description="JSONL -> base64 decode -> dosyalar")
    ap.add_argument("--in", dest="inp", required=True, help="Girdi JSONL (fetch_json.py çıktısı)")
    ap.add_argument("--outdir", default="downloads", help="Çıktı klasörü")
    ap.add_argument("--skip-non200", action="store_true",
                    help="HTTP 200 olmayanları atla (varsayılan: kaydetmez ama ERROR dosyası yazar)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    total = 0
    saved = 0
    errors = 0

    for lineno, rec in iter_jsonl(args.inp):
        total += 1

        id_ = str(rec.get("id") or f"line{lineno}")
        status = rec.get("status")
        payload = rec.get("payload", {})
        if args.skip_non200 and status != 200:
            print(f"[{total}] id={id_}: HTTP {status} -> atlandı (--skip-non200).")
            continue

        if not isinstance(payload, dict):
            # fetch_json bir hata yazmış olabilir
            errfile = os.path.join(args.outdir, f"ERROR_{id_}.txt")
            with open(errfile, "w", encoding="utf-8") as w:
                w.write(f"Line {lineno} malformed payload or error: {payload}\n")
            errors += 1
            print(f"[{total}] id={id_}: payload dict değil -> ERROR dosyasına yazıldı.")
            continue

        b64 = payload.get("responseObject")
        if not b64:
            errfile = os.path.join(args.outdir, f"ERROR_{id_}.txt")
            with open(errfile, "w", encoding="utf-8") as w:
                w.write(f"Line {lineno} no responseObject; status={status}\n")
            errors += 1
            print(f"[{total}] id={id_}: responseObject yok -> ERROR dosyasına yazıldı.")
            continue

        try:
            mime = sniff_mime_from_b64(b64)
            raw = base64.b64decode(b64)
            fname = filename_for(id_, mime)
            dest = os.path.join(args.outdir, fname)
            with open(dest, "wb") as w:
                w.write(raw)
            saved += 1
            print(f"[{total}] id={id_}: -> {dest} ({len(raw)} bayt, {mime})")
        except Exception as e:
            errfile = os.path.join(args.outdir, f"ERROR_{id_}.txt")
            with open(errfile, "w", encoding="utf-8") as w:
                w.write(f"Line {lineno} decode error: {e}\n")
            errors += 1
            print(f"[{total}] id={id_}: decode hatası -> ERROR dosyasına yazıldı.")

    print(f"\nTamamlandı. Kaydedilen: {saved}/{total}, Hata: {errors}, Klasör: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
