import cv2
import pandas as pd
from api import extract_from_img
import argparse
from pathlib import Path
from datetime import datetime

if __name__ == "__main__":

    EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    parser = argparse.ArgumentParser(description="Nightly-Batch für Rechnungs-OCR")
    parser.add_argument("--in", dest="in_dir", required=True, help="Eingangsordner mit Bildern")
    parser.add_argument("--out", dest="out_dir", required=True, help="Ordner für CSV-Ausgabe")
    parser.add_argument("--archive", dest="archive_dir", default=None, help="(Optional) Verarbeitete Bilder hierhin verschieben")
    args = parser.parse_args()

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.archive_dir:
        archive_dir = Path(args.archive_dir); archive_dir.mkdir(parents=True, exist_ok=True)
    else:
        archive_dir = None

    rows = []
    processed = 0
    failed = []

    for fp in sorted(in_dir.iterdir()):
        if fp.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(fp))
        if img is None:
            failed.append((fp.name, "Dekodierung fehlgeschlagen"))
            continue
        try:
            res = extract_from_img(img)
            rows.append({
                "filename": fp.name,
                "amount_due": res["amount_due"],
                "tax": res["tax"],
                "date_issue": res["date_issue"],
                "date_due": res["date_due"],
                "invoice_number": res["invoice_number"],
                "customer_number": res["customer_number"],
            })
            processed += 1
            if archive_dir:
                sub = archive_dir / datetime.now().strftime("%Y-%m-%d")
                sub.mkdir(parents=True, exist_ok=True)
                fp.rename(sub / fp.name)
        except Exception as e:
            failed.append((fp.name, str(e)))

    if rows:
        ts = datetime.now().strftime("%Y-%m-%d")
        out_csv = out_dir / f"invoices_{ts}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, encoding="utf-8", sep=";")
        print(f"[OK] {processed} Dateien → {out_csv}")
    else:
        print("[OK] Keine neuen Bilder gefunden.")

    if failed:
        for name, err in failed:
            print(f"[FEHLER] {name}: {err}")
