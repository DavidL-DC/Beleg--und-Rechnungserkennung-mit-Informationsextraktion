import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from daten_extrahieren import find_candidates, df_lines_text, detect_labels_per_line, pick_best_amount, pick_best_date, pick_best_number
from typing import List, Optional
import pandas as pd


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

LANG = "deu+eng"

app = FastAPI(title="Invoice OCR (minimal)", version="1.0")

def extract_from_img(img_img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    data = pytesseract.image_to_data(th, lang=LANG, output_type=pytesseract.Output.DATAFRAME)

    df = data[(data.conf >= 0) & (data.text.notna()) & (data.text.str.strip() != "")]
    df = df.copy()
    df["conf_norm"] = (df["conf"] / 100.0).round(3)

    dates, monies, numbs = find_candidates(df, n_max=4)
    lines = df_lines_text(df)
    line_labels = detect_labels_per_line(lines)

    best_total = pick_best_amount(monies, line_labels, label_key="amount_due")
    best_tax = pick_best_amount(monies, line_labels, label_key="tax")
    best_date_issue  = pick_best_date(dates, line_labels, "date_issue")
    best_date_due    = pick_best_date(dates, line_labels, "date_due")
    best_customer_number = pick_best_number(numbs, line_labels, "cust")
    best_invoice_number  = pick_best_number(numbs, line_labels, "inv")

    return {
        "amount_due": f"{best_total['amount']}{best_total['currency']}" if best_total else None,
        "tax": f"{best_tax['amount']}{best_tax['currency']}" if best_tax else None,
        "date_issue": best_date_issue["value"] if best_date_issue else None,
        "date_due": best_date_due["value"] if best_date_due else None,
        "customer_number": str(best_customer_number["value"]) if best_customer_number else None,
        "invoice_number": str(best_invoice_number["value"]) if best_invoice_number else None,
    }

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        return JSONResponse(
            status_code=400,
            content={"error": "Bitte ein Bild (.jpg/.png/.tif) hochladen."}
        )
    content = await file.read()
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Bild konnte nicht dekodiert werden."})
    result = extract_from_img(img)
    return result

@app.post("/extract-batch")
async def extract_batch(files: List[UploadFile] = File(..., description="Mehrere Bilddateien"), save_csv: bool = False, csv_path: Optional[str] = None):
    results = []
    rows = []

    for f in files:
        if not f.filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            results.append({"filename": f.filename, "error": "Ungültiges Format"})
            continue
        content = await f.read()
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            results.append({"filename": f.filename, "error": "Bild konnte nicht dekodiert werden."})
            continue

        res = extract_from_img(img)
        results.append({"filename": f.filename, "result": res})

        rows.append({
            "filename": f.filename,
            "amount_due": res["amount_due"],
            "tax": res["tax"],
            "date_issue": res["date_issue"],
            "date_due": res["date_due"],
            "invoice_number": res["invoice_number"],
            "customer_number": res["customer_number"],
        })

    # CSV optional speichern
    csv_written = None
    if save_csv:
        df = pd.DataFrame(rows)
        path = csv_path or "batch_results.csv"
        df.to_csv(path, index=False, encoding="utf-8", sep=";")
        csv_written = path

    return {
        "count": len(results),
        "csv_path": csv_written,
        "items": results
    }
