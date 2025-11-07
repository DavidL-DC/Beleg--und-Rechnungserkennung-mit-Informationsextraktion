from dateparser import parse as dparse
from price_parser import Price
import re
from typing import Literal


def df_lines(df):
    # Gruppiere pro Seite/Block/Zeile, damit nichts vermischt wird
    for k in ("page_num", "block_num", "line_num"):
        if k not in df.columns:
            df[k] = 0
    g = (df.groupby(["page_num", "block_num", "line_num"], sort=True)["text"]
           .apply(list)
           .reset_index())
    out = []
    for _, row in g.iterrows():
        toks = [t for t in row.text if str(t).strip()]
        if toks:
            line_id = (int(row.page_num), int(row.block_num), int(row.line_num))
            out.append((line_id, toks))
    return out

def ngrams(tokens, n_max=4):
    T = len(tokens)
    for n in range(1, n_max+1):
        for i in range(T - n + 1):
            yield " ".join(tokens[i:i+n])

def parse_date_robust(s):
    # Zwei Versuche: US/EN und dann DE
    for date_order in ("MDY", "DMY"):
        dt = dparse(s, languages=["en","de"], settings={"PREFER_DAY_OF_MONTH": "first", "DATE_ORDER": date_order})
        if dt:
            return dt.date().isoformat()
    return None

def parse_money_robust(s):
    p = Price.fromstring(s)
    if p.amount is None:
        return None, None
    try:
        return round(float(p.amount), 2), (p.currency or "").upper() or None
    except:
        return None, (p.currency or "").upper() or None

def find_candidates(df, n_max=4):
    DECIMAL_2 = re.compile(r"[\.,]\d{2}\b")
    NUMBERS = re.compile(r"\b\d+(?:\s*[-\u2010-\u2015]\s*\d+)*\b")
    lines = df_lines(df)
    date_cands, money_cands, number_cands = [], [], []
    for lid, toks in lines:
        joined = list(ngrams(toks, n_max=n_max))
        for chunk in joined:
            d = parse_date_robust(chunk)
            if d:
                date_cands.append({"line_id": lid, "text": chunk, "value": d})
            for m in NUMBERS.finditer(chunk):
                raw = m.group(0)
                # Normalisieren: Leerzeichen & Dashes raus
                only_minus = re.sub(r"[\u2010-\u2015]", "-", raw)
                norm_number = re.sub(r"\s", "", only_minus)
                try:
                    value = int(norm_number)
                except ValueError:
                    value = norm_number
                number_cands.append({"line_id": lid, "text": chunk, "value": value, "raw": raw})
            amt, cur = parse_money_robust(chunk)
            if amt is not None:
                if cur is None and not DECIMAL_2.search(chunk):
                    continue
                amt = int(amt) if amt.is_integer() else amt
                money_cands.append({"line_id": lid, "text": chunk, "amount": amt, "currency": cur})

    return date_cands, money_cands, number_cands


# Label-Definitionen für Zeilen
LABELS = {
    "amount_due": re.compile(r"\b(?:amount\s*(?:now\s*)?due|balance\s*(?:now\s*)?due|amount\s*payable|payable\s*amount|amount\s*to\s*pay|pay\s*now|due|gross\s*worth|offener\s*(?:rest)?betrag|f[aä]lliger\s*betrag|zahlbetrag|zu\s*zahlen|betrag\s*f[aä]llig)(?!\w)", re.I),
    "tax": re.compile(r"\b(mwst|ust|tax|vat|sales\s*tax|mehrwertsteuer|umsatzsteuer)(?!\w)", re.I),
    "date_issue": re.compile(r"\b(?:invoice\s*date|date\s*of\s*issue|issue\s*date|issued\s*on|rechnungsdatum|ausstellungsdatum|erstellt\s*am)(?!\w)", re.I),
    "date_due": re.compile(r"\b(?:due\s*date|date\s*due|payment\s*due(?:\s*date)?|due\s*on|pay\s*by|payable\s*by|due|f[aä]lligkeits?datum|zahlung\s*f[aä]llig(?:\s*am|\s*bis)?|sp[aä]testens\s*bis)(?!\w)", re.I),
    "inv": re.compile(r"\b(?:invoice\s*(?:no|nr|number|#)|rechnungsnr|rechnungsnummer|rg-?nr)(?!\w)", re.I),
    "cust": re.compile(r"\b(?:customer\s*id|kundennr|kundennummer|customerid|cid)(?!\w)", re.I),
}

def df_lines_text(df):
    # Text je (Seite, Block, Zeile) – für Label-Erkennung
    for k in ("page_num", "block_num", "line_num"):
        if k not in df.columns:
            df[k] = 0
    g = (df.groupby(["page_num", "block_num", "line_num"], sort=True)["text"]
           .apply(lambda s: " ".join(t for t in s.astype(str) if t.strip()))
           .reset_index())
    out = []
    for _, r in g.iterrows():
        if r.text.strip():
            line_id = (int(r.page_num), int(r.block_num), int(r.line_num))
            out.append((line_id, r.text))
    return out

def detect_labels_per_line(lines):
    out = {}
    for ln, txt in lines:
        found = set()
        for key, rx in LABELS.items():
            if rx.search(txt):
                found.add(key)
        if found:
            out[ln] = found
    return out


def pick_best_date(date_cands, line_labels, label_key: Literal["date_issue", "date_due"]):
    labeled = [lid for lid, labs in line_labels.items() if label_key in labs]
    if not labeled or not date_cands:
        return None
    
    def same_block(a, b):
        return(a[0], a[1]) == (b[0], b[1])
    
    def line_dist(a, b):
        return abs(a[2] - b[2]) if same_block(a, b) else 10**6

    best, best_score = None, -1.0
    for d in date_cands:
        lid = d["line_id"]
        score = 0.0
        if lid in line_labels and label_key in line_labels[lid]:
            score += 0.8
        for llid in labeled:
            dis = line_dist(lid, llid)
            if dis < 10**6:
                score += max(0.0, 0.8 - 0.2 * dis)
        score += min(0.2, 0.02 * len(d["text"]))
        if score > best_score:
            best, best_score = d, score
    return best


def pick_best_amount(money_cands, line_labels, label_key: Literal["amount_due", "tax"]):
    labeled = [lid for lid, labs in line_labels.items() if label_key in labs]
    if not labeled or not money_cands:
        return None

    def same_block(a, b):
        return(a[0], a[1]) == (b[0], b[1])
    
    def line_dist(a, b):
        return abs(a[2] - b[2]) if same_block(a, b) else 10**6

    max_amt = max(m.get("amount") or 0 for m in money_cands)
    min_amt = min(m.get("amount") or 0 for m in money_cands)

    best, best_score = None, -1.0
    for m in money_cands:
        lid = m["line_id"]
        score = 0.0

        # Starker Boost, wenn die Zeile als richtiges Label erkennbar ist
        if lid in line_labels and label_key in line_labels[lid]:
            score += 0.8
        
        # Moderater Boost bei Nähe zu passender Zeile
        for llid in labeled:
            dis = line_dist(lid, llid)
            if dis < 10**6:
                score += max(0.0, 0.8 - 0.2 * dis)

        # Bonus: Währung vorhanden
        if m.get("currency"):
            score += 0.8

        # Boost je nach größe des Betrags
        amt = m.get("amount") or 0.0
        if label_key == "amount_due" and max_amt > 0:
            amt_norm = (amt / max_amt) ** 0.5  # Werte 0..1
            score += 0.7 * amt_norm
        if label_key == "tax" and min_amt > 0:
            amt_norm = (min_amt / amt) ** 0.5  # Werte 0..1+
            score += 0.7 * amt_norm

        if score > best_score:
            best, best_score = m, score

    return best


def pick_best_number(number_cands, line_labels, label_key: Literal["inv", "cust"]):
    labeled = [lid for lid, labs in line_labels.items() if label_key in labs]
    if not labeled:
        return None
    
    def same_block(a, b):
        return(a[0], a[1]) == (b[0], b[1])
    
    def line_dist(a, b):
        return abs(a[2] - b[2]) if same_block(a, b) else 10**6

    best, best_score = None, -1.0
    for n in number_cands:
        lid = n["line_id"]
        score = 0.0
        if lid in line_labels and label_key in line_labels[lid]:
            score += 0.8
        for llid in labeled:
            dis = line_dist(lid, llid)
            if dis < 10**6:
                score += max(0.0, 0.8 - 0.2 * dis)
        score += min(0.2, 0.02 * len(n["text"]))
        if score > best_score:
            best, best_score = n, score
    return best