import argparse
import os
import sys
import requests

# =========================
# PocketBase configuration
# =========================

PB_URL = os.getenv("PB_URL", "http://127.0.0.1:8090").rstrip("/")
PB_TOKEN = os.getenv("PB_TOKEN", "")  # Optional auth token

# True: preview changes without modifying the database.
# False: actually update records.
DRY_RUN = False

PAGE_SIZE = 200

# Old English codes -> new English codes
ENGLISH_MAP = {
    "A": "A",
    "B": "B",
    "D": "De",
    "Gh": "GH",
    "H": "H",
    "J": "J",
    "L": "L",
    "M": "M",
    "N": "N",
    "P": "P",
    "PuV": "EIN",
    "PwD": "ZHE",
    "Sad": "SAD",
    "Sin": "Sin",
    "T": "T",
    "Taxi": "X",
    "V": "V",
    "Y": "Y",
}

# Old Persian characters -> new Persian categories
PERSIAN_MAP = {
    "آ": "الف",
    "ب": "ب",
    "د": "د",
    "ه": "ه",
    "ج": "ج",
    "ل": "ل",
    "م": "م",
    "ن": "ن",
    "پ": "پ",
    "ع": "ع",
    "ژ": "ژ (معلولین و جانبازان)",
    "ص": "ص",
    "س": "س",
    "ط": "ط",
    "ت": "ت",
    "و": "و",
    "ی": "ی",
}

# New Persian categories -> canonical English codes
NEW_MAP = {
    "الف": "A",
    "ب": "B",
    "ت": "X",
    "تشریفات": "CER",
    "ث": "Se",
    "ج": "J",
    "د": "De",
    "ز": "Z",
    "س": "Sin",
    "ش": "SH",
    "ص": "SAD",
    "ط": "T",
    "ظ": "ZA",
    "ع": "EIN",
    "ف": "F",
    "ق": "GH",
    "ل": "L",
    "م": "M",
    "ن": "N",
    "ه": "H",
    # "ه‌": "HX",
    "و": "V",
    "پ": "P",
    "ژ (معلولین و جانبازان)": "ZHE",
    "ک": "K",
    "گ": "G",
    "ی": "Y",
}

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

if PB_TOKEN:
    session.headers["Authorization"] = PB_TOKEN


def convert_english(value):
    """Convert old English codes, preserving unknown values."""
    if not isinstance(value, str):
        return value
    return ENGLISH_MAP.get(value, value)


def convert_persian(value):
    """Convert old Persian labels to the new Persian labels."""
    if not isinstance(value, str):
        return value
    return PERSIAN_MAP.get(value, value)


def convert_plate_number(value):
    """
    Convert known English codes embedded in plateNumber.
    Unknown characters and digits remain unchanged.
    """
    if not isinstance(value, str):
        return value

    for old, new in ENGLISH_MAP.items():
        value = value.replace(old, new)

    return value


# Map each field name to its converter. Any field passed on the command
# line that isn't listed here falls back to convert_plate_number.
CONVERTERS = {
    "englishAlphabet": convert_english,
    "persinalAlphabet": convert_persian,
}


def get_all_records(collection):
    """Fetch every record in the collection, page by page."""
    page = 1

    while True:
        response = session.get(
            f"{PB_URL}/api/collections/{collection}/records",
            params={"page": page, "perPage": PAGE_SIZE},
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        records = result.get("items", [])

        if not records:
            break

        for record in records:
            yield record

        if page >= result.get("totalPages", 1):
            break

        page += 1


def migrate(collection, fields):
    scanned = 0
    updated = 0
    failed = 0

    print(f"Collection: {collection}")
    print(f"Fields: {', '.join(fields)}")
    print(f"Dry run: {DRY_RUN}\n")

    for record in get_all_records(collection):
        scanned += 1
        record_id = record["id"]
        changes = {}

        for field in fields:
            if field not in record:
                continue

            old_value = record[field]
            converter = CONVERTERS.get(field, convert_plate_number)
            new_value = converter(old_value)

            if new_value != old_value:
                changes[field] = new_value

        if not changes:
            continue

        print(
            f"[{'DRY RUN' if DRY_RUN else 'UPDATE'}] "
            f"{record_id}: {changes}"
        )

        if DRY_RUN:
            updated += 1
            continue

        try:
            response = session.patch(
                f"{PB_URL}/api/collections/{collection}/records/{record_id}",
                json=changes,
                timeout=30,
            )
            response.raise_for_status()
            updated += 1

        except requests.RequestException as exc:
            failed += 1
            print(f"[ERROR] Record {record_id}: {exc}")
            if getattr(exc, "response", None) is not None:
                print(exc.response.text)

    print("\n========== SUMMARY ==========")
    print(f"Records scanned: {scanned}")
    print(f"Records changed / would change: {updated}")
    print(f"Failed updates: {failed}")
    print(f"Dry run: {DRY_RUN}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate old alphabet codes in a PocketBase collection."
    )
    parser.add_argument(
        "collection",
        help="Name of the PocketBase collection to migrate",
    )
    parser.add_argument(
        "fields",
        nargs="+",
        help="Field names to convert, e.g. englishAlphabet persinalAlphabet plateNumber",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        migrate(args.collection, args.fields)
    except requests.RequestException as exc:
        print(f"PocketBase connection/API error: {exc}")
        sys.exit(1)