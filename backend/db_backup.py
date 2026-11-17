"""Tahlil tarixi (analyses.db) va yuklangan rasmlarni zip'ga eksport/import qiladi.

analyses.db git'da yo'q -- boshqa kompyuterga o'tganda yoki qayta o'rnatganda
statistika yo'qolmasligi uchun shu skript ishlatiladi.

    python backend/db_backup.py export [zip_yo'li]   # default: backups/grid-bot-backup-<vaqt>.zip
    python backend/db_backup.py import <zip_yo'li>   # mavjud ma'lumotga QO'SHADI, hech narsani o'chirmaydi

Import dublikatlarni (timestamp + image_path bir xil) o'tkazib yuboradi, shuning
uchun bir zip'ni ikki marta import qilish xavfsiz.
"""
import datetime as dt
import os
import sqlite3
import sys
import tempfile
import zipfile

import db  # noqa: F401  -- jadval/ustunlar mavjudligini ta'minlaydi (yangi kompyuterda ham)
from config import DB_PATH, UPLOAD_DIR as CONFIG_UPLOAD_DIR

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = str(CONFIG_UPLOAD_DIR)
DEFAULT_BACKUP_DIR = os.path.join(BACKEND_DIR, "..", "backups")
DB_NAME_IN_ZIP = "analyses.db"
UPLOADS_PREFIX = "uploads/"


def export_backup(zip_path: str | None = None) -> str:
    if zip_path is None:
        os.makedirs(DEFAULT_BACKUP_DIR, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        zip_path = os.path.join(DEFAULT_BACKUP_DIR, f"grid-bot-backup-{stamp}.zip")

    with tempfile.TemporaryDirectory() as tmp:
        snapshot = os.path.join(tmp, DB_NAME_IN_ZIP)
        # sqlite backup API -- server ishlab turganda ham izchil nusxa beradi
        src = sqlite3.connect(DB_PATH)
        dst = sqlite3.connect(snapshot)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

        chk = sqlite3.connect(snapshot)
        try:
            count = chk.execute("select count(*) from analyses").fetchone()[0]
        finally:
            chk.close()  # Windows'da ochiq fayl vaqtinchalik papkani o'chirishga to'sqinlik qiladi
        n_images = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(snapshot, DB_NAME_IN_ZIP)
            if os.path.isdir(UPLOAD_DIR):
                for name in os.listdir(UPLOAD_DIR):
                    path = os.path.join(UPLOAD_DIR, name)
                    if os.path.isfile(path) and name != ".gitkeep":
                        zf.write(path, UPLOADS_PREFIX + name)
                        n_images += 1
    print(f"Eksport qilindi: {os.path.abspath(zip_path)}  ({count} ta tahlil, {n_images} ta rasm)")
    return zip_path


def import_backup(zip_path: str) -> dict:
    if not os.path.isfile(zip_path):
        sys.exit(f"Fayl topilmadi: {zip_path}")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            if DB_NAME_IN_ZIP not in zf.namelist():
                sys.exit(f"Zip ichida {DB_NAME_IN_ZIP} yo'q -- bu grid-bot zaxirasi emas")
            zf.extract(DB_NAME_IN_ZIP, tmp)
            image_names = [n for n in zf.namelist() if n.startswith(UPLOADS_PREFIX) and not n.endswith("/")]

            src = sqlite3.connect(os.path.join(tmp, DB_NAME_IN_ZIP))
            dst = sqlite3.connect(DB_PATH)
            try:
                src_cols = [r[1] for r in src.execute("PRAGMA table_info(analyses)")]
                dst_cols = {r[1] for r in dst.execute("PRAGMA table_info(analyses)")}
                cols = [c for c in src_cols if c != "id" and c in dst_cols]  # id yangidan beriladi; eski zip'da yangi ustunlar bo'lmasligi mumkin
                existing = set(dst.execute("select timestamp, image_path from analyses"))

                added = skipped = 0
                col_list = ", ".join(cols)
                for row in src.execute(f"select {col_list} from analyses order by id"):
                    rec = dict(zip(cols, row))
                    if (rec["timestamp"], rec["image_path"]) in existing:
                        skipped += 1
                        continue
                    dst.execute(
                        f"insert into analyses ({col_list}) values ({', '.join('?' * len(cols))})", row
                    )
                    added += 1
                dst.commit()
            finally:
                src.close()
                dst.close()

            os.makedirs(UPLOAD_DIR, exist_ok=True)
            copied = 0
            for name in image_names:
                target = os.path.join(UPLOAD_DIR, os.path.basename(name))
                if not os.path.exists(target):
                    with zf.open(name) as fsrc, open(target, "wb") as fdst:
                        fdst.write(fsrc.read())
                    copied += 1

    print(f"Import tugadi: {added} ta yangi tahlil qo'shildi, {skipped} ta dublikat o'tkazib yuborildi, {copied} ta rasm ko'chirildi")
    return {"added": added, "skipped": skipped, "images": copied}


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ("export", "import"):
        sys.exit(__doc__)
    if args[0] == "export":
        export_backup(args[1] if len(args) > 1 else None)
    else:
        if len(args) < 2:
            sys.exit("Import uchun zip yo'lini bering: python backend/db_backup.py import <zip>")
        import_backup(args[1])


if __name__ == "__main__":
    main()
