import shutil
from pathlib import Path

def copy_html_md(src_dir: Path, dest_dir: Path):
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    for src_sub_dir in src_dir.iterdir():
        if not src_sub_dir.is_dir():
            continue

        src_files = list(src_sub_dir.glob("*.anon.md"))
        if not src_files:
            continue

        dest_sub_dir = dest_dir / src_sub_dir.name
        dest_sub_dir.mkdir(parents=True, exist_ok=True)

        for src_file in src_files:
            if not src_file.is_file():
                continue

            shutil.copy2(src_file, dest_sub_dir / src_file.name)

copy_html_md(Path("G:\\Edu\\_sprint_7\\_work_data\\docs"), Path("G:\\Edu\\_sprint_7\\_work_data\\res"))