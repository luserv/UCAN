import os
import tarfile
import shutil
from huggingface_hub import hf_hub_download

SCALE = 4
BASE_DIR = "datasets/SR"

# (local_dir, hub_id, tar_prefix)
datasets = [
    ("Set5", "eugenesiow/Set5", "Set5"),
    ("Set14", "eugenesiow/Set14", "Set14"),
    ("B100", "eugenesiow/BSD100", "BSD100"),
    ("Urban100", "eugenesiow/Urban100", "Urban100"),
]

for name, hub_id, tar_prefix in datasets:
    print(f"\n=== {name} ===")

    for suffix, dest_subdir in [("HR", "HR"), (f"LR_x{SCALE}", f"LR_bicubic/X{SCALE}")]:
        dest_dir = os.path.join(BASE_DIR, name, dest_subdir)
        if os.path.exists(dest_dir) and any(os.scandir(dest_dir)):
            print(f"  {dest_subdir} ya existe ({len(os.listdir(dest_dir))} archivos), saltando...")
            continue

        tar_file = f"data/{tar_prefix}_{suffix}.tar.gz"
        print(f"  Descargando {tar_file}...")
        tar_path = hf_hub_download(repo_id=hub_id, filename=tar_file, repo_type="dataset")

        print(f"  Extrayendo a {dest_dir}...")
        os.makedirs(dest_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)

        items = list(os.scandir(dest_dir))
        if len(items) == 1 and items[0].is_dir():
            subdir = items[0].path
            for f in os.scandir(subdir):
                shutil.move(f.path, os.path.join(dest_dir, os.path.basename(f.path)))
            os.rmdir(subdir)

        count = len(os.listdir(dest_dir))
        print(f"  Extraídas {count} imágenes")

print("\n¡Todos los datasets descargados!")
