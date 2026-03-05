import zipfile, os

project = r'c:\Users\Abhishek\Downloads\AE117_ParkinsonDiseaseDetection-20251208T083701Z-3-001\AE117_ParkinsonDiseaseDetection'
out_zip = os.path.join(os.path.expanduser('~'), 'Downloads', 'parkisense_render_deploy.zip')

# Root files for Render
core_files = {
    'main.py': 'main.py',
    'utils.py': 'utils.py',
    'voiceTest.py': 'voiceTest.py',
    'requirements_render.txt': 'requirements.txt',  # Rename for Render
    'drawing_model.pkl': 'drawing_model.pkl',
    'drawing_scaler.pkl': 'drawing_scaler.pkl',
    'labels.txt': 'labels.txt',
    'mydatabase.db': 'mydatabase.db'
}

include_dirs = ['templates', 'static', 'src']
exclude_subdirs = {'spiral_data', 'spiral_data_clean', 'spiral_data_improved',
                   'DataSet', 'archive_extracted', '__pycache__', '.venv', '.git'}
exclude_exts = {'.pyc', '.bak', '.bak2', '.bak_improved', '.h5', '.toml', 'Dockerfile'}

count = 0
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    # 1. Root files
    for src_name, dst_name in core_files.items():
        fp = os.path.join(project, src_name)
        if os.path.exists(fp):
            zf.write(fp, dst_name)
            count += 1
        else:
            print(f"Warning: {src_name} not found")

    # 2. Directories
    for d in include_dirs:
        dp = os.path.join(project, d)
        if not os.path.isdir(dp):
            continue
        for root, dirs, files in os.walk(dp):
            dirs[:] = [x for x in dirs if x not in exclude_subdirs]
            for file in files:
                if any(file.endswith(e) for e in exclude_exts):
                    continue
                full = os.path.join(root, file)
                rel  = os.path.relpath(full, project)
                zf.write(full, rel)
                count += 1

print('ZIP created for Render:', out_zip)
print('Total files included:', count)
size_mb = os.path.getsize(out_zip) / (1024 * 1024)
print('ZIP size: %.1f MB' % size_mb)
