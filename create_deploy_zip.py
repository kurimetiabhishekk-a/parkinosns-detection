import zipfile, os

project = r'c:\Users\Abhishek\Downloads\AE117_ParkinsonDiseaseDetection-20251208T083701Z-3-001\AE117_ParkinsonDiseaseDetection'
out_zip = os.path.join(os.path.expanduser('~'), 'Downloads', 'parkisense_deploy.zip')

include_files = [
    'main.py', 'utils.py', 'find_features.py',
    'labels.txt', 'keras_model.h5', 'trainedModel.sav',
    'requirements_pythonanywhere.txt', 'wsgi.py',
]
include_dirs = ['templates', 'static']
exclude_subdirs = {'spiral_data', 'spiral_data_clean', 'spiral_data_improved',
                   'DataSet', 'archive_extracted', '__pycache__'}
exclude_exts = {'.pyc', '.bak', '.bak2', '.bak_improved'}

count = 0
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in include_files:
        fp = os.path.join(project, f)
        if os.path.exists(fp):
            zf.write(fp, 'parkisense/' + f)
            count += 1

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
                zf.write(full, 'parkisense/' + rel)
                count += 1

    db = os.path.join(project, 'mydatabase.db')
    if os.path.exists(db):
        zf.write(db, 'parkisense/mydatabase.db')
        count += 1

size_mb = os.path.getsize(out_zip) / (1024 * 1024)
print('ZIP created:', out_zip)
print('Files included:', count)
print('ZIP size: %.1f MB' % size_mb)
