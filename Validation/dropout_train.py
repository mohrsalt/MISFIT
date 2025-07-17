import os
import random
import numpy as np
import shutil

# --- Input train directories ---
train_dirs = [
    '/home/Mohor.Banerjee@mbzuai.ac.ae/Task8DataBrats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData',
    '/home/Mohor.Banerjee@mbzuai.ac.ae/Task8DataBrats/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData',
    '/home/Mohor.Banerjee@mbzuai.ac.ae/Task8DataBrats/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional'
]

# --- Output pseudo-train folder ---
pseudo_train_folder = '/home/Mohor.Banerjee@mbzuai.ac.ae/Task8DataBrats/pseudo_train_set'
os.makedirs(pseudo_train_folder, exist_ok=True)

# --- Modalities ---
modality_list = ['t1c', 't1n', 't2f', 't2w']
np.random.seed(42)

# --- Collect all subject folders ---
all_subjects = []
for d in train_dirs:
    subjects = [os.path.join(d, f) for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
    all_subjects.extend(subjects)

# --- Sample 15 randomly ---
sampled_subjects = random.sample(all_subjects, 15)

# --- Process each subject ---
for subject_path in sampled_subjects:
    subject_id = os.path.basename(subject_path)
    dest_subject_path = os.path.join(pseudo_train_folder, subject_id)
    os.makedirs(dest_subject_path, exist_ok=True)

    drop_modality = random.choice(modality_list)

    for fname in os.listdir(subject_path):
        if not any(fname.endswith(ext) for ext in ['.nii', '.nii.gz']):
            continue  # skip non-image files

        if drop_modality in fname:
            print(f"{drop_modality} is dropped for subject {subject_id}")
            new_fname = f"Missing_Target_{fname}"
        else:
            new_fname = fname

        src = os.path.join(subject_path, fname)
        dst = os.path.join(dest_subject_path, new_fname)
        shutil.copyfile(src, dst)
