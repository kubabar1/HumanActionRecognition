import os
import shutil
import zipfile


def setup_test_resources(test_results_path, test_dataset_path, test_dataset_resource_path):
    if os.path.exists(test_results_path):
        shutil.rmtree(test_results_path)
    if os.path.exists(test_dataset_path):
        shutil.rmtree(test_dataset_path)
    with zipfile.ZipFile(test_dataset_resource_path, 'r') as zip_ref:
        zip_ref.extractall(test_dataset_path)
