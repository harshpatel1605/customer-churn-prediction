import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name = "CustomerChurnPrediction"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/predicton_pipeline.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/utils/exception.py",
    f"src/{project_name}/utils/common.py",
    "artifacts"
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok = True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                pass
            logging.info(f"Creating empty file: {filepath}")

        else:
            logging.info(f"{filename} already exists — skipping")

