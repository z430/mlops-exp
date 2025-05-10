import argparse
from loguru import logger
from datetime import datetime

import fiftyone as fo
import torch
from clearml import Task, OutputModel
import shutil
import os

PROJECT_NAME = "DetectionEvaluation"


def load_model(model_path: str):
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=model_path,
        force_reload=True,
    )
    return model


def main(args):
    try:
        dataset = fo.load_dataset(args.dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {args.dataset}. Error: {e}")
        return

    clearml_task = Task.get_task(task_id=args.task_id)
    task_name = clearml_task.name
    model_output = clearml_task.get_models()["output"][-1]
    model_path = model_output.get_weights()

    task_name = f"{args.model_name.upper()} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    task = Task.init(project_name=PROJECT_NAME, task_name=task_name)
    new_model_name = f"{args.model_name.lower()}_v{datetime.now().strftime('%Y%m%d%H%M%S')}.pt"

    # Create a new output model
    output_model = OutputModel(
        task=task,
        label_enumeration={"person": 1, "car": 2, "bus": 3, "bike": 4},
        name=new_model_name,
        framework="pytorch",
    )
    shutil.copy(
        model_path,
        os.path.join(
            os.path.dirname(model_path),
            new_model_name,
        ),
    )
    output_model.update_weights_package(os.path.join(os.path.dirname(model_path), new_model_name))

    # evaluate


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse task ID, model name, and dataset.")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID in string format.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name in string format.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset in string format.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(f"Task ID: {args.task_id}")
    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}")
    main(args)
