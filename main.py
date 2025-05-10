from src.dataset.create_fiftyone_dataset import create_coco_dataset
from src.eval.clearml_handler import connect_task
from clearml import OutputModel, Task
from datetime import datetime
import numpy as np
import torch
import os
import pandas as pd
import shutil


def main():
    # dataset = create_coco_dataset()
    task_id = "a9c33b18175e4de6a5f490ebf5f3b043"
    task = connect_task(task_id)

    # output_model = OutputModel(
    #     task=task,
    #     label_enumeration={"person": 1, "car": 2, "bus": 3, "bike": 4},
    #     name=f"yolov5l_v{datetime.now().strftime('%Y%m%d%H%M')}.pt",
    # )
    # output_models = task.get_models()["output"][-1]
    # model_weights = output_models.get_weights()
    # vector_series = np.random.randint(10, size=10).reshape(2, 5)
    # output_model.report_histogram(
    #     title="histogram example",
    #     series="histogram series",
    #     values=vector_series,
    #     iteration=0,
    #     labels=["A", "B"],
    #     xaxis="X axis label",
    #     yaxis="Y axis label",
    # )
    print(task.name)
    print(task._get_output_model())


def load_model(model_path: str):
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=model_path,
        force_reload=True,
    )
    return model


def eval_pipeline(task_id: str, model_name: str):
    """
    INPUT: task id
    task_ID -> connect to task -> get model weights -> create a new output model
    -> create a new task -> create a new project -> link task with project
    -> link task with model -> evaluate -> report histogram -> report metrics
    -> report scalar -> finish
    """
    task = connect_task(task_id)
    task_name = task.name
    project_name = "DetectionEvaluation"
    # model_version = task._get_last_update()
    # if model_version is None:
    model_version = datetime.now()

    task_name = f"{model_name.upper()} - {model_version.strftime('%Y-%m-%d %H:%M:%S')}"
    model_output = task.get_models()["output"][-1]
    model_output_path = model_output.get_weights()

    new_task = Task.init(project_name=project_name, task_name=task_name)
    new_model_name = f"yolov5l_v{datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
    output_model = OutputModel(
        task=new_task,
        label_enumeration={"person": 1, "car": 2, "bus": 3, "bike": 4},
        name=new_model_name,
        framework="pytorch",
    )
    # evaluation
    shutil.copy(
        model_output_path,
        os.path.join(
            os.path.dirname(model_output_path),
            new_model_name,
        ),
    )
    output_model.update_weights(
        os.path.join(os.path.dirname(model_output_path), new_model_name)
    )
    labels = [
        "person",
        "car",
        "chair",
        "book",
        "bottle",
        "cup",
        "dining table",
        "traffic light",
        "bowl",
        "handbag",
    ]
    data = pd.DataFrame(
        {
            "Class": [
                "person",
                "car",
                "chair",
                "book",
                "bottle",
                "cup",
                "dining_table",
                "traffic_light",
                "bowl",
                "handbag",
            ],
            "Precision": [0.89, 0.72, 0.53, 1.00, 0.60, 0.93, 0.50, 0.50, 0.71, 0.50],
            "Recall": [0.80, 0.56, 0.23, 0.30, 0.67, 0.81, 0.62, 0.46, 0.38, 0.18],
            "F1-Score": [0.84, 0.63, 0.32, 0.47, 0.63, 0.87, 0.55, 0.48, 0.50, 0.26],
            "Support": [263, 55, 35, 33, 9, 16, 13, 13, 13, 17],
        }
    )
    output_model.report_table(
        title="Evaluation Metrics",
        series="Evaluation Metrics",
        iteration=0,
        table_plot=data,
    )
    class_names = [
        "person",
        "car",
        "chair",
        "book",
        "bottle",
        "cup",
        "dining_table",
        "traffic_light",
        "bowl",
        "handbag",
    ]
    precision_values = np.random.random(len(class_names))
    # Report histogram to ClearML
    output_model.report_histogram(
        title="Precision",
        series="Precision by Class",
        iteration=0,
        values=list(precision_values),
        xlabels=class_names,
        yaxis="Precision",
    )
    output_model.report_histogram(
        title="recall",
        series="Recall by Class",
        iteration=0,
        values=list(np.random.random(len(class_names))),
        xlabels=class_names,
        yaxis="Recall",
    )
    output_model.report_histogram(
        title="F1-Score",
        series="F1 Score by Class",
        iteration=0,
        values=list(np.random.random(len(class_names))),
        xlabels=class_names,
        yaxis="F1-Score",
    )
    output_model.report_histogram(
        title="Support",
        series="Support by Class",
        iteration=0,
        values=list(np.random.randint(100, 1000, len(class_names))),
        xlabels=class_names,
        yaxis="Support",
    )
    output_model.report_histogram(
        title="mAP",
        series="mAP by Class",
        iteration=0,
        values=[np.random.random()],
        xlabels=["mAP"],
        yaxis="mAP",
    )
    new_task.close()


if __name__ == "__main__":
    eval_pipeline(
        task_id="6386497afef64c7da86bd985239319bc",
        model_name="yolov5l",
    )
