import argparse
from loguru import logger
from datetime import datetime

import fiftyone as fo
import cv2
import torch
from clearml import Task, OutputModel
import shutil
import os
import tqdm
import os
from pathlib import Path
import pandas as pd

PROJECT_NAME = "DetectionEvaluation"
COCO_LABELS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}


def load_model(model_path: str):
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=model_path,
        force_reload=True,
    )
    return model


def eval(dataset, model):
    for sample in tqdm.tqdm(dataset, desc="Evaluating samples", total=len(dataset)):
        sample_id = sample.id
        try:
            results = model(sample.filepath)
            img = cv2.imread(sample.filepath)
            height, width, _ = img.shape
            predictions = results.pandas().xyxy[0]
            detections = []
            for _, row in predictions.iterrows():
                label = row["name"]
                confidence = row["confidence"]
                x_min = row["xmin"]
                y_min = row["ymin"]
                x_max = row["xmax"]
                y_max = row["ymax"]
                detections.append(
                    fo.Detection(
                        label=label,
                        confidence=confidence,
                        bounding_box=[
                            x_min / width,
                            y_min / height,
                            (x_max - x_min) / width,
                            (y_max - y_min) / height,
                        ],
                    )
                )
            sample["preds_v20250510"] = fo.Detections(detections=detections)
            sample.save()
        except Exception as e:
            logger.error(f"Failed to evaluate sample: {sample_id}. Error: {e}")

    return


def fo_evaluatae(dataset, gt_field, pred_field):
    # Evaluate the dataset
    results = fo.evaluate_detections(
        dataset,
        gt_field=gt_field,
        pred_field=pred_field,
        compute_mAP=True,
        eval_key=f"eval_{pred_field}",
    )
    map = results.mAP()
    return results.report(), map


def clearml_report_histogram(clearml_logger, title, series, values, xlabels, yaxis):
    clearml_logger.report_histogram(
        title=title,
        series=series,
        iteration=0,
        values=values,
        xlabels=xlabels,
        yaxis=yaxis,
    )


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
    training_time = clearml_task._get_last_update()

    # evaluate
    # model = load_model(model_path)
    new_model_name = f"{args.model_name.lower()}_v{training_time.strftime('%Y%m%d')}.pt"

    task_name = (
        f"{args.model_name.upper()} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    task = Task.init(project_name=PROJECT_NAME, task_name=task_name)
    task_logger = task.logger

    report, map = fo_evaluatae(
        dataset,
        gt_field="ground_truth",
        pred_field="preds_yolov5l_v20250511",
    )

    df = pd.DataFrame(report)
    df = df.transpose().loc[list(COCO_LABELS.keys())]

    precision = df["precision"]
    recall = df["recall"]
    f1_score = df["f1-score"]
    support = df["support"]
    mAP = map

    task_logger.report_table(
        title=f"Evaluation Report - {datetime.now().strftime('%Y-%m-%d')}",
        series="Evaluation Report",
        table_plot=df,
        iteration=0,
    )
    clearml_report_histogram(
        task_logger,
        "Precision",
        "Precision",
        precision,
        COCO_LABELS.keys(),
        "Precision",
    )
    clearml_report_histogram(
        task_logger,
        "Recall",
        "Recall",
        recall,
        COCO_LABELS.keys(),
        "Recall",
    )
    clearml_report_histogram(
        task_logger,
        "F1 Score",
        "F1 Score",
        f1_score,
        COCO_LABELS.keys(),
        "F1 Score",
    )
    clearml_report_histogram(
        task_logger,
        "Support",
        "Support",
        support,
        COCO_LABELS.keys(),
        "Support",
    )
    clearml_report_histogram(
        task_logger,
        "mAP",
        "mAP",
        [mAP],
        ["mAP"],
        "mAP",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse task ID, model name, and dataset."
    )
    parser.add_argument(
        "--task_id", type=str, required=True, help="Task ID in string format."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name in string format."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset in string format."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(f"Task ID: {args.task_id}")
    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}")
    main(args)
