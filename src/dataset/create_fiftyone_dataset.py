import fiftyone.zoo as foz


def create_coco_dataset():
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        classes=["person", "car", "bus", "truck", "motorcycle"],
        label_types=["detections"],
    )
    dataset.persistent = True
    return dataset
