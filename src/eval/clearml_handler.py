from clearml import Task, OutputModel


def connect_task(task_id: str) -> Task:
    """
    Connect to a ClearML task using the provided task ID.
    """
    task = Task.get_task(task_id=task_id)
    return task
