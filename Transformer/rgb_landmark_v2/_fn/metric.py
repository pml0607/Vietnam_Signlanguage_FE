import numpy as np

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}