from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

def evaluate_model(model, test_dataset):
    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()

    # Perform evaluation on test data
    for batch in test_dataset.as_numpy_iterator():
        X_true, y_true = batch
        yhat = model.predict(X_true)

        y_true = y_true.flatten()
        yhat = yhat.flatten()

        pre.update_state(y_true, yhat)
        re.update_state(y_true, yhat)
        acc.update_state(y_true, yhat)

    print(
        f"Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}"
    )
