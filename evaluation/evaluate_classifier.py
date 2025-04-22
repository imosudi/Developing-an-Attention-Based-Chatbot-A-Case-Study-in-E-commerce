from sklearn.metrics import classification_report
import numpy as np

def evaluate(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test).logits
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
