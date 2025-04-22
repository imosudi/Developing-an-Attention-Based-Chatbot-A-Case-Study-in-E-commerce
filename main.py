from src.preprocess import load_and_clean_data, tokenize_inputs
from src.train_classifier import build_classifier
from evaluation.evaluate_classifier import evaluate
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

# Load and preprocess
path = 'data/customer_support_twitter.csv'
data, label_encoder = load_and_clean_data(path)
inputs = tokenize_inputs(data['text'])

# Convert TensorFlow tensors to numpy arrays before splitting
input_ids_np = inputs['input_ids'].numpy()
attention_mask_np = inputs['attention_mask'].numpy() if 'attention_mask' in inputs else None
labels_np = np.array(data['label'])

# Split data
if attention_mask_np is not None:
    (X_train, X_test, 
     mask_train, mask_test,
     y_train, y_test) = train_test_split(
        input_ids_np,
        attention_mask_np,
        labels_np,
        test_size=0.2,
        random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        input_ids_np,
        labels_np,
        test_size=0.2,
        random_state=42
    )

# Convert back to TensorFlow tensors for training
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

if attention_mask_np is not None:
    mask_train = tf.convert_to_tensor(mask_train)
    mask_test = tf.convert_to_tensor(mask_test)

# Train model
model = build_classifier(num_labels=len(label_encoder.classes_))

if attention_mask_np is not None:
    model.fit(
        x={'input_ids': X_train, 'attention_mask': mask_train},
        y=y_train,
        validation_data=({'input_ids': X_test, 'attention_mask': mask_test}, y_test),
        epochs=2,
        batch_size=16
    )
else:
    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        epochs=2,
        batch_size=16
    )

# Evaluate
evaluate(model, X_test, y_test, label_encoder)