from transformers import TFBertForSequenceClassification
import tensorflow as tf

def build_classifier(num_labels):
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model