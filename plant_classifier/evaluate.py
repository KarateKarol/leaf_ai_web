import tensorflow as tf
from data_loader import load_data

def evaluate():
    _, val_ds = load_data()
    model = tf.keras.models.load_model("leaf_classifier.h5")
    loss, acc = model.evaluate(val_ds)
    print(f" Dokładność: {acc*100:.2f}%")

if __name__ == "__main__":
    evaluate()
