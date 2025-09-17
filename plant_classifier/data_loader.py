import deeplake
import tensorflow as tf

def load_data():
    # Pobieranie datasetu PlantVillage bez augmentacji
    ds = deeplake.load('hub://activeloop/plantvillage-without-augmentation')

    # Konwersja do tensorflow dataset
    train_ds = ds.tensorflow(
        tensors=["images", "labels"],
        split="train",
        shuffle=True,
        batch_size=32
    )

    val_ds = ds.tensorflow(
        tensors=["images", "labels"],
        split="test",
        shuffle=False,
        batch_size=32
    )

    return train_ds, val_ds
