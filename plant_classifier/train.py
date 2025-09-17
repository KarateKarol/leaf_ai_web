from data_loader import load_data
from model import create_model

def main():
    train_ds, val_ds = load_data()

    # liczba klas w dataset
    num_classes = 38  # PlantVillage ma 38 klas (możesz sprawdzić w dataset)

    model = create_model(num_classes)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    model.save("leaf_classifier.h5")
    print(" Model zapisany jako leaf_classifier.h5")

if __name__ == "__main__":
    main()
