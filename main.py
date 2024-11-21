from data_loader import load_data
from image_detection import train_model

if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = load_data()
    train_model(X_train, Y_train, X_val, Y_val)
