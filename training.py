import tensorflow as tf


def set_memory_growth():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("GPU is available and memory growth is enabled.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")
else:
    print("No GPU available, using CPU.")
    
def train_model(model, x_train, y_train, epochs=250, batch_size=8):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
