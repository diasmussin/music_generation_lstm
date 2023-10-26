import keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
NUM_UNITS = [256]  # equal to one internal layer in LSTM model with 256 neurons
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64  # number of samples network will see before doing BPP
SAVE_MODEL_PATH = "model.h5"

def build_model(output_units, num_units, loss, learning_rate):

    # create the model architecture (using Keras Functional API)
    input = keras.layers.Input(shape=(None, output_units))   # none - unlimited number of input sequences, output_units - how many elements we have for each time-step
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)  # regularization technique, avoids over-fitting

    output = keras.layers.Dense(output_units, activation="softmax")(x)
    model = keras.Model(input, output)

    # compile the model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model



def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):

    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)
    # output units - neurons in output layer (equal to vocabulary size from JSON mappings)
    # num_units - number of units of neurons in the internal layers of the network
    # loss - loss function

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
