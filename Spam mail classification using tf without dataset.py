import tensorflow as tf
import numpy as np

# --- Custom dataset ---
texts = [
    "You won a free iPhone! Click now",
    "Earn money fast, limited offer",
    "Claim your free reward today",
    "Urgent: update your account info",
    "Winner! You have been selected",
    "Hey, are we meeting today?",
    "Please send the report",
    "Let's have lunch tomorrow",
    "Call me when you're free",
    "I will join the meeting soon"
]

labels = np.array([1,1,1,1,1, 0,0,0,0,0])  # 1 = spam, 0 = ham

# --- Text vectorizer ---
vec = tf.keras.layers.TextVectorization(max_tokens=1000, output_sequence_length=20)
vec.adapt(tf.constant(texts)) 

# --- model ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.string),
    vec,
    tf.keras.layers.Embedding(1000, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# --- Train ---
model.fit(tf.constant(texts), labels, epochs=30, verbose=0)

def predict(msg):
    p = float(model.predict(tf.constant([msg]), verbose=0)[0][0])
    print("\nMessage:", msg)
    print("Prediction:", "SPAM" if p >= 0.5 else "HAM")
    print("Score:", round(p, 4))

# --- Test ---
predict("Free prize! Click to win")
predict("Send me the assignment file")
