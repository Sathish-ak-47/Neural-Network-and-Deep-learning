import os, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split

# ---- Config ----
DATA_FILE = "emails.csv"
BATCH, EPOCHS = 32, 4
MAX_TOKENS, SEQ_LEN, EMBED_DIM = 8000, 120, 64
MODEL_FILE = "spam_model.keras"
VOCAB_FILE = "spam_model_vocab.txt"

# ---- Load & normalize ----
def load_csv(p):
    if not os.path.exists(p): raise FileNotFoundError(p)
    df = pd.read_csv(p, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.lower()
    # try common patterns
    if "v1" in df and "v2" in df:
        df = df[["v1","v2"]].rename(columns={"v1":"label","v2":"text"})
    elif "text" in df and "label" in df:
        df = df[["text","label"]]
    elif "text" in df and "spam" in df:
        df = df.rename(columns={"spam":"label"})[["text","label"]]
    else:
        df = df.iloc[:, :2]
        df.columns = ["text","label"]
    df = df.dropna().reset_index(drop=True)
    df['label'] = df['label'].astype(str).str.strip().str.lower().replace({'ham':'0','spam':'1'})
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    return df[df['label'].isin([0,1])]

df = load_csv(DATA_FILE)
train_text, val_text, train_labels, val_labels = train_test_split(
    df['text'].astype(str), df['label'], test_size=0.15, random_state=42, stratify=df['label']
)

# ---- Vectorizer & datasets ----
vec = tf.keras.layers.TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN)
vec.adapt(train_text.values)

def make_ds(texts, labels, batch=BATCH, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((texts.values, labels.values))
    if shuffle: ds = ds.shuffle(len(texts), seed=42)
    return ds.batch(batch).map(lambda x,y: (vec(x), y)).prefetch(tf.data.AUTOTUNE)

train_ds, val_ds = make_ds(train_text, train_labels), make_ds(val_text, val_labels, shuffle=False)

# ---- Model ----
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Input(shape=(SEQ_LEN,)),
    layers.Embedding(MAX_TOKENS, EMBED_DIM),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# ---- Train / Eval / Save ----
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"Validation loss={loss:.4f}  acc={acc:.4f}")
model.save(MODEL_FILE)
with open(VOCAB_FILE, "w", encoding="utf-8") as f: f.write("\n".join(vec.get_vocabulary()))

# ---- Predict helper ----
def predict_text(m, v, text):
    # Apply the vectorizer to the text before prediction
    p = float(m.predict(v([text]), verbose=0)[0,0])  # Correctly vectorize input
    label = "SPAM" if p >= 0.5 else "NOT SPAM"
    print("\n==============================")
    print(f"Input Message : {text}")
    print(f"Prediction    : {label}")
    print(f"Spam Score    : {p:.4f}")
    print("==============================\n")
    return p

# ---- Quick test ----
if _name_ == "_main_":
    predict_text(model, vec, "Congratulations! You've won a free prize. Click here.")
    predict_text(model, vec, "Hey, are we still meeting tomorrow?")
