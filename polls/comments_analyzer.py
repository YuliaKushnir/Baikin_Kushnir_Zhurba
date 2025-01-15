import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from keras import layers
from keras import losses

# Завантаження та видобування даних IMDB до папки aclImdb_v1
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb_v1', 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')

# Підготовка даних для тренування
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

# Побудова словника для токенізації
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, f'[{re.escape(string.punctuation)}]', '')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Вивчення словника на тренувальних даних
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Функція для векторизації вхідних даних
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)

# Оптимізація роботи з даними
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Побудова моделі
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, 16),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'), # Додано новий шар
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss=losses.BinaryCrossentropy(from_logits=False),
              optimizer='adam',
              metrics=[tf.metrics.BinaryAccuracy()])

# Навчання моделі
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

def analyze_sentiment(comment):
    print(f"Original comment: {comment}")
    comment = tf.expand_dims(comment, 0)
    comment = vectorize_layer(comment)
    print(f"Vectorized comment: {comment}")
    prediction = model.predict(comment)
    print(f"Raw prediction: {prediction}")
    sentiment = tf.sigmoid(prediction).numpy()[0][0]
    print(f"Sigmoid applied prediction: {sentiment}")
    if sentiment > 0.5:
        return "Позитивний"
    elif sentiment < 0.5:
        return "Негативний"
    else:
        return "Нейтральний"

