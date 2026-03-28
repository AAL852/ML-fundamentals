"""
CAB420 Assignment 1A — Question 3: Deep Networks
=================================================
Training and comparing DCNNs (with and without data augmentation)
against an SVM baseline on the SVHN digit recognition dataset.

Dataset: Q3/q3_train.mat (1,000 samples), Q3/q3_test.mat (10,000 samples)
Task:    10-class digit classification (0–9)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat
from time import process_time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ── 1. Utility Functions ──────────────────────────────────────────────────────
def load_data(train_path: str, test_path: str):
    """Load SVHN .mat files and return normalised numpy arrays."""
    train = loadmat(train_path)
    test  = loadmat(test_path)

    # Transpose to (N, H, W, C) and normalise to [0, 1]
    train_X = np.transpose(train['train_X'], (3, 0, 1, 2)) / 255.0
    train_Y = train['train_Y'].reshape(-1)
    test_X  = np.transpose(test['test_X'],  (3, 0, 1, 2)) / 255.0
    test_Y  = test['test_Y'].reshape(-1)

    # Remap label '10' → '0' (SVHN uses 10 to represent the digit 0)
    train_Y[train_Y == 10] = 0
    test_Y[test_Y == 10]   = 0

    return train_X, train_Y, test_X, test_Y


def vectorise(X: np.ndarray) -> np.ndarray:
    """Flatten image arrays to 1D vectors for use with sklearn models."""
    return X.reshape(X.shape[0], -1)


def plot_images(X: np.ndarray, Y: np.ndarray, n: int = 10):
    """Display a row of sample images with their labels."""
    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i, ax in enumerate(axes):
        img = X[i] if X[i].shape[-1] == 3 else X[i, :, :, 0]
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(str(Y[i]))
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ── 2. Data Loading ───────────────────────────────────────────────────────────
train_X, train_Y, test_X, test_Y = load_data(
    '../Data/Q3/q3_train.mat',
    '../Data/Q3/q3_test.mat',
)

print(f"Train: {train_X.shape} | Test: {test_X.shape}")
print(f"Classes: {np.unique(train_Y)}")
plot_images(train_X, train_Y)


# ── 3. DCNN Architecture ──────────────────────────────────────────────────────
def build_dcnn(input_shape: tuple, name: str = 'dcnn') -> keras.Model:
    """
    Three-block convolutional network for 10-class digit classification.

    Architecture:
        Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64)
        → Flatten → Dense(64) → Dense(10, softmax)
    """
    model = models.Sequential(name=name)
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


input_shape = train_X.shape[1:]  # (32, 32, 3)


# ── 4. Train DCNN — No Augmentation ──────────────────────────────────────────
print("\n[DCNN] Training without augmentation...")
dcnn = build_dcnn(input_shape, name='dcnn_no_aug')
dcnn.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

t_start = process_time()
history_no_aug = dcnn.fit(
    train_X, train_Y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
)
train_time_dcnn = process_time() - t_start
print(f"Training time (DCNN): {train_time_dcnn:.2f}s")


# ── 5. Train DCNN — With Data Augmentation ───────────────────────────────────
# Augmentation rationale:
#   - Rotation (±20°): digits can appear at slight angles in street photos
#   - Width/height shift (20%): handles off-centre digit positioning
#   - Horizontal flip: not used (flipped digits are not valid digits)
print("\n[DCNN + Aug] Training with data augmentation...")

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),            # ±20 degrees
    layers.RandomTranslation(0.2, 0.2),   # ±20% width and height shift
], name='augmentation')

dcnn_aug = models.Sequential([
    layers.Input(shape=input_shape),
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
], name='dcnn_with_aug')

dcnn_aug.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

t_start = process_time()
history_aug = dcnn_aug.fit(
    train_X, train_Y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1,
)
train_time_dcnn_aug = process_time() - t_start
print(f"Training time (DCNN + Aug): {train_time_dcnn_aug:.2f}s")


# ── 6. SVM Baseline ───────────────────────────────────────────────────────────
print("\n[SVM] Training baseline...")
train_vec = vectorise(train_X)
test_vec  = vectorise(test_X)

svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

t_start = process_time()
svm.fit(train_vec, train_Y)
train_time_svm = process_time() - t_start
print(f"Training time (SVM): {train_time_svm:.2f}s")


# ── 7. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(name, model, test_data, test_labels, train_time, model_type='dcnn'):
    """Evaluate a model on the test set and report F1, accuracy, and timing."""
    t_start = process_time()
    if model_type == 'dcnn':
        probs  = model.predict(test_data, verbose=0)
        y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = model.predict(test_data)
    inference_time = process_time() - t_start

    f1  = f1_score(test_labels, y_pred, average='macro')
    acc = np.mean(y_pred == test_labels)
    print(f"\n{name}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  F1 (macro):     {f1:.4f}")
    print(f"  Training time:  {train_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")
    print(classification_report(test_labels, y_pred))
    return y_pred, f1, acc, inference_time


y_pred_dcnn,     f1_dcnn,     acc_dcnn,     inf_dcnn     = evaluate('DCNN (no aug)',       dcnn,     test_X,   test_Y, train_time_dcnn,     'dcnn')
y_pred_dcnn_aug, f1_dcnn_aug, acc_dcnn_aug, inf_dcnn_aug = evaluate('DCNN (with aug)',     dcnn_aug, test_X,   test_Y, train_time_dcnn_aug, 'dcnn')
y_pred_svm,      f1_svm,      acc_svm,      inf_svm      = evaluate('SVM',                 svm,      test_vec, test_Y, train_time_svm,      'svm')


# ── 8. Summary Table ──────────────────────────────────────────────────────────
results = pd.DataFrame({
    'Model':          ['DCNN', 'DCNN + Augmentation', 'SVM'],
    'F1 Score':       [f1_dcnn, f1_dcnn_aug, f1_svm],
    'Accuracy':       [acc_dcnn, acc_dcnn_aug, acc_svm],
    'Train Time (s)': [train_time_dcnn, train_time_dcnn_aug, train_time_svm],
    'Infer Time (s)': [inf_dcnn, inf_dcnn_aug, inf_svm],
})
import pandas as pd
print("\nTable 1: Model Comparison")
print(results.to_string(index=False))


# ── 9. Training Curves ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for history, label, style in [
    (history_no_aug, 'DCNN',        '-'),
    (history_aug,    'DCNN + Aug',  '--'),
]:
    axes[0].plot(history.history['loss'],     linestyle=style, label=f'{label} Train')
    axes[0].plot(history.history['val_loss'], linestyle=style, alpha=0.6, label=f'{label} Val')
    axes[1].plot(history.history['accuracy'],     linestyle=style, label=f'{label} Train')
    axes[1].plot(history.history['val_accuracy'], linestyle=style, alpha=0.6, label=f'{label} Val')

for ax, title in zip(axes, ['Loss', 'Accuracy']):
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(fontsize=8)

fig.suptitle('Fig 1: Training Curves — DCNN vs DCNN with Augmentation')
plt.tight_layout()
plt.savefig('outputs/q3_training_curves.png', dpi=150)
plt.show()


# ── 10. Confusion Matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
class_labels = list(range(10))

for ax, (name, y_pred) in zip(axes, [
    ('DCNN',           y_pred_dcnn),
    ('DCNN + Aug',     y_pred_dcnn_aug),
    ('SVM',            y_pred_svm),
]):
    cm = confusion_matrix(test_Y, y_pred, labels=class_labels)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

fig.suptitle('Fig 2: Confusion Matrices on Test Set')
plt.tight_layout()
plt.savefig('outputs/q3_confusion_matrices.png', dpi=150)
plt.show()
