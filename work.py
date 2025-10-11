import os
import pandas as pd
import numpy as np
from dollarpy import Recognizer, Template, Point
import textdollar
import tkinter as tk
from tkinter import filedialog


def perform_pca(df):
    df = df.select_dtypes(include=[np.number])
    if 'time' in df.columns:
        df = df.drop(columns=['time'])

    df = df.loc[:, df.std() > 0]

    if df.shape[1] < 2:
        raise ValueError("Not enough valid numeric columns for PCA.")

    X = (df - df.mean()) / df.std()

    cov = np.cov(X.T)

    eig_vals, eig_vecs = np.linalg.eig(cov)

    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]

    W = eig_vecs[:, :2]

    projected = np.dot(X, W)
    x, y = projected[:, 0], projected[:, 1]

    return [Point(float(px), float(py)) for px, py in zip(x, y)]


def classify_csv(csv_path):
    print(f"\nProcessing test file: {csv_path}")

    df = pd.read_csv(csv_path)
    test_points = perform_pca(df)

    templates = [
        getattr(textdollar, name)
        for name in dir(textdollar)
        if isinstance(getattr(textdollar, name), Template)
    ]

    if not templates:
        print("No templates found in textdollar.py! Please generate them first.")
        return

    recognizer = Recognizer(templates)

    result = recognizer.recognize(test_points)
    label, confidence = result[0], result[1]

    print(f"\nClassified Stroke: {label}")
    print(f"Confidence: {confidence:.3f}")

    if confidence < 0.15:
        print("Low confidence")


if __name__ == "__main__":
    print("\nSwimming Stroke Classifier")

    root = tk.Tk()
    root.withdraw()  

    csv_path = filedialog.askopenfilename(
        title="Select a CSV file to classify",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not csv_path:
        print("No file selected. Exiting.")
    else:
        classify_csv(csv_path)
