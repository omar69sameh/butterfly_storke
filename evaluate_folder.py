import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from dollarpy import Recognizer, Template, Point
import textdollar


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


def classify_csv(csv_path, recognizer):
    df = pd.read_csv(csv_path)
    test_points = perform_pca(df)
    label, confidence = recognizer.recognize(test_points)
    return label, confidence


def evaluate_folder(root_folder):
    templates = [
        getattr(textdollar, name)
        for name in dir(textdollar)
        if isinstance(getattr(textdollar, name), Template)
    ]

    if not templates:
        print("No templates found in textdollar.py. Run main.py first!")
        return

    recognizer = Recognizer(templates)

    results = {}
    total_correct = total_files = 0

    print(f"\nEvaluating all CSV files in: {root_folder}\n")

    for stroke_name in os.listdir(root_folder):
        stroke_path = os.path.join(root_folder, stroke_name)
        if not os.path.isdir(stroke_path):
            continue

        stroke_correct = stroke_total = 0

        for file in os.listdir(stroke_path):
            if not file.endswith(".csv"):
                continue

            csv_path = os.path.join(stroke_path, file)
            try:
                predicted, confidence = classify_csv(csv_path, recognizer)
                stroke_total += 1
                total_files += 1

                correct = (predicted.lower() == stroke_name.lower())
                if correct:
                    stroke_correct += 1
                    total_correct += 1

                print(f"File: {file}")
                print(f" → True: {stroke_name} | Predicted: {predicted} | "
                      f"Confidence: {confidence:.3f} | {'True' if correct else 'False'}\n")

            except Exception as e:
                print(f"Error processing {file}: {e}")

        if stroke_total > 0:
            acc = (stroke_correct / stroke_total) * 100
            results[stroke_name] = (stroke_correct, stroke_total, acc)

    print("\n📊 Evaluation Summary:")
    for stroke, (correct, total, acc) in results.items():
        print(f" - {stroke}: {correct}/{total} correct ({acc:.2f}%)")

    if total_files > 0:
        overall_acc = (total_correct / total_files) * 100
        print(f"\nOverall Accuracy: {total_correct}/{total_files} ({overall_acc:.2f}%)")
    else:
        print("\nNo CSV files found to evaluate.")


if __name__ == "__main__":
    print("\nSwimming Stroke Classifier - Evaluation Mode")

    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(
        title="Select the folder containing untrained stroke data"
    )

    if not folder_path:
        print("No folder selected. Exiting.")
    else:
        evaluate_folder(folder_path)
