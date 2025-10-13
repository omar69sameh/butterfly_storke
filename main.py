import os
import pandas as pd
import numpy as np
from dollarpy import Point

def perform_pca(df):
    df = df.select_dtypes(include=[np.number])
    if 'time' in df.columns:
        df = df.drop(columns=['time'])

    df = df.loc[:, df.std() > 0]

    if df.shape[1] < 2:
        return None, None

    X = (df - df.mean()) / df.std()

    cov = np.cov(X.T)

    eig_vals, eig_vecs = np.linalg.eig(cov)

    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]

    W = eig_vecs[:, :2]

    projected = np.dot(X, W)
    x, y = projected[:, 0], projected[:, 1]
    return x, y


def process_csvs_in_folder(root_folder):
    output_file = "textdollar.py"
    f = open(output_file, "w", encoding="utf-8")
    f.write("from dollarpy import Recognizer, Template, Point\n\n")

    template_lines = []
    processed_count = 0

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                filepath = os.path.join(subdir, file)
                try:
                    df = pd.read_csv(filepath)
                    x, y = perform_pca(df)
                    if x is None or y is None:
                        print(f"Skipped {file}: not enough valid columns.")
                        continue

                    label = os.path.basename(subdir)
                    var_name = os.path.splitext(file)[0].replace("-", "_")

                    points_str = ", ".join(
                        [f"Point({float(px):.4f}, {float(py):.4f})" for px, py in zip(x, y)]
                    )
                    template_lines.append(f"{var_name} = Template('{label}', [{points_str}])")
                    processed_count += 1
                    print(f"Processed: {file} -> label '{label}'")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    if processed_count > 0:
        f.write("\n".join(template_lines))
        f.write(f"\n\nrecognizer = Recognizer([{', '.join([t.split('=')[0].strip() for t in template_lines])}])\n")
        print(f"\nDone! textdollar.py saved successfully with {processed_count} templates.")
    else:
        print("\nNo valid CSV files found or processed!")

    f.close()


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "C:\\Users\\khale\\OneDrive\\Documents\\GitHub\\butterfly_storke\\data")
    print(f"\nSearching recursively in: {data_path}\n")
    process_csvs_in_folder(data_path)
