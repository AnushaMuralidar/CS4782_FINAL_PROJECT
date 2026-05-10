import argparse
import os

import h5py
import numpy as np


def convert_mat_to_npz(input_file, output_dir, chunk_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(input_file, "r") as f:
        print("Available keys:", list(f.keys()))

        n_samples = f["features_with_offset"].shape[-1]
        print(f"Total samples: {n_samples}")

        chunk_idx = 0

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            print(f"Chunk {chunk_idx}: samples {start} to {end}")

            A = np.transpose(
                np.array(f["features_with_offset"][..., start:end])
            ).astype(np.float16)

            B = np.transpose(
                np.array(f["features_without_offset"][..., start:end])
            ).astype(np.float16)

            L = np.transpose(
                np.array(f["labels_gaussian_2d"][..., start:end])
            ).astype(np.float16)

            G = np.transpose(
                np.array(f["labels"][..., start:end])
            ).astype(np.float32)

            out_path = os.path.join(output_dir, f"chunk_{chunk_idx:02d}.npz")
            np.savez_compressed(out_path, A=A, B=B, L=L, G=G)

            size_mb = os.path.getsize(out_path) / (1024**2)
            print(f"Saved {out_path} ({size_mb:.1f} MB)")

            chunk_idx += 1

    print(f"Done. {chunk_idx} chunk files created in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert DLoc MATLAB .mat files into compressed .npz chunks.")

    parser.add_argument("--input",required=True,help="Path to input .mat file",)
    parser.add_argument("--output",required=True, help="Directory where .npz chunks will be saved",)
    parser.add_argument("--chunk_size",type=int,default=1000,help="Number of samples per output chunk",)

    args = parser.parse_args()

    convert_mat_to_npz(input_file=args.input,output_dir=args.output,chunk_size=args.chunk_size,)

if __name__ == "__main__":
    main()