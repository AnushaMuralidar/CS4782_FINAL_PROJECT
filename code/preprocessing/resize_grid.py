import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F


def resize_npz_folder(input_folder, output_folder, target_h=101, target_w=161):
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".npz")]

    print(f"Found {len(files)} files")

    for f in files:
        data = np.load(os.path.join(input_folder, f))

        A = data["A"]   # [N, C, H, W]
        B = data["B"]   # [N, C, H, W]
        L = data["L"]   # [N, H, W] or [N,1,H,W]
        G = data["G"]   # [N, 2] (do NOT resize)

        A_resized = []
        B_resized = []
        L_resized = []

        for i in range(A.shape[0]):

            # A
            a = torch.tensor(A[i], dtype=torch.float32).unsqueeze(0)
            a = F.interpolate(a, size=(target_h, target_w),
                              mode="bilinear", align_corners=True)
            A_resized.append(a.squeeze(0).numpy())

            # B
            b = torch.tensor(B[i], dtype=torch.float32).unsqueeze(0)
            b = F.interpolate(b, size=(target_h, target_w),
                              mode="bilinear", align_corners=True)
            B_resized.append(b.squeeze(0).numpy())

            # L
            l = torch.tensor(L[i], dtype=torch.float32)

            if l.ndim == 2:
                l = l.unsqueeze(0)

            l = l.unsqueeze(0)  # [1,1,H,W]
            l = F.interpolate(l, size=(target_h, target_w),
                              mode="bilinear", align_corners=True)
            L_resized.append(l.squeeze().numpy())

        A_resized = np.array(A_resized)
        B_resized = np.array(B_resized)
        L_resized = np.array(L_resized)

        save_path = os.path.join(output_folder, f)

        np.savez_compressed(
            save_path,
            A=A_resized,
            B=B_resized,
            L=L_resized,
            G=G   # preserve coordinates
        )

        print(f"Saved: {f}")

    print("All files resized and saved.")


def main():
    parser = argparse.ArgumentParser(description="Resize NPZ heatmaps to a common grid.")

    parser.add_argument("--input", required=True, help="Input folder with .npz files")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--height", type=int, default=101)
    parser.add_argument("--width", type=int, default=161)

    args = parser.parse_args()

    resize_npz_folder(
        input_folder=args.input,
        output_folder=args.output,
        target_h=args.height,
        target_w=args.width,
    )


if __name__ == "__main__":
    main()