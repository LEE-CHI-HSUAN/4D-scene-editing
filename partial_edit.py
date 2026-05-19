import cv2
import argparse
import sys
from pathlib import Path
from typing import Iterator, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Blend inpainted patches into original images using masks.")
    parser.add_argument("--org_dir", type=str, required=True, help="Path to original images directory")
    parser.add_argument("--inpaint_dir", type=str, required=True, help="Path to inpainted images directory")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to mask images directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save the blended results")
    parser.add_argument("--ext", type=str, default="jpg", help="Extension for images (default: jpg)")
    parser.add_argument("--mask_ext", type=str, default="png", help="Extension for masks (default: png)")
    return parser.parse_args()

def get_index(path_str: Path) -> int:
    """
    Extracts the trailing integer index from a filename.
    Assumes format: name_index.ext (e.g., frame_01.jpg)
    """
    try:
        return int(path_str.stem.split('_')[-1])
    except (ValueError, IndexError):
        # Handle cases where the filename might not follow the expected format
        return -1


def get_file_triplets(
    org_dir: Path, 
    inpaint_dir: Path, 
    mask_dir: Path, 
    ext: str = "jpg", 
    mask_ext: str = "png"
) -> Iterator[Tuple[Path, Path, Path, int]]:
    """
    Discovers and aligns files from three directories.
    Returns a sorted generator of (original, inpaint, mask, index).
    """
    # Create mappings of index -> file path for fast O(1) lookup
    inpaint_files = {get_index(p): p for p in inpaint_dir.glob(f"*.{ext}") if get_index(p) != -1}
    mask_files = {get_index(p): p for p in mask_dir.glob(f"*.{mask_ext}") if get_index(p) != -1}
    
    # Get original files and sort them by index
    org_files = sorted(
        [p for p in org_dir.glob(f"*.{ext}") if get_index(p) != -1], 
        key=get_index
    )

    for org_path in org_files:
        idx = get_index(org_path)
        if idx in inpaint_files and idx in mask_files:
            yield org_path, inpaint_files[idx], mask_files[idx], idx


def main():
    args = parse_args()

    # Convert to Path objects
    org_path = Path(args.org_dir)
    inp_path = Path(args.inpaint_dir)
    msk_path = Path(args.mask_dir)
    sav_path = Path(args.save_dir)

    # Check if all input paths exist
    error_found = False
    for p in [org_path, inp_path, msk_path]:
        if not p.exists() or not p.is_dir():
            print(f"Error: Directory not found: {p}")
            error_found = True
    
    if error_found:
        sys.exit(1)

    # Ensure save directory exists
    sav_path.mkdir(parents=True, exist_ok=True)

    # Process images via generator
    triplets = get_file_triplets(org_path, inp_path, msk_path, args.ext, args.mask_ext)

    count = 0
    for org_p, inp_p, mask_p, idx in triplets:
        # Load single frames
        org_img = cv2.imread(str(org_p), cv2.IMREAD_COLOR)
        inp_img = cv2.imread(str(inp_p), cv2.IMREAD_COLOR)
        mask_img = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)

        if org_img is None or inp_img is None or mask_img is None:
            print(f"Warning: Could not read files for index {idx}. Skipping.")
            continue

        # Mask Logic: Ensure mask is boolean for faster indexing
        mask_bool = mask_img > 0

        # In-place operation: Modify the original image directly
        # If mask is 2D and image is 3D, numpy handles broadcasting
        org_img[mask_bool] = inp_img[mask_bool]

        # Saving
        target_path = sav_path / inp_p.name
        
        cv2.imwrite(str(target_path), org_img)
        print(f"Saved: {target_path}")
        count += 1

    if count == 0:
        print("No matching file triplets found. Check your file naming (trailing _index) and extensions.")
    else:
        print(f"Done. Processed {count} images.")


if __name__ == "__main__":
    main()