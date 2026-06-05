import cv2
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, List


def parse_args():
    parser = argparse.ArgumentParser(description="Blend inpainted patches into original images using masks.")
    parser.add_argument("--img_dirs", type=str, nargs='+', required=True, help="Paths to images directories")
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


def get_file_n_tuples(
    img_dirs: List[Path],
    mask_dir: Path,
    ext: str = "jpg",
    mask_ext: str = "png"
) -> Iterator[Tuple[List[Path], Path, int]]:
    """
    Discovers and aligns files from multiple image directories and a mask directory.
    Returns a sorted generator of (list_of_image_paths, mask_path, index).
    """
    # Create mappings of index -> file path for each directory
    img_files_maps = []
    for d in img_dirs:
        img_files_maps.append({get_index(p): p for p in d.glob(f"*.{ext}") if get_index(p) != -1})
        
    mask_files = {get_index(p): p for p in mask_dir.glob(f"*.{mask_ext}") if get_index(p) != -1}
    
    # Find common indices
    if not img_files_maps:
        return
        
    all_indices = set(mask_files.keys())
    for m in img_files_maps:
        all_indices &= set(m.keys())
        
    for idx in sorted(all_indices):
        paths = [m[idx] for m in img_files_maps]
        yield paths, mask_files[idx], idx


def main():
    args = parse_args()

    # Convert to Path objects
    img_paths = [Path(d) for d in args.img_dirs]
    msk_path = Path(args.mask_dir)
    sav_path = Path(args.save_dir)

    # Check if all input paths exist
    error_found = False
    for p in img_paths + [msk_path]:
        if not p.exists() or not p.is_dir():
            print(f"Error: Directory not found: {p}")
            error_found = True
    
    if error_found:
        sys.exit(1)

    # Ensure save directory exists
    sav_path.mkdir(parents=True, exist_ok=True)

    # Process images via generator
    tuples = get_file_n_tuples(img_paths, msk_path, args.ext, args.mask_ext)

    count = 0
    for img_ps, mask_p, idx in tuples:
        # Load images
        images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in img_ps]
        mask_img = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)

        if any(img is None for img in images) or mask_img is None:
            print(f"Warning: Could not read files for index {idx}. Skipping.")
            continue

        # Composite logic
        final_img = np.zeros_like(images[0])
        for i, img in enumerate(images):
            final_img[mask_img == i] = img[mask_img == i]

        # Saving
        target_path = sav_path / img_ps[0].name
        
        cv2.imwrite(str(target_path), final_img)
        print(f"Saved: {target_path}")
        count += 1

    if count == 0:
        print("No matching file triplets found. Check your file naming (trailing _index) and extensions.")
    else:
        print(f"Done. Processed {count} images.")


if __name__ == "__main__":
    main()