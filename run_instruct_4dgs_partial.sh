#!/bin/bash
set -euo pipefail

# ===================================================================
# ./run_instruct_4dgs_partial.sh -d [dataset] -s [scene_name] -p [editing_prompts] -m [mask_prompt] -g [guidance_scale] -i [image_guidance_scale] -o [inpainting_dir]
# ===================================================================

# Default values
DATASET=""
SCENE_NAME=""
EDITING_PROMPTS=""
MASK_PROMPT=""
GUIDANCE_SCALE=7.5
IMAGE_GUIDANCE_SCALE=1.5
INPAINTING_DIR=""

while getopts "d:s:p:m:g:i:o:" opt; do
  case $opt in
    d) DATASET="$OPTARG" ;;
    s) SCENE_NAME="$OPTARG" ;;
    p) EDITING_PROMPTS="$OPTARG" ;;
    m) MASK_PROMPT="$OPTARG" ;;
    g) GUIDANCE_SCALE="$OPTARG" ;;
    i) IMAGE_GUIDANCE_SCALE="$OPTARG" ;;
    o) INPAINTING_DIR="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$DATASET" ] || [ -z "$SCENE_NAME" ] || [ -z "$EDITING_PROMPTS" ]; then
    echo "Usage: $0 -d <dataset> -s <scene_name> -p <comma_separated_prompts> [-m <mask_prompt>] [-g <guidance_scale>] [-i <image_guidance_scale>] [-o <inpainting_dir>]"
    exit 1
fi

# Convert comma-separated prompts to array
IFS=',' read -r -a PROMPT_ARRAY <<< "$EDITING_PROMPTS"

echo "------------------------------------------"
echo "  - dataset: ${DATASET}"
echo "  - scene: ${SCENE_NAME}"
echo "  - editing_prompts: ${PROMPT_ARRAY[*]}"
echo "  - mask_prompt: \"${MASK_PROMPT}\""
echo "------------------------------------------"
echo ""

# echo "[1/] Collect time0 images..."
# python time0_collect.py --dataset ${DATASET} --scene_name ${SCENE_NAME}
# echo ""

# echo "[2/] Inpainting time0 images..."
# for i in "${!PROMPT_ARRAY[@]}"; do
#     prompt="${PROMPT_ARRAY[$i]}"
#     index=$((i + 1))
#     echo "  - Inpainting with prompt: ${prompt}"
#     python ./ip2p_models/multiview_edit.py \
#         --dataset "${DATASET}" \
#         --scene_name "${SCENE_NAME}" \
#         --prompt "${prompt}" \
#         --out_folder "prompt_${index}" \
#         --resize 1024 \
#         --steps 20 \
#         --guidance_scale ${GUIDANCE_SCALE} \
#         --image_guidance_scale ${IMAGE_GUIDANCE_SCALE}
# done
# echo "✅ Completed time0 image editing."
# echo ""

# echo "[3/] Generate masks for time0 images..."
# python grounded_sam2_mask_gen.py \
#     --data_dir "data/${DATASET}/${SCENE_NAME}/time0" \
#     --output_dir "data/${DATASET}/${SCENE_NAME}/object_mask" \
#     --prompt "${MASK_PROMPT}"
# echo "✅ Completed mask generation."
# echo ""

# echo "[4/] Edit time0 images..."

# # Build img_dirs list
# IMG_DIRS=("data/${DATASET}/${SCENE_NAME}/time0")
# for i in "${!PROMPT_ARRAY[@]}"; do
#     index=$((i + 1))
#     IMG_DIRS+=("data/${DATASET}/${SCENE_NAME}/inpainting/prompt_${index}")
# done

# python partial_edit.py \
#     --img_dirs "${IMG_DIRS[@]}" \
#     --mask_dir "data/${DATASET}/${SCENE_NAME}/object_mask" \
#     --save_dir "data/${DATASET}/${SCENE_NAME}/partial_edited"
# echo "✅ Completed time0 image editing."
# echo ""

echo "[5/] 3D editing"
python edit_3d.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --ply_path "./output/${DATASET}/${SCENE_NAME}/point_cloud/iteration_14000/point_cloud.ply" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "./output/${DATASET}/${SCENE_NAME}" \
    --dataset "${DATASET}" \
    --scene "${SCENE_NAME}" \
    --prompt "${EDITING_PROMPTS%%,*}" \
    --edited_images_path "data/${DATASET}/${SCENE_NAME}/partial_edited"
echo "✅ Completed 3d editing."
echo ""

# echo "[6/] 3D Segmentation, adding labels"
# PYTHONPATH=. python ObjectGS/ply_preprocessing.py \
#     --dataset_path data/dynerf/cook_spinach \
#     --input_ply_path "./output/${DATASET}/${SCENE_NAME}/point_cloud_3dedit/${EDITING_PROMPTS%%,*}/iteration_1000/point_cloud.ply" \
#     --algorithm majority \
#     --invert \
#     --add_label_only
# echo "✅ Completed 3d editing."
# echo ""

# echo "[7/] Score refinement"
# python refine_sds.py \
#     --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
#     --ply_path "./output/${DATASET}/${SCENE_NAME}/point_cloud_3dedit/${EDITING_PROMPTS%%,*}/iteration_1000/point_cloud_with_label.ply" \
#     -s "./data/${DATASET}/${SCENE_NAME}" \
#     --model_path "./output/${DATASET}/${SCENE_NAME}" \
#     --prompt "${EDITING_PROMPTS%%,*}" \
#     --guidance_scale ${GUIDANCE_SCALE} \
#     --image_guidance_scale ${IMAGE_GUIDANCE_SCALE}
# echo "✅ Completed score refinement."
# echo ""

echo "🎉 All pipeline steps have been executed."
