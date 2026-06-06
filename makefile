run:
	bash run_instruct_4dgs_partial.sh \
		-d dynerf -s cook_spinach -p "Make it look like a fauvism painting" \
		-m "guy." \
		-g 10.5 -i 1.2

sds:
	python render_edited4d.py     --configs ./arguments/dynerf/cook_spinach.py     --ply_path "output/dynerf/cook_spinach/point_cloud_refine/Make it look like a fauvism painting/iteration_800/point_cloud.ply"     -s ./data/dynerf/cook_spinach     --model_path ./output/dynerf/cook_spinach