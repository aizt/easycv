echo $JOB_INPUT
echo $JOB_OUTPUT
python main-all-final.py --n_class 0 --gpu 0 --seed 4396 -b 64 -p 5 --epochs 36 --decay_epoch 12 \
	--min_scale 0.2 --weight-decay 2e-4 --workers 0 --log_name ad_all_pretrained_new --data ${JOB_INPUT}
