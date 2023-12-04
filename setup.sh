pip install -r requirements.txt
cd data
sh getdata.sh
cd ..
torchrun --nnodes=1 --nproc_per_node=2 train.py --batch 6 --max_frames 40