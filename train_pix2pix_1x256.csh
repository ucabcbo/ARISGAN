source /cs/student/msc/aisd/2022/cboehm/projects/sis2/venv/bin/activate.csh
set current_date = `date "+%m%d-%H%M"`
set outpath = "/cs/student/msc/aisd/2022/cboehm/projects/sis2/nohup/"$current_date"_pix2pix_1x256.txt"
echo $outpath
python /cs/student/msc/aisd/2022/cboehm/projects/sis2/train.py --model pix2pix --batch_size 1 >& $outpath &
