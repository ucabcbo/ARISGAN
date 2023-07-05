source /cs/student/msc/aisd/2022/cboehm/projects/sis2/venv/bin/activate.csh
set current_date = `date "+%m%d-%H%M"`

echo "Model: "$1
echo "Batch Size: "$2

set outpath = "/cs/student/msc/aisd/2022/cboehm/projects/sis2/nohup/"$current_date"_"$1"_"$2"x256.txt"
echo $outpath
#TODO: turned off shuffling for performance temporarily
python /cs/student/msc/aisd/2022/cboehm/projects/sis2/train.py --model $1 --batch_size $2 --shuffle n >& $outpath &
