source /cs/student/msc/aisd/2022/cboehm/projects/sis2/venv/bin/activate.csh
set current_date = `date "+%m%d-%H%M"`

echo "Model: "$1
echo "Batch Size: "$2
echo "Shuffle: "$3

set outpath = "/cs/student/msc/aisd/2022/cboehm/projects/sis2/nohup/"$current_date"_"$1"_"$2"x256.txt"
echo $outpath
python /cs/student/msc/aisd/2022/cboehm/projects/sis2/train.py --m $1 --b $2 --shuffle $3 >& $outpath &
