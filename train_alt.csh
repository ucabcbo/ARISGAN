source /cs/student/msc/aisd/2022/cboehm/projects/sis2/venv/bin/activate.csh
set current_date = `date "+%m%d-%H%M"`

echo "Experiment: "$1
# echo "Batch Size: "$2
# echo "Shuffle: "$3

set outpath = "/cs/student/msc/aisd/2022/cboehm/projects/sis2/nohup/"$1"_"$current_date"_alt.txt"
echo $outpath

# python /cs/student/msc/aisd/2022/cboehm/projects/sis2/train.py --m $1 --b $2 --shuffle $3 >& $outpath &
python ./train_alt.py --exp $1 >& $outpath &
