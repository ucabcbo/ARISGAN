source /cs/student/msc/aisd/2022/cboehm/projects/sis2/venv/bin/activate.csh
# set current_date = `date "+%m%d-%H%M"`
set datetime = `date "+%m%d-%H%M"`

echo "Experiment: "$1
# echo "Batch Size: "$2
# echo "Shuffle: "$3

set experiment_root = $(grep -o '"experiment_root": *"[^"]*"' environment.json | awk -F '"' '{print $4}')

set outputfolder = "/cs/student/msc/aisd/2022/cboehm/projects/li1_output/"$datetime"_"$1"/"
mkdir $outputfolder
echo $outputfolder

# set outpath = "/cs/student/msc/aisd/2022/cboehm/projects/sis2/nohup/"$1"_"$current_date".txt"
# echo $outpath

# python /cs/student/msc/aisd/2022/cboehm/projects/sis2/train.py --m $1 --b $2 --shuffle $3 >& $outpath &
# python ./train.py --exp $1 --out $outputfolder >& $outputfolder"nohup.out" &
