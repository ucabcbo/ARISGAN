if ($#argv == 0) then
    echo "Error: Missing parameter. Please provide the required argument(s)."
    exit 1
endif

source /cs/student/msc/aisd/2022/cboehm/projects/sis2/venv/bin/activate.csh
set timestamp = `date "+%m%d-%H%M"`

echo "Experiment: "$1
echo "Timestamp: "$timestamp
echo "Restore: "$2

set experiment_root = `grep -o '"experiment_root":[[:space:]]*"[^"]*"' environment.json | awk -F '"' '{print $4}'`
set environment = `grep -o '"environment":[[:space:]]*"[^"]*"' environment.json | awk -F '"' '{print $4}'`

set outputfolder = $experiment_root$1"/nohup/"
set outputfile = $outputfolder$timestamp"_"$environment".out"
mkdir $outputfolder

echo "Experiment Root: "$experiment_root
echo "Outputfile: "$outputfile

if ($#argv < 2) then
    python ./train.py --exp $1 --timestamp $timestamp >& $outputfile &
else
    python ./train.py --exp $1 --restore $2 --timestamp $timestamp >& $outputfile &
endif
