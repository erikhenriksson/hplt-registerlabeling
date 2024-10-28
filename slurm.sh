#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=project_2011109
#SBATCH --mail-user=pytorchregisterlabeling@gmail.com 
#SBATCH --mail-type=ALL

if [[ -z "$SLURM_JOB_ID" ]]; then
  PARTITION="gpumedium"
  TIME="36:00:00"
  NUM_GPUS=4
  MEM=8
  if [[ $1 == "1h" ]]; then
    TIME="1:00:00"
    shift 
  elif [[ $1 == "2h" ]]; then
    TIME="2:00:00"
    shift 
  elif [[ $1 == "6h" ]]; then
    TIME="6:00:00"
    shift 
  elif [[ $1 == "12h" ]]; then
    TIME="12:00:00"
    shift 
  elif [[ $1 == "30m" ]]; then
    TIME="0:30:00"
    shift 
  fi

  GRES_GPU="gpu:a100:$NUM_GPUS"
  DYNAMIC_JOBNAME="$1"
  shift  
  JOB_SUBMISSION_OUTPUT=$(sbatch --job-name="$DYNAMIC_JOBNAME" --time="$TIME" --gres="$GRES_GPU" --mem="$MEM"G --partition="$PARTITION" -o "logs/${DYNAMIC_JOBNAME}-%j.log" "$0" "$@")
  echo "Submission output: $JOB_SUBMISSION_OUTPUT"
  JOB_ID=$(echo "$JOB_SUBMISSION_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
  LOG_FILE="logs/${DYNAMIC_JOBNAME}-${JOB_ID}.log"
  touch $LOG_FILE
  echo "tail -f $LOG_FILE"
  tail -f "$LOG_FILE"
  exit $?
else
  source venv/bin/activate
  srun python3 "$@"
fi
