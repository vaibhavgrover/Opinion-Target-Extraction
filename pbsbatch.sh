#!/bin/sh
### Set the job name
#PBS -N elman
### Set the project name, your department dc by default
#PBS -P maths
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M mt5110581@maths.iitd.ac.in
####
# PBS -l select=1:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=12:00:00
#### Get environment variables from submitting shell
#PBS -V
#PBS -l software=
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
# cd $PBS_O_WORKDIR
cd /home/maths/integrated/mt5110581/NLP/Elman
#job 
./cv-batch.sh restaurant Senna 50 50 elman
time -p mpirun -n {n*m} executable
#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE