#!/bin/bash
# Export environmental vars and set which queue to send the job to.
#PBS -V
#PBS -q saxelab
#
# Name of job and redirection for stdout & stderr.
#PBS -N uniqueinformation1
#PBS -o /mindhive/saxelab2/EIB/logs/uniqueinformation1.log
#PBS -e /mindhive/saxelab2/EIB/logs/uniqueinformation1.err

cd /mindhive/saxelab/scripts/aesscripts/mvpaptb
matlab -nosplash -nodisplay -r "run_classification_EIB('EIB', 'EIB_main', makeIDs('EIB', [27]), '1to8')"
