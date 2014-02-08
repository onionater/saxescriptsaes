#!/bin/bash
# Export environmental vars and set which queue to send the job to.
#PBS -V
#PBS -q saxelab
#
# Name of job and redirection for stdout & stderr.
#PBS -N searchlight_EIB_2_allsubj
#PBS -o /mindhive/saxelab2/EIB/logs/EIBsearchlight_01.log
#PBS -e /mindhive/saxelab2/EIB/logs/EIBsearchlight_01.err

cd /mindhive/saxelab/scripts/aesscripts/mvpaptb
matlab -nosplash -nodisplay -r "run_searchlight(makeIDs('EIB', [1]))"