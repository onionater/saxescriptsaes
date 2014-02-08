#!/bin/bash
# Export environmental vars and set which queue to send the job to.
#PBS -V
#PBS -q {NODE}
#
# Name of job and redirection for stdout & stderr.
#PBS -N {NAME}
#PBS -o /mindhive/saxelab2/EIB/logs/{NAME}.log
#PBS -e /mindhive/saxelab2/EIB/logs/{NAME}.err

cd /mindhive/saxelab2/EIB
matlab -nosplash -nodisplay -r "startstuff"

cd /mindhive/saxelab/scripts/aesscripts/mvpaptb
matlab -nosplash -nodisplay -r "{CMD}"