
#!/bin/bash
# Export environmental vars and set which queue to send the job to.
#PBS -V
#PBS -q saxelab
#
# Name of job and redirection for stdout & stderr.
#PBS -N SL_3_SAX_EIB_10
#PBS -o /mindhive/saxelab2/EIB/logs/SL_3_SAX_EIB_10.log
#PBS -e /mindhive/saxelab2/EIB/logs/SL_3_SAX_EIB_10.err

cd /mindhive/saxelab2/EIB
matlab -nosplash -nodisplay -r "startstuff"

cd /mindhive/saxelab/scripts/aesscripts/mvpaptb
matlab -nosplash -nodisplay -r "run_searchlight('EIB', 'EIB_main', {'SAX_EIB_10'}, '1to8', {'discsubset', 'negVSposONE', 'xvalsubset', 'crossrunsONEselector', 'voxsubset', 3, 'voxelrange', []})"