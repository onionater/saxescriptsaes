
#!/bin/bash
# Export environmental vars and set which queue to send the job to.
#PBS -V
#PBS -q saxelab
#
# Name of job and redirection for stdout & stderr.
#PBS -N SAX_EIB_11_permutation_2
#PBS -o /mindhive/saxelab2/EIB/logs/SAX_EIB_11_permutation_2.log
#PBS -e /mindhive/saxelab2/EIB/logs/SAX_EIB_11_permutation_2.err

cd /mindhive/saxelab2/EIB
matlab -nosplash -nodisplay -r "startstuff"

cd /mindhive/saxelab/scripts/aesscripts/mvpaptb
matlab -nosplash -nodisplay -r "run_classification_EIB('EIB_main', {'SAX_EIB_11'}, '1to8', {'folderprefix', 'permutation_', 'scramble', 1, 'suffix', 2})"