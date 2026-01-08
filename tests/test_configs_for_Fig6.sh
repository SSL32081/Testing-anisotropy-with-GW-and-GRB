#!/usr/bin/env zsh

source /cvmfs/software.igwn.org/conda/etc/profile.d/conda.sh
conda activate gw-school-2025

DATA_DIR='../data'
PLOT_DIR='../published_results'

for nths in 180 500; do
    for lmax in 26 128; do
        echo "Onto nthetas = ${nths} and lmax = ${lmax}"
        cd ${DATA_DIR}
        python compute_synthetic_data_gamma_fit.py --ntheta ${nths} --lmax ${lmax} 
        python compute_synthetic_data_gamma_fit.py --ntheta ${nths} --lmax ${lmax} --nowindow
        cd ${PLOT_DIR}
        python plot_Fig6_autocorrelation.py --ntheta ${nths} --lmax ${lmax} & 
        python plot_Fig6_autocorrelation.py --ntheta ${nths} --lmax ${lmax} --nowindow & 
        python plot_Fig6_autocorrelation.py --ntheta ${nths} --lmax ${lmax} --gammafit &
        python plot_Fig6_autocorrelation.py --ntheta ${nths} --lmax ${lmax} --nowindow --gammafit &
    done
done

echo "Waiting all jobs to complete"
wait

echo "All done, exiting..."
