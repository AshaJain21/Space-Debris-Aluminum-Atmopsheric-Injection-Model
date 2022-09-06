#!/bin/bash
iteration=$1
/Applications/MATLAB_R2021b.app/bin//matlab -nojvm -nosplash -nodisplay -nodesktop -r "equalAreaBinning($iteration); exit"
