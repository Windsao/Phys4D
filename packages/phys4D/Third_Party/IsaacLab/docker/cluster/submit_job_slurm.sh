






cat <<EOT > job.sh












bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh
