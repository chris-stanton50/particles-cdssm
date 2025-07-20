cd ./tests/linear_sdes/
source run_linear_sde_tests.sh
cd ../filtering 
source run_filtering_tests.sh
cd ../..
git add .
git commit -m "new dev commits"
git push
git pull