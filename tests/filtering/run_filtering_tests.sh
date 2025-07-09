# Run filtering tests for each of the models:

# Make folder to store the results:
mkdir results

python filtering_test.py 80 ou
python filtering_test.py 80 mv_ou
python filtering_test.py 80 iou
# python filtering_test.py 10 i2ou

python filtering_test.py 80 bm
python filtering_test.py 80 mv_bm
python filtering_test.py 80 ibm
# python filtering_test.py 10 i2bm  

# Check whether the tests all passed:
python filtering_test_results.py 80

# Delete the results
rm -rf results