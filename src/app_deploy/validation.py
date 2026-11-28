"""
@Thang

layer 2: Validate inputs
Input: 
    - dataframe (df) returned by read_housing() (this is the unseen data, could include multiple rows)
    - contract.json (output of layer 1)
Process: 
    -1 check schema: if not enough columns then break to 5
    -2 check type: all num_cols have numeric dtype, vice versa with cat_cols
    -3 check value range: required not NaN, some columns required value >0, cat_cols value belongs to allowed set
    -4 check others: value out of [min,max] in contract.json, lack optional cols, strange cols**
    -5 combine results of checks: return Valid_report object
Output: 
    - Issue object
    - Valid_report object **


---------notes    
** strange cols: could be either in these 3 cases:
    - columns that have been pruned during data engineering
    - the target column
    - columns that have not appeared in the test_cleaned before.

** returns a report, never raise error. 
** This layer does not write/update to the input df.   
"""

