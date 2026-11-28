"""
@Thang 

layer 3: Inference layer

Input:
    - load models(str) : .joblib file paths; cached in bundle
    - dataframe(df): returned by read_housing() 
                    (this is the unseen data, could include multiple rows), 
                    same as layer 2's input
Process:
    - load model: take the input of joblib file paths, return dict
    - predict: take the input of model and df -> output

Output: (of predict)
    - result (df): the predicted result by the predict.py
    - runtime metadata (dict): latency, ...

=======note
1. the loaded model (dict), key:val:
        - estimator : the model
        - log_radius : to calculate lower and upper (idc)
        - model: name of the model
        - parameters : dict of hyperparams: learning_rate, max_depth, ... 

"""