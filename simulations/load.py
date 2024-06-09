"""
Small script which reads the pickle files created in the optimization scripts
and prints their best loss and the final parameters

@author: Lynn Schmittwilken, June 2024
"""

import pickle

results_file = "results_multi_5.pickle"


if __name__ == "__main__":
    # Load data from pickle:
    with open(results_file, 'rb') as handle:
        data_pickle = pickle.load(handle)
    
    best_params = data_pickle["best_params_auto"]
    best_loss = data_pickle["best_loss_auto"]
    model_params = data_pickle["params_dict"]
    
    try:
        print("Gain:", data_pickle["model_params"]["gain"])
        print("Same noise instances:", data_pickle["model_params"]["sameNoise"])
    except:
        pass

    print("Best loaded loss:", best_loss)
    print("Best params", best_params)


