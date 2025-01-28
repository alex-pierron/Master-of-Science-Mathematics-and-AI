import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from tqdm import tqdm
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


nb_cpu = os.cpu_count()
# PARAMETERS TO CHANGE: DATA PATHS AND RANDOM SEEDS
random_seed = 42 #Controls the randomness when using scikit-learn models. Change this value to change the random seed.
random_starting_point = 42 # Controls the randomness for splitting the dataset between train and test. Change this value to change the random seed.
training_data_path = "Data\wine_train.csv"      # !!!!! Replace this value with your own path to the train set !!!!
testing_data_path = "Data\wine_test.csv"        # !!!!! Replace this value with your own path to the test set !!!!


data_train = pd.read_csv(training_data_path, index_col=0)
X = data_train.drop("target",axis=1)
Y = data_train.target
data_test = pd.read_csv(testing_data_path, index_col=0)


def train_and_eval(model, X, Y,test_size=0.15, random_state= random_starting_point,full_train = False):
    """
    Train and evaluate a scikit-learn regression model.

    Args:
        model: An instance of a scikit-learn regression model.
        X: Covariates (features).
        Y: Target variable to predict.
        test_size : the proportion of data in the validation set.
        random_state: Seed for the random number generator (default: 0).
        full_train: Flag indicating whether to train the model on the full dataset(default: False).

    Returns:
        If 'full_train' is False:
        X_result: Predicted values for the test set.
        Y_test: Actual target values for the test set.
        score: Evaluation scores, including the R^2 score and RMSE.
    """

    if not full_train:
        # Train test split
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = test_size,stratify = Y,
                                                            random_state = random_state if random_state !=0 else None)
        # Normalization
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled,X_test_scaled = scaler.transform(X_train),scaler.transform(X_test)
        X_train_scaled, X_test_scaled = pd.DataFrame(X_train_scaled,columns = X.columns), pd.DataFrame(X_test_scaled,columns= X.columns)

        
        model.fit(X_train_scaled,Y_train)
        X_result =  model.predict(X_test_scaled)
        # Evaluation : r2 score 
        score =  [r2_score(Y_test, X_result ),mean_squared_error(Y_test,X_result)**(1/2)]

        return X_result,Y_test,score

    else:
        scaler = StandardScaler()
        scaler.fit(X)
        X_used = pd.DataFrame(scaler.transform(X),columns = X.columns)
        model.fit(X_used,Y)
        
        return model
    pass

def multi_test( model, X, Y, test_size = 0.15, n = 25,
               random_start = random_starting_point, display_boxplot = True):
    """
    Train and evaluate a scikit-learn regression model multiple times with different random states.

    Args:
        model: An instance of a scikit-learn regression model.
        X: Covariates (features).
        Y: Target variable to predict.
        test_size : the proportion of data in the validation set.
        n: Number of models to generate and test (default: 25).
        random_start: Seed for the random number generator (default: 0).
        display_boxplot: Flag indicating whether to display a box plot of the evaluation results (default: True).

    Returns:
        all_scores: Evaluation scores for each model, including R^2 scores and RMSE.
    """
    all_scores = [train_and_eval(model,X,Y,test_size=test_size, random_state=k)[2] for k in range(random_start,random_start+n)]
    scores_r2 = [all_scores[i][0] for i in range(len(all_scores))]
    scores_rmse = [all_scores[i][1] for i in range(len(all_scores))]
    all_scores = [scores_r2,scores_rmse]

    if display_boxplot:
        plt.boxplot(all_scores,labels=["r2_score","rmse"])
        plt.title(f"Result after training {n} {model} with different random_state")
        plt.grid()
        plt.show()
        
    return all_scores


def grid_search(model, X, Y, hyperparameters,
                test_size = 0.15, n = 25, random_start = random_starting_point, display_boxplot = False,
                saving_path = None, scikit_model = True):
    """
    Perform grid search with cross-validation to find the best hyperparameters for a regression model.

    Args:
        model: A regression model object, either scikit-learn compatible or custom.
        X: Features (covariates) for regression.
        Y: Target variable to predict.
        hyperparameters: Dictionary of hyperparameters and their possible values for grid search.
        test_size : the proportion of data in the validation set.
        n: Number of estimators.
        random_start: Random seed for initialization.
        display_boxplot: Flag to display a box plot of the evaluation results.
        saving_path: Path to save the best model, or None if not saving.
        scikit_model: True if using a scikit-learn compatible model, False if it a PyTorch model.

    Returns:
        best_model: The best model with the optimal hyperparameters.
        best_params: The hyperparameters that resulted in the best performance.
        best_score: The best performance score (mean r2_score).
        all_results: A list of results for all hyperparameter combinations.
    """
    best_score = None
    best_params = None
    best_model = None
    all_results = []  

    if scikit_model:
        for params in tqdm(ParameterGrid(hyperparameters), desc="Grid Search Progress"):
            current_model = model.set_params(**params)

            scores = multi_test(current_model, X, Y, test_size = test_size,
                                 n = n, random_start = random_start, display_boxplot = False)
            current_r2_score = np.array(scores[0])

            if best_score is None or np.mean(current_r2_score) > best_score:
                best_score = np.mean(current_r2_score)
                best_params = params
                best_model = copy.copy(current_model)
                best_model_scores = scores

            all_results.append({
                "params": params,
                "scores": scores
            })

    else:
        for params in tqdm(ParameterGrid(hyperparameters), desc="Grid Search Progress"):
            current_model = model(**params)

            scores = multi_test(current_model, X, Y, test_size = test_size,
                                 n=n, random_start=random_start, display_boxplot=False)
            current_r2_score = np.array(scores[0])

            if best_score is None or np.mean(current_r2_score) > best_score:
                best_score = np.mean(current_r2_score)
                best_params = params
                best_model = current_model
                best_model_scores = scores

            all_results.append({
                "params": params,
                "scores": scores
            })
    print(current_model)
    if display_boxplot:
        plt.boxplot(best_model_scores, labels=["r2_score", "rmse"])
        plt.title(f"Best Model: {best_model} with Hyperparameters: {best_params}")
        plt.grid()
        #Saving the plot
        if saving_path is not None:
            output_folder = os.path.dirname(saving_path)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            boxplot_filename = os.path.join(output_folder, "boxplot.png")
            plt.savefig(boxplot_filename)

        plt.show()

    print("Best hyperparameters:", best_params)
    print("Best score (r2_score):", best_score)

    if saving_path is not None:
        output_folder = os.path.dirname(saving_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        joblib.dump(best_model, saving_path)

    return best_model, best_params, best_score, all_results
    

def submission(model,X_test=data_test,X_train=X,Y_train=Y,name_file = "Submissions/soumission.csv",pretrained=False):
    """
    Generate a submission file using the desired model.

    Args:
        model: A trained regression model.
        X_test: Test dataset features (default: data_test).
        X_train: Training dataset features.
        Y_train: Training dataset target variable.
        name_file: Name of the output submission file (default: "Submissions/submission.csv").
        pretrained: Whether or not the model is already trained.

    Returns:
        submission_data: A DataFrame containing the wine_ID and predicted target.
    """

    scaler = StandardScaler()
    scaler.fit(X_train)
    index_list = X_test.index
    if not pretrained:
        model = train_and_eval(model,X_train,Y_train,full_train=True)
    X_test = pd.DataFrame(scaler.transform(X_test),columns = X_test.columns)
    prediction = model.predict(X_test)
    submission_data = pd.DataFrame({'wine_ID': index_list, 'target': prediction})

    output_folder = os.path.dirname(name_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    submission_data.to_csv(name_file, index=False)
    return submission_data


def mean_submission(liste_soumission,path_result = "Soumissions/mean_submission.csv"):
    """
    Generate a mean submission file from a list of submission files.

    Args:
        liste_soumission: List of submission file paths.
        path_result: Path for the mean submission file (default: "Submissions/mean_submission.csv").

    Returns:
        None
    """
    dataframes = []
    for file in liste_soumission:
        df = pd.read_csv(file)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, axis=1)
    combined_df = combined_df.T.drop_duplicates().T
    columns_average = ['target']
    row_averages = combined_df[columns_average].mean(axis=1)

    if 'target' in combined_df.columns:
        combined_df = combined_df.drop(columns=['target'])

    combined_df['target'] = row_averages
    combined_df["wine_ID"] = combined_df['wine_ID'].astype(int)
    output_folder = os.path.dirname(path_result)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    combined_df.to_csv(path_result, index=False)

    pass

def save_model(model,path_to_save="Archives_Model/Default_best_model.pkl"):
    output_folder = os.path.dirname(path_to_save)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    joblib.dump(model, path_to_save)