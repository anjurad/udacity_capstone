import argparse
import joblib
import os
import numpy as np
import pandas as pd

# sklearn
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV

# azureml
from azureml.core.run import Run
from azureml.core import Workspace


def main():
    # Get script arguments
    parser = argparse.ArgumentParser()

    # Input dataset
    parser.add_argument("--input-data", type=str,
                        dest='input_data', help='training dataset')

    # Hyperparameters
    parser.add_argument('--C', type=float, default=1,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")

    parser.add_argument('--max_iter', type=int, default=100,
                        help="Maximum number of iterations to converge")

    parser.add_argument('--min_features', type=int, default=10,
                        help="RFE - Min features to select")

    args = parser.parse_args()

    print("start experiment")

    run = Run.get_context()

    run.log("arg_C", np.float(args.C))
    run.log("arg_max_iter", np.int(args.max_iter))
    run.log("arg_min_features", np.int(args.min_features))

    # load the dataset
    print("loading data")

    # Get the training data from the estimator input
    dataset = run.input_datasets['training_data'].to_pandas_dataframe()

    # change objects to category to impute
    for col in dataset.select_dtypes(object):
        dataset[col] = dataset[col].astype('category')

    print("data loaded")

    X = dataset.drop(columns=['target'], axis=1)
    y = np.array(dataset['target'])

    print("start test_train_split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print("ended test_train_split")

    print("start model")

    # Setting up the sklean pipeline
    # imputer
    imp = IterativeImputer(max_iter=10, random_state=0)

    # RFE
    svc = SVC(kernel="linear")
    min_features_to_select = args.min_features
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy',
                  min_features_to_select=min_features_to_select)

    # model
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        class_weight='balanced',
        solver="liblinear",
        random_state=42)

    # transformer
    numeric_transformer = Pipeline(steps=[
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude="category")),
        ('cat', categorical_transformer, selector(dtype_include="category"))
    ])

    # pipeline
    pipe = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('rfecv', rfecv),
            ('model', model)])

    pipe.fit(X_train, y_train)

    print("end model")

    print("start logging metrics")

    y_pred = pipe.predict(X_test)

    for metric in [balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score]:
        run.log(f"{metric.__name__}", np.float(metric(y_test, y_pred)))

    print("end logging metrics")
    print("start output")

    # files saved in the "outputs" folder are automatically uploaded into run history
    # The outputs folder is however never created locally, just uploaded to the run instance
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(pipe, 'outputs/model.pkl')

    print("end output")
    print("end experiment")

    run.complete()


if __name__ == '__main__':
    main()
