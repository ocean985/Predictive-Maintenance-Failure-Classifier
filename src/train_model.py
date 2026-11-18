"""
CS 480 Final Project - Predictive Maintenance Failure Classifier
This part deals with: training the models, metrics, and model comparison.

This code loads the data from data/ai4i2020.csv (from the UCI Machine Learning Repository).
This trains and compares models and results by evaluating their performance and explaining their predictions.
The models we are creating are 
    1. Baseline Classifier
    2. Logistic Regression
    3. Decision Tree
    4. Random Forest
The best model is saved to models/best_model.joblib and the model comparison results are saved to results/model_comparison.csv.
"""
# LIBRARIES we are importing
import json #to save model info to a json file
from pathlib import Path # makes file paths that work across operating systems

import joblib # saving trained ml model
import matplotlib # plotting library
matplotlib.use("Agg")  # lets plots be saved to files without needing a display (headless server)

import matplotlib.pyplot as plt #for graphs and visualizations
import numpy as np # for numerical operations and arrays
import pandas as pd # data manipulation and analysis

from sklearn.compose import ColumnTransformer # for preprocessing different types of features (numeric, categorical)
from sklearn.dummy import DummyClassifier # for creating a baseline model to predict the majority class
from sklearn.ensemble import RandomForestClassifier # for building a random forest model
from sklearn.inspection import permutation_importance # for evaluating feature importance by measuring the effect of shuffling features
from sklearn.linear_model import LogisticRegression # for building a logistic regression model

from sklearn.metrics import (
    ConfusionMatrixDisplay, # for visualizing confusion matrices
    accuracy_score,
    average_precision_score, # for calculating the average precision score (for imbalanced classifications)
    confusion_matrix, # for calculating the confusion matrix
    f1_score, 
    make_scorer, # for creating custom scoring functions
    precision_score,
    recall_score,
    PrecisionRecallDisplay, # for visualizing precision-recall curves
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict, # creates cross-validation folds preserving class proportions
    cross_val_score, # predictions using held-out training folds
    train_test_split # splitting the dataset into training and testing sets
)
from sklearn.pipeline import Pipeline # for creating a pipeline of preprocessing and ml models (reference: https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)
from sklearn.preprocessing import OneHotEncoder, StandardScaler # OneHotEncoder for preprocessing categorical and numerical features and StandardScaler for standarizing numerical features
from sklearn.tree import DecisionTreeClassifier, plot_tree # for building and visualizing decision tree models

#Setting up the paths for the data, models, and results directories. 
# The paths are built from the location of this file, so it works on Windows, Mac, and Linux.

# assumes the the script is inside the project's src folder 
ROOT = Path(__file__).resolve().parent.parent
# location of dataset
DATA_FILE = ROOT / "data" / "ai4i2020.csv"
# where the trained model will be saved
MODEL_DIR = ROOT / "models"
# where the results (tables, plots, evaluations) will be saved
RESULTS_DIR = ROOT / "results"

#makes the models and results folders if not already there
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42  # for reproducibility of results
NUMBER_OF_FOLDS = 5  # for cross-validation

#classification threshold 
THRESHOLD = np.round(np.arange(0.10, 1.00, 0.05), 2) # threshold of 0.5 = probability of 50% or more is classified as a failure

# LOADING THE DATASET

#checking if the dataset exists in the data folder, 
# if not it raises an error and asks the user to put the dataset in the data folder
if not DATA_FILE.exists():
    raise SystemExit(
        f"Cannot find {DATA_FILE}\n" 
        "Put ai4i2020.csv in the data/ folder.")

df = pd.read_csv(DATA_FILE) # reads csv file into a pandas dataframe

# Displays the dataset dimensions
print("\nDATASET INFORMATION:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# The original column names have brackets and spaces. so instead we rename them to shorter names
# (makes names easier to use in flask as well later)
df = df.rename(columns={
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Type": "type",
    "Machine failure": "failure",
})

#FEATURES AND TARGET VARIABLE

# machine sensor measurements
NUMERIC_FEATURES = [
    "air_temp", 
    "process_temp", 
    "rpm", 
    "torque", 
    "tool_wear"
]
#  machine type (L, M, or H)
CATEGORICAL_FEATURES = ["type"]            # L, M, H
# combines all 6 input features (5 numeric + 1 categorical)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# precautionary check that the dataset has all the required columns

# makes sure all required columns exist in the dataset.
REQUIRED_COLUMNS = FEATURES + ["failure"]
# gives a list of any required columns that are missing from the dataset
missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]
# stops execution if a required column is missing
if missing_columns:
    raise SystemExit(
        f"Missing required dataset columns: {missing_columns}"
    )

# We do NOT use UDI / Product ID (just identifiers) and we do NOT use TWF, HDF, PWF, OSF, RNF
# Those five say WHICH kind of failure happened, so they would give away the answer (data leakage)
# so we only use the 6 input features and make separate DataDrame containing those six features
X = df[FEATURES].copy()
# sets machine failure as the target variable (1 = failure, 0 = normal)
y = df["failure"].copy()

#FOR DATA VALIDATION

# makes sure if any selected feature or target contains missing values
if X.isnull().any().any() or y.isnull().any():
    raise SystemExit(
        "Missing values were found in the selected features or target. "
        "Inspect the CSV before training."
    )

# see if the target contains only 0 and 1 (binary classification problem)
if not set(y.unique()).issubset({0, 1}):
    raise SystemExit(
        "The failure target must contain only 0 and 1."
    )

# see if both classes are present.
if y.nunique() != 2:
    raise SystemExit(
        "The dataset must contain both normal and failure records."
    )

# target to integer values. reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html
y = y.astype(int)

# check if all numerical features contain valid numerical values
for column in NUMERIC_FEATURES:

    # sends error if a numerical column contains non-numerical data
    X[column] = pd.to_numeric(X[column], errors="raise")

    # see if there are infinite or invalid numerical values
    if not np.isfinite(X[column]).all():
        raise SystemExit(
            f"Invalid numerical values found in {column}."
        )

# see if machine types match ouur expected dataset categories (prevents unexpected categories)
if not set(X["type"].unique()).issubset({"L", "M", "H"}):
    raise SystemExit(
        "Unexpected machine type found. Expected L, M, or H."
    )

# shows the number and percentage of each class.
failure_count = int(y.sum())
normal_count = int((y == 0).sum())

print("\nCLASS DISTRIBUTION: ")
print(f"Normal records: {normal_count}")
print(f"Failure records: {failure_count}")
print(f"Failure percentage: {y.mean():.2%}")

# shows a warning if the dataset is highly imbalanced.
if y.mean() < 0.10:
    print(
        "WARNING: For this Failure, Accuracy alone will not be sufficient for evaluation."
    )

# DATA SPLIT AND TRAINING/TESTING

# stratify=y keeps the same failure % in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("\nDATA SPLIT: ")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

#PREPROCESSING AND MODEL PIPELINES

# our each model consists of preprocessing + model in one object and is a pipeline
# so that the Flask app only needs to call model.predict() on raw values later
def make_preprocessor():
    #creates preprocessing object
    return ColumnTransformer([
        # standarizes numerical snesor measurements
        ("num", StandardScaler(), NUMERIC_FEATURES),
        # encodes categorical variables (handles_unknown prevents error for unseen categories)
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

# ML MODELS

models = {

    # our baseline model
    "Dummy Classifier": Pipeline([
        #for all models to have same general input structure
        # reference: https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data
        ("prep", make_preprocessor()),
        # predicts most frequent class in its training data (normal or failure)
        # reference: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
        ("model", DummyClassifier(strategy="most_frequent")),
    ]),

    "Logistic Regression": Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression( # reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
            class_weight="balanced", # gives more weight to the minority class
            max_iter=1000, # allows more iterations for convergence
            random_state=RANDOM_STATE)),
    ]),
    "Decision Tree": Pipeline([
        ("prep", make_preprocessor()),
        ("model", DecisionTreeClassifier(
            max_depth=5, # tress depth
            class_weight="balanced", 
            random_state=RANDOM_STATE)),
    ]),
    "Random Forest": Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=200, #200 trees in the forest
            class_weight="balanced",
            random_state=RANDOM_STATE, 
            n_jobs=1 #one cpu core to avoid memory issues on some systems
        )),
    ]),
}

# SETTING UP CROSS-VALIDATION AND EVALUATION METRICS

cv = StratifiedKFold(
    n_splits=NUMBER_OF_FOLDS, #each fold has the same proportion of normal and failure records as the original dataset roughly
    shuffle=True, 
    random_state=RANDOM_STATE)

F1_scorer = make_scorer(
    f1_score, 
    zero_division=0) # F1 scoring func that gives 0 if no positive predictions 

rows = [] # stores comparision results for all models
predictions = {} # stores final test probabilities and predictions for plotting
selected_threshold = {} # selected threshold used to train data for each model

# TRAINING, EVALUATION, AND COMPARISON THE MODELS

for name, pipe in models.items():
    # Cross-validation uses onlythe training data 
    # We have used it to choose the best model so 
    # We don't have pick the winner by looking at the test set
    print(f"\nTraining and evaluating: {name}")
    # reference: 
    # https://www.geeksforgeeks.org/machine-learning/what-is-the-oofout-of-fold-approach/
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html
    oof_probabilities = cross_val_predict(
        pipe,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba" #returns class probabilities
    )[:, 1] # only take the probabilities for the positive class (failure)

    #default 0.50 threshold
    oof_default_predictions = (
        oof_probabilities >= 0.50
    ).astype(int)

    #cross-validated F1 score at default threshold of 0.5
    cv_f1_default = f1_score(
        y_train,
        oof_default_predictions,
        zero_division=0
    )

    # stores the F1-score calculated at each candidate threshold
    threshold_scores = []

    # tests each threshold using out-of-fold training probabilities
    for threshold in THRESHOLD:

        # probabilities to normal/failure predictions
        threshold_predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        #F1-score at specified threshold.
        score = f1_score(
            y_train,
            threshold_predictions,
            zero_division=0
        )
        #threshold and its corresponding F1-score
        threshold_scores.append((threshold, score))

    #threshold with the highest training out-of-fold F1 (if tie among multiplr, the first one is selected)
    best_threshold, best_cv_f1 = max(
        threshold_scores,
        key=lambda item: item[1]
    )

    # selected threshold for later use.
    selected_threshold[name] = float(best_threshold)

    #fitting the pipeline using all training records but preprocessing is fitted only on training data
    pipe.fit(X_train, y_train)

    #failure probability for each test record
    test_probabilities = pipe.predict_proba(X_test)[:, 1]

    #adds the threshold selected using training data
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    #predictions for later graphs and example analysis
    predictions[name] = (
        test_predictions,
        test_probabilities
    )

    # for calculating the confusion matrix 
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    ).ravel()

    # calculating accuracy percentage of all correct predictions
    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    #for calculating precision percentage of correctly predicted failures
    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    #for calculating recall percentage of actual failures detected
    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    #F1-score (harmonic mean of precision and recall)
    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    #calculating Average Precision from test probabilities that summarizes precision-recall performance across thresholds
    test_average_precision = average_precision_score(
        y_test,
        test_probabilities
    )

    # saving results for the model

    rows.append({

        "model": name,

        # cross-validated F1 at default threshold of 0.5
        "cv_f1_default": round(cv_f1_default, 3),

        #best cross-validated F1 after threshold tuning
        "cv_f1_tuned": round(best_cv_f1, 3),

        #threshold selected using training data
        "selected_threshold": round(float(best_threshold), 2),

        # test set evaluation metrics
        "test_accuracy": round(test_accuracy, 3),
        "test_precision": round(test_precision, 3),
        "test_recall": round(test_recall, 3),
        "test_f1": round(test_f1, 3),
        "test_average_precision": round(test_average_precision, 3),

        # confusion matrix values
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn

    })

    # Dataframe for the model's confusion matrix

    # figure for the model's confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))

    # plots the confusion matrix 
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=[0, 1],
        display_labels=["Normal", "Failure"],
        ax=ax,
        colorbar=False
    )

    # adds title to the confusion matrix graph 
    ax.set_title(
        f"{name}\nThreshold = {best_threshold:.2f}"
    )

    # adjusts the layout of the graph to prevent overlapping
    fig.tight_layout()

    # saving the graph to the results folder
    filename = name.replace(" ", "_").lower()

    fig.savefig(
        RESULTS_DIR / f"confusion_matrix_{filename}.png",
        dpi=150
    )

    # release memory
    plt.close(fig)


# model comparision table

# creates a pandas DataFrame from the list of dictionaries containing model evaluation results
comparison = pd.DataFrame(rows)

# saves the model comparison table to a CSV file in our results folder
comparison.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

# displays the model comparison table in the console
print("\nMODEL COMPARISON")
print("----------------")
print(comparison.to_string(index=False))

# selected based on their cross-validated F1-score after threshold experimentation
eligible_models = comparison[
    comparison["model"] != "Dummy Classifier"
]

# selects the model with the highest cross-validated F1-score after threshold tuning
best_name = eligible_models.sort_values(
    "cv_f1_tuned",
    ascending=False
).iloc[0]["model"]

# gets the trained pipeline for the selected model
best_model = models[best_name]

# gets the threshold selected for that model
best_threshold = selected_threshold[best_name]

print("\nSELECTED MODEL")
print("--------------")
print(f"Best model: {best_name}")
print(f"Selected threshold: {best_threshold:.2f}")
print("Selection basis: highest cross-validated F1 on training data.")


#saves pipeline, preprocessing and classifier.
#later Flask application can load this file and provide raw sensor values
joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib"
)

# saving model name, feature order, machine types, and threshold
#for frontend Flask application can read this information when making predictions
model_info = {

    "best_model": best_name,

    "feature_order": FEATURES,

    "numeric_features": NUMERIC_FEATURES,

    "categorical_features": CATEGORICAL_FEATURES,

    "type_values": ["L", "M", "H"],

    "classification_threshold": float(best_threshold),

    "positive_class": 1,

    "negative_class": 0

}

# saves model information to a JSON file 
# reference: https://docs.python.org/3/library/json.html#basic-usage
with open(
    MODEL_DIR / "model_info.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        model_info,
        file,
        indent=4
    )


# RANDOM FOREST

#trained Random Forest model is selected for feature explanation
rf = models["Random Forest"]

# names of the features after preprocessing including the encoded machine-type columns
feature_names = (
    rf.named_steps["prep"].get_feature_names_out()
)

# gets random forests feature importance values (built-in)
builtin_importance = (
    rf.named_steps["model"].feature_importances_
)

#pandas Series for each feature with its importance
builtin = pd.Series(
    builtin_importance,
    index=feature_names
).sort_values()

#changes in when each input feature is shuffled (we are not using this to choose the model or tune threshold)
perm = permutation_importance(
    rf,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=RANDOM_STATE
)

# links each original input feature with thier average importance
perm_series = pd.Series(
    perm.importances_mean,
    index=FEATURES
).sort_values()

# makes two graphs(side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# plot Random Forest importance
builtin.plot.barh(ax=axes[0])

#title to the first graph
axes[0].set_title("Random Forest: Built-in Importance")

# plots permutation importance
perm_series.plot.barh(ax=axes[1])

# title to the second graph
axes[1].set_title("Random Forest: Permutation Importance")

# spacing between the graphs
fig.tight_layout()

# saves feature importance graph
fig.savefig(
    RESULTS_DIR / "feature_importance.png",
    dpi=150
)

plt.close(fig)

pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values(
    "importance_mean",
    ascending=False
).to_csv(
    RESULTS_DIR / "permutation_importance.csv",
    index=False
)

# LOGISTIC REGRESSION COEFFICIENTS

#gdets the trained Logistic Regression model
lr = models["Logistic Regression"]

# names of the preprocessed feature
lr_feature_names = (
    lr.named_steps["prep"].get_feature_names_out()
)

# gets learned coefficients (+ve increase the model's log-odds for failure and -ve decreases)
coefficients = lr.named_steps["model"].coef_[0]

#linkseach coefficient with its corresponding feature
coef = pd.Series(
    coefficients,
    index=lr_feature_names
)

#sorting features by the absolute size of their coefficients
coef = coef.reindex(
    coef.abs().sort_values(ascending=False).index
)

# coefficients saved  to a CSV file.
coef.to_csv(
    RESULTS_DIR / "logistic_regression_coefficients.csv",
    header=["coefficient"]
)

# DECISION TREE

dt = models["Decision Tree"]
dt_feature_names = (
    dt.named_steps["prep"].get_feature_names_out()
)
#figure for the decision tree
fig, ax = plt.subplots(figsize=(18, 8))

#only drawing top three levels to make it easier to read
plot_tree(
    dt.named_steps["model"],
    max_depth=3,
    feature_names=dt_feature_names,
    class_names=["Normal", "Failure"],
    filled=True,
    fontsize=7,
    ax=ax
)

#adjusting the layout
fig.tight_layout()

# Saves the decision tree image.
fig.savefig(
    RESULTS_DIR / "decision_tree.png",
    dpi=150
)

plt.close(fig)

#PRECISION-RECALL CURVES
fig, ax = plt.subplots(figsize=(7, 5))

#one curve for each model
for name, (_, probabilities) in predictions.items():

    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities,
        name=name,
        ax=ax
    )

ax.set_title("Precision-Recall Curves")

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "precision_recall_curves.png",
    dpi=150
)

plt.close(fig)

# EXAMPLE PREDICTIONS

# predictions and probabilities for the selected model
best_predictions, best_probabilities = predictions[best_name]

#copying the test features so the original data is not changed
examples = X_test.copy()

#adding the actual class for each test record
examples["actual"] = y_test.values

#adding predicted class
examples["predicted"] = best_predictions

#predicted probability of failure
examples["failure_probability"] = best_probabilities.round(3)

#grouping examples according to prediction outcomes
groups = {

    #actual failure and predicted failure
    "CORRECT FAILURE (TRUE POSITIVE)": examples[
        (examples.actual == 1) &
        (examples.predicted == 1)
    ],

    #actual failure but predicted normal
    "MISSED FAILURE (FALSE NEGATIVE)": examples[
        (examples.actual == 1) &
        (examples.predicted == 0)
    ],

    #actual normal but predicted failure
    "FALSE ALARM (FALSE POSITIVE)": examples[
        (examples.actual == 0) &
        (examples.predicted == 1)
    ],

    #actual normal and predicted normal
    "CORRECT NORMAL (TRUE NEGATIVE)": examples[
        (examples.actual == 0) &
        (examples.predicted == 0)
    ]

}

#text file to save example predictions
with open(
    RESULTS_DIR / "sample_predictions.txt",
    "w",
    encoding="utf-8"
) as file:

    #selected model name and threshold
    file.write(f"Selected model: {best_name}\n")
    file.write(f"Selected threshold: {best_threshold:.2f}\n")

    #five examples (max) from each prediction group
    for title, group in groups.items():

        file.write(
            f"\n=== {title}: {len(group)} records ===\n"
        )

        file.write(
            group.head(5).to_string() + "\n"
        )

#FINAL

print("\n...TRAINING COMPLETED...")

#location of the saved model
print(f"Saved model: {MODEL_DIR / 'best_model.joblib'}")

#location of the model information
print(f"Model information: {MODEL_DIR / 'model_info.json'}")

#location of the comparison table
print(f"Model comparison: {RESULTS_DIR / 'model_comparison.csv'}")

#location of all generated graphs and result files
print(f"Results folder: {RESULTS_DIR}")

print("\nAll models were trained and evaluated successfully.")
