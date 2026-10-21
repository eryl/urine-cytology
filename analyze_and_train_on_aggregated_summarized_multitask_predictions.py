import argparse
from pathlib import Path
import re

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

label_dict = {'P2': 0, 'P3': 1, 'P4': 2, 'P5': 2}

def get_class(base_name):
    class_label = base_name.split(' ')[0]
    return label_dict[class_label]

def explicit_classification(features):
    nc_above_p5 = features["NC ratio 0.5 < 0.7"] + features["NC ratio > 0.7"]
    print(nc_above_p5)
    return nc_above_p5



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('aggregated_predictions_path', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--feature-columns', nargs='+', default=["NC ratio 0.5 < 0.7",  
                                                                 "NC ratio < 0.5",  
                                                                 "NC ratio > 0.7",  
                                                                 "irregular_chromatin",  
                                                                 "coarse_chromatin",  
                                                                 "hyperchromasia",  
                                                                 "inflammatory",  
                                                                 "reactive_atypia"])
    args = parser.parse_args()

    df_logistic_regression = pd.read_csv(args.aggregated_predictions_path)
    df_logistic_regression['class'] = df_logistic_regression['base_image'].map(get_class)
    
    explicit_classifications = []
    logistic_regressions = []
    random_forest_predictions = []
    # We'll do two different evaluations: first is performance per predictor. Second is to take the median of the predictions across predictors for each image, and evaluate that "ensemble" predictor. 
    predictor_grouped = df_logistic_regression.groupby('predictor')
    for predictor, group in tqdm(predictor_grouped):
        y_true = group['class']
        features_df = group[args.feature_columns]
        #classification = explicit_classification(group)
        #explicit_classifications.append(classification)
        features = features_df.values
        splitter = LeaveOneOut()
        model_predictions_linear_regression = {}
        model_predictions_random_forest = {}
        for train_idx, test_idx in tqdm(splitter.split(features), total=len(features), desc=f"Predictor {predictor} LOOCV"):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = y_true.iloc[train_idx], y_true.iloc[test_idx]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            # For simplicity, we'll just do a logistic regression here, but you could try more complex models or cross-validation if you wanted.
            linear_regression_model = sklearn.linear_model.LogisticRegression(max_iter=1000)
            linear_regression_model.fit(X_train, y_train)
            y_pred_linear_regression = linear_regression_model.predict(X_test)
            test_image_name = group.iloc[test_idx]['base_image'].values[0]
            model_predictions_linear_regression[test_image_name] = y_pred_linear_regression[0]

            random_forest_model = sklearn.ensemble.RandomForestClassifier(n_estimators=50)
            random_forest_model.fit(X_train, y_train)
            y_pred_random_forest = random_forest_model.predict(X_test)
            model_predictions_random_forest[test_image_name] = y_pred_random_forest[0]
        logistic_regressions.append(model_predictions_linear_regression)
        random_forest_predictions.append(model_predictions_random_forest)
            # report_df = pd.DataFrame(report).transpose()
            # report_df.to_csv(args.output_dir / f"{predictor}_classification_report.csv")
    
    df_logistic_regression = pd.DataFrame(logistic_regressions)  # rows = models, columns = images
    # Count votes per image per class
    vote_counts = df_logistic_regression.apply(pd.Series.value_counts, axis=0).T.fillna(0).astype(int)
    # Now: rows = images, columns = classes (0,1,2,3)
    vote_counts.columns.name = "Class"
    vote_counts.index.name = "Image"
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        vote_counts,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt="d",
        linewidths=0.5,
        cbar_kws={"label": "# predictors"}
    )
    ax.set_xlabel("Severity class")
    ax.set_ylabel("Image")
    ax.set_xticklabels(["P2", "P3", "P4", "P5"])
    plt.tight_layout()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.output_dir / "logistic_regression_predictions_heatmap.png")

    
    df_random_forest = pd.DataFrame(random_forest_predictions)  # rows = models, columns = images
    # Count votes per image per class
    vote_counts = df_random_forest.apply(pd.Series.value_counts, axis=0).T.fillna(0).astype(int)
    # Now: rows = images, columns = classes (0,1,2,3)
    vote_counts.columns.name = "Class"
    vote_counts.index.name = "Image"
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        vote_counts,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt="d",
        linewidths=0.5,
        cbar_kws={"label": "# predictors"}
    )
    ax.set_xlabel("Severity class")
    ax.set_ylabel("Image")
    ax.set_xticklabels(["P2", "P3", "P4", "P5"])
    plt.tight_layout()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.output_dir / "random_forest_predictions_heatmap.png")


if __name__ == "__main__":
    main()