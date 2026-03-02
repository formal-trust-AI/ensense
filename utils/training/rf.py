import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import argparse
import json

class RandomForestConfig:
    """Holds all hyperparameters for Random Forest."""
    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

class RandomForestTrainer:
    """Handles training, evaluation, and saving the Random Forest model."""
    def __init__(self, config: RandomForestConfig, train_file: str, test_file: str):
        self.config = config
        self.train_file = train_file
        self.test_file = test_file
        self.model = None

    def load_data(self):
        train_df = pd.read_csv(self.train_file)
        test_df = pd.read_csv(self.test_file)
        target_column = "label"

        self.X_train = train_df.drop(columns=[target_column])
        self.y_train = train_df[target_column]

        self.X_test = test_df.drop(columns=[target_column])
        self.y_test = test_df[target_column]

    def train(self):
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            criterion=self.config.criterion,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_impurity_decrease=self.config.min_impurity_decrease,
            bootstrap=self.config.bootstrap,
            oob_score=self.config.oob_score,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        return accuracy

    def save_model_json(self, path, accuracy):
        def tree_to_dict(tree, feature_names):
            tree_ = tree.tree_
            def recurse(node):
                if tree_.feature[node] != -2:  # not leaf
                    return {
                        "feature": feature_names[tree_.feature[node]],
                        "threshold": float(tree_.threshold[node]),
                        "left": recurse(tree_.children_left[node]),
                        "right": recurse(tree_.children_right[node])
                    }
                else:
                    return {"value": tree_.value[node].tolist()}
            return recurse(0)

        model_dict = {
            "config": vars(self.config),
            "accuracy": accuracy,
            "trees": [tree_to_dict(t, self.X_train.columns) for t in self.model.estimators_]
        }

        with open(path, "w") as f:
            json.dump(model_dict, f, indent=2)
        print(f"Trained Random Forest model saved as {path}")


    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Random Forest classifier and save as JSON")
    parser.add_argument("train_csv", help="Path to training CSV file")
    parser.add_argument("test_csv", help="Path to test CSV file")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--criterion", type=str, default="gini")
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--max_leaf_nodes", type=int, default=None)
    parser.add_argument("--min_impurity_decrease", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=bool, default=True)
    parser.add_argument("--oob_score", type=bool, default=False)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="random_forest_model.joblib")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = RandomForestConfig(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        max_leaf_nodes=args.max_leaf_nodes,
        min_impurity_decrease=args.min_impurity_decrease,
        bootstrap=args.bootstrap,
        oob_score=args.oob_score,
        n_jobs=args.n_jobs,
        random_state=args.random_state
    )

    trainer = RandomForestTrainer(config, args.train_csv, args.test_csv)
    trainer.load_data()
    trainer.train()
    accuracy = trainer.evaluate()
    # trainer.save_model_json(args.model_path, accuracy)
    trainer.save_model(args.model_path)
