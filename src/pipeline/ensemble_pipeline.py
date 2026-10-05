import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.ensemble_trainer import EnsembleTrainer


def run_ensemble(module_name):

    try:

        # ==============================================
        # STEP 1: DATA INGESTION
        # ==============================================

        ingestion = DataIngestion(
            module_name=module_name
        )

        train_path, test_path = (
            ingestion.initiate_data_ingestion()
        )


        # ==============================================
        # STEP 2: DATA TRANSFORMATION
        # ==============================================

        transformation = DataTransformation(
            module_name=module_name
        )

        X_train, y_train, X_test, y_test = (
            transformation.initiate_data_transformation(
                train_path,
                test_path
            )
        )


        # ==============================================
        # STEP 3: ENSEMBLE TRAINING
        # ==============================================

        trainer = EnsembleTrainer(
            module_name=module_name
        )

        best_name, best_score, results = (
            trainer.initiate_ensemble_training(
                X_train,
                y_train,
                X_test,
                y_test
            )
        )


        print("\n")
        print("=" * 60)

        print(
            f"FINAL BEST MODEL: {best_name}"
        )

        print(
            f"FINAL F1 SCORE: {best_score:.4f}"
        )

        print("=" * 60)


    except Exception as e:

        print(
            f"ERROR: {str(e)}"
        )

        raise


if __name__ == "__main__":

    module = "diabetes"

    run_ensemble(module)