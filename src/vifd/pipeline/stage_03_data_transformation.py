from vifd.config.configuration import ConfigurationManager
from vifd.components.data_transformation import DataTransformation
from vifd import logger
from pathlib import Path

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_splitting()
            else:
                logger.error("Your data schema is not valid")

        except Exception as e:
            logger.exception(f"An error occurred: {str(e)}")
            raise e
