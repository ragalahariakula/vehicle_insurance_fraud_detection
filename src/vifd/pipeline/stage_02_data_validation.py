from vifd.config.configuration import ConfigurationManager
from vifd.components.data_validation import DataValidation
from vifd import logger 
from pathlib import Path

STAGE_NAME= "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()

if __name__=='__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj=DataValidationTrainingPipeline()
        obj.main()
        logger.info(r">>>>>>>>> stage {} completed <<<<<<<<<\n=============x".format(STAGE_NAME))
    except Exception as e:
        logger.exception(e)
        raise e
