from vifd.config.configuration import ConfigurationManager
from vifd.components.model_trainer import ModelTrainer
from vifd import logger 
from pathlib import Path

STAGE_NAME= "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()

if __name__=='__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj=ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(r">>>>>>>>> stage {} completed <<<<<<<<<\n=============x".format(STAGE_NAME))
    except Exception as e:
        logger.exception(e)
        raise e

