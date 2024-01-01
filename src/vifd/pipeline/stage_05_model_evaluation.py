from vifd.config.configuration import ConfigurationManager
from vifd.components.model_evaluation import ModelEvaluation
from vifd import logger 

STAGE_NAME= "Model Evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()

if __name__=='__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj=ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(r">>>>>>>>> stage {} completed <<<<<<<<<\n=============x".format(STAGE_NAME))
    except Exception as e:
        logger.exception(e)
        raise e
