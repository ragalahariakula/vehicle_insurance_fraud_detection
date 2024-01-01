from vifd import logger
from src.vifd.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.vifd.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.vifd.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.vifd.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.vifd.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_ingestion = DataValidationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataTransformationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = ModelTrainerTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME= "Model Evaluation stage" 
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj=ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info(r">>>>>>>>> stage {} completed <<<<<<<<<\n=============x".format(STAGE_NAME))
except Exception as e:
    logger.exception(e)
    raise e