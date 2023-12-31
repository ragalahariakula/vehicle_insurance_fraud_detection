from vifd.config.configuration import ConfigurationManager
from vifd.components.data_ingestion import DataIngestion
from vifd import logger 

STAGE_NAME= "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__=='__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj=DataIngestionTrainingPieline()
        obj.main()
        logger.info(r">>>>>>>>> stage {} completed <<<<<<<<<\n=============x".format(STAGE_NAME))
    except Exception as e:
        logger.exception(e)
        raise e
