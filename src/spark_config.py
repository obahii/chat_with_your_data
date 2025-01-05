from pyspark.sql import SparkSession
import os
import sys

def create_spark_session():
    """Create and configure Spark session with proper initialization checks."""
    try:
        # Set environment variables for Spark
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        # Create Spark session with necessary configurations
        spark = (
            SparkSession.builder.appName("DocumentProcessing")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.maxResultSize", "2g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            # Local mode configuration
            .config("spark.master", "local[*]")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )

        # Test Spark context
        if spark.sparkContext is None:
            raise Exception("Failed to initialize Spark context")

        return spark

    except Exception as e:
        print(f"Error initializing Spark session: {str(e)}")
        raise


def cleanup_spark(spark):
    """Clean up Spark session safely."""
    try:
        if spark and spark.sparkContext:
            spark.stop()
    except Exception as e:
        print(f"Error during Spark cleanup: {str(e)}")
