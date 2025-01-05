from pyspark.sql import SparkSession


def create_spark_session():
    """Create and configure Spark session."""
    spark = (
        SparkSession.builder.appName("DocumentProcessing")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )
    return spark


def cleanup_spark(spark):
    """Clean up Spark session."""
    if spark is not None:
        spark.stop()
