import boto3
from app.core.config import settings
from botocore.exceptions import ClientError
def get_s3_client():
    """
    Create and return a boto3 S3 client using credentials from environment variables.
    """
    session = boto3.Session(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.S3_REGION
    )

    s3_client = session.client('s3')

    return s3_client
