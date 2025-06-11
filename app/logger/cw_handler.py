import watchtower
import boto3
from app.core.config import settings


session = boto3.Session(
    aws_access_key_id = settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY,
    region_name = settings.AWS_CLOUDWATCH_REGION
)
boto3_client = session.client('logs', region_name = settings.AWS_CLOUDWATCH_REGION)

boto3_client = session.client('logs', region_name = settings.AWS_CLOUDWATCH_REGION)
response = boto3_client.describe_log_groups()

cloud_handler = watchtower.CloudWatchLogHandler(
    boto3_client = boto3_client,
    log_group_name = settings.AWS_CLOUDWATCH_LOG_GROUP,
    log_stream_name = settings.AWS_CLOUDWATCH_LOG_STREAM
    )
	