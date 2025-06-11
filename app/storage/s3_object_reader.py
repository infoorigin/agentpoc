from app.storage.object_reader import ObjectReader
import boto3
from io import BytesIO

class S3ObjectReader(ObjectReader):
    def __init__(self, bucket: str, key: str, s3_client=None):
        self.bucket = bucket
        self.key = key
        self.s3_client = s3_client or boto3.client("s3")

    def read(self):
        response = self.s3_client.get_object(Bucket=self.bucket, Key=self.key)
        return BytesIO(response["Body"].read())
