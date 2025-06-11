from pydantic_settings import BaseSettings
class Settings(BaseSettings):
  
  AZURE_OPENAI_API_KEY: str
  AI_AGENT_API_KEY: str
  OPENAI_API_KEY: str
  DEFAULT_OPENAI_MODEL: str
  CACHE_ENVIRONMENT: str
  APP_PREFIX: str
  CACHING_KEY_PREFIX: str
  REDIS_HOST: str
  REDIS_PORT: int
  REDIS_DB: str
  REDIS_USER: str
  REDIS_PASSWORD: str
  REDIS_NAMESPACE: str
  AWS_ACCESS_KEY_ID: str
  AWS_SECRET_ACCESS_KEY: str
  AWS_CLOUDWATCH_REGION: str
  AWS_CLOUDWATCH_LOG_GROUP: str
  AWS_CLOUDWATCH_LOG_STREAM: str
  BUCKET_NAME: str
  S3_REGION: str

  class Config:
    extra = 'allow'
    env_file = ".env"
    case_sensitive = False

settings = Settings() # type: ignore