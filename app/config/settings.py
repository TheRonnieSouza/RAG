from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, BaseModel, SecretStr
from pathlib import Path

class GoogleSettings(BaseModel):
    embedding_model_name: str = "Falhou"
    embedding_api_key: SecretStr = SecretStr("Falhou")
    
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=r"C:\Projetos\Python\RAG\.env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    google: GoogleSettings
   

