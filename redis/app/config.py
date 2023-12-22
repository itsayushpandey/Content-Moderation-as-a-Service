import os

class Config(object):
        SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
        GET_HOSTS_FROM = os.environ.get('GET_HOSTS_FROM') or 'dns'
        REDIS_HOST = os.environ.get('REDIS_HOST') or 'localhost'
        REDIS_PORT = os.environ.get('REDIS_PORT') or 6379
        MINIO_HOST = os.environ.get("MINIO_HOST") or "localhost:9000"
        MINIO_USER = os.environ.get("MINIO_USER") or "rootuser"
        MINIO_PASSWD = os.environ.get("MINIO_PASSWD") or "rootpass123"
