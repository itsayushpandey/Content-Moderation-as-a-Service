VERSION=v4
DOCKERUSER=aypa2130

build:
	docker build -t minio-example .

push:
	docker tag minio-example $(DOCKERUSER)/minio-example:$(VERSION)
	docker push $(DOCKERUSER)/minio-example:$(VERSION)
	docker tag minio-example $(DOCKERUSER)/minio-example:latest
	docker push $(DOCKERUSER)/minio-example:latest