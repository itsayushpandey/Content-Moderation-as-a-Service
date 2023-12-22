VERSION=v029
DOCKERUSER=aypa2130

build:
	docker build -f Dockerfile -t demucs-worker:${VERSION} .
	docker tag demucs-worker:${VERSION} $(DOCKERUSER)/demucs-worker:$(VERSION)
	docker push $(DOCKERUSER)/demucs-worker:$(VERSION)
	docker tag demucs-worker:${VERSION} $(DOCKERUSER)/demucs-worker:latest
	docker push $(DOCKERUSER)/demucs-worker:latest
