.PHONY: upload download

upload: metacentrum/upload_list.txt
	rsync -urv --no-relative --files-from metacentrum/upload_list.txt ./ metacentrum:byzkit/

download:
	rsync -urv metacentrum:byzkit/dev-models/ ./dev-models/
