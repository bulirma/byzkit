.PHONY: upload download

upload: metacentrum/upload_list.txt
	rsync -urv --no-relative --files-from metacentrum/upload_list.txt ./ metacentrum:byzkit/

dl_ds:
	rsync -urv metacentrum:byzkit/dsl1k.zip ./notrack/

dl_m:
	rsync -urv metacentrum:byzkit/m1k ./notrack/

dl_dm:
	rsync -urv metacentrum:byzkit/m32 ./notrack/
