.PHONY: upload download

upload: metacentrum/upload_list.txt
	rsync -urv --no-relative --files-from metacentrum/upload_list.txt ./ metacentrum:byzkit/

dl_dds:
	rsync -urv metacentrum:byzkit/dsl4c.zip ./notrack/testing/

dl_ds:
	rsync -urv metacentrum:byzkit/dsl1k.zip ./notrack/public/

dl_m:
	rsync -urv metacentrum:byzkit/m1k ./notrack/public/

dl_dm:
	rsync -urv metacentrum:byzkit/m4c ./notrack/testing/

dl_t:
	rsync -urv metacentrum:byzkit/stderr.txt ./notrack/public/
