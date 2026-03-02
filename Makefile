base_dst := $(HOME)/tmp
dst_public := $(base_dst)/sensitivity-public

runtests:
	python ./unit-tests/run.py
	
iclr2026:
	python ./ICLR2026/run_test.py

