.PHONY: clean clean_data install gen_data train
all:

gen_data:
	python main.py gen_data

train:
	python main.py train Tensor

clean:
	rm -rf dist build data ml_experiments.egg-info __pycache__

clean_data:
	rm -rf data

install:
	git submodule init
	git submodule update
	cd camino
	make -C ./camino
	cd ..
