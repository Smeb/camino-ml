.PHONY: clean install gen_data

all:

gen_data:
	python main.py gen_data

train:
	python main.py train Tensor

clean:
	rm -rf data

install:
	git submodule init
	git submodule update
	cd camino
	make -C ./camino
	cd ..
	python setup.py install
