.PHONY: clean install gen_data

all:

clean:
	rm -rf data

install:
	git submodule init
	git submodule update
	cd camino
	make -C ./camino
	cd ..
	python3 setup.py install
