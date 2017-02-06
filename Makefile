.PHONY: clean clean_data install gen_data

gen_data:
	python3 main.py

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
	python3 setup.py install
