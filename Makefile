.PHONY: clean clean_data install generate train-all
all:

extract-fits:
	python main.py extract-all

generate:
	python main.py generate

train-all:
	python main.py train-all

fit-all:
	python main.py fit-all

visualise:
	python main.py visualise

divide-fits:
	python main.py divide-fits

divide-dataset:
	python main.py divide-dataset

clean:
	rm -rf dist build data __pycache__

clean_data:
	rm -rf data

install:
	git submodule init
	git submodule update
	cd camino
	make -C ./camino
	cd ..
