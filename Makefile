.PHONY: clean clean_data install gen_data train-tensor train-zeppelingdrcylinderssphere train-ballgdrcylinderdot train-ballcylinderdot train-tensorcylindersphere train-all
all:

generate:
	python main.py generate

train-all:
	python main.py train-all

evaluate-all:
	python main.py evaluate-all

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
