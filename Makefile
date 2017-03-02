.PHONY: clean clean_data install gen_data train-tensor train-zeppelingdrcylinderssphere train-ballgdrcylinderdot train-ballcylinderdot train-tensorcylindersphere train-all
all:

gen_data:
	python main.py gen_data

train-all:
	python main.py train Tensor
	python main.py train ZeppelinGDRCylindersSphere
	python main.py train BallGDRCylinderDot
	python main.py train BallCylinderDot
	python main.py train TensorCylinderSphere

train-tensor:
	python main.py train Tensor

train-zeppelingdrcylinderssphere:
	python main.py train ZeppelinGDRCylindersSphere

train-ballgdrcylinderdot:
	python main.py train BallGDRCylinderDot

train-ballcylinderdot:
	python main.py train BallCylinderDot

train-tensorcylindersphere:
	python main.py train TensorCylinderSphere

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
