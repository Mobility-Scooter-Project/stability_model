.PHONY: model

setup:
	mkdir data

test:
	python -B ./src/test.py

train:
	python -B ./src/train.py

black:
	black ./

predict: 
	python -B ./src/predict.py

pca:
	python -B ./src/pca.py

build:
	pip install pyinstaller
	pyinstaller --noconsole --onefile GUI.py 
	pyinstaller GUI.spec
	python -B ./build_utils.py