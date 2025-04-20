# Identification of of bags under the eyes and swelling of the eyes

This project is ML model which can detect bags under the eyes and show 2d view of face without bags. 

We used Python 3.10


ver 0.1.0
Now there is a test code to train Random Forest.


HowTo:
Firstly you need to install needable libraries for your python interpreter or create an virtual environment (venv). I recommend you to use ```$ python3 -m venv .venv``` this command will make directory ".venv" with virtual environment. To activate an environment ```$ source working_directory/.venv/bin/activate```. After activation you have to install all libraries from <<file requirements.txt>> Use command ```pip install -r requirements.txt``` in directory ```$*/ML-PROJECT```. To deactivate environment ```$ deactivate```.


Before training delete all *.txt files from directory data and it's subdirectories. They consist of information about data should be in that directory of project to work properly.

To train and use RF:
1. Load images in data/raw/Dataset/ directories with predefined names of classes
2. Execute refactor_dataset.py from src/data_preprocessing
3. Execute src/data_preprocessing/create_csv_from_images.py
4. Execute train.py in src/models/RandomForest You've trained the model
5. Configure your csv file like in directory data/processed/csv and call it test.csv
6. Execute inference.py


Have a nice day)
