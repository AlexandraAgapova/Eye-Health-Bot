# Identification of of bags under the eyes and swelling of the eyes

This project is ML model with UI which can detect bags under the eyes and show 2d view of face without bags. 

I used Python 3.10.6

ver 0.0.1 
There are two programs which allows to lead around interesting area for us.


HowTo:
Firstly you need to install needable libraries for your python interpreter or create an virtual environment (venv). I recommend you to use ```$ python3 -m venv .venv``` this command will make directory ".venv" with virtual environment. To activate an environment ```$ source working_directory/.venv/bin/activate```. To deactivate environment ```$ deactivate```.




To train and use RF:
1. Load images in src/dataset/Dataset/ directories with predefined names of classes
2. Execute refactor_dataset.py in the same directory
3. Execute src/creation_data/create_csv_from_images.py
4. Execute learn.py in src/models/RandomForest You've learned the model
5. configure your csv file like in directory dataset/csv and call it test.csv
6. Execute Apply_model.py


Have a nice day)