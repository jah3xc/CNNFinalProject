import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import final project function
from src.final_project import final_project

if __name__ == "__main__":
    # run the final project
    final_project()