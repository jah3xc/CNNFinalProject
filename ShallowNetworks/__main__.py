import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import final project function
from ShallowNetworks.src.run import run

if __name__ == "__main__":
    # run the final project
    run()