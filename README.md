# FEDn implementation of cifar10 with keras 
This repository contains an example client (and compute package and seed file) for FEDn training of a Keras VGG16 model using CIFAR-10.
This example is well suited for Jetson-Nano 2gb developed kit.
## Deploy the FEDn network (Reducer and Combiner)
Please follow the instructions in FEDn documentation https://scaleoutsystems.github.io/fedn/deployment.html to setup the Reducer and the Combiner in a distributed way

## Setting up and configuring the Jetson Nano NVIDIA  client natively
Jetson Nano is a small, powerful computer for embedded AI systems and IoT that delivers the power of modern AI in a low-power platform. The Jetson Nano is targeted to get started fast with the NVIDIA Jetpack SDK and a full desktop Linux environment, and start exploring a new world of embedded products, for more details please visit https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit .
<br/><b> *** In order to run FEDn client on Jetson Nano Tensorflow have be installed.</b><br/>

<u> <b> 1- Prerequisites and Dependencies </b></u><br/>
Before you install TensorFlow for Jetson, ensure you:

   - Install [JetPack](https://developer.nvidia.com/embedded/jetpack) on your Jetson device.
   - Install system packages required by TensorFlow:
      ```
        sudo apt-get update
        sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
      ```
   - Install and upgrade pip3:
      ```
        sudo apt-get install python3-pip
        sudo -H pip3 install --upgrade setuptools
      ```
   - Install the Python package dependencies:
      ```
        sudo -H pip3 install -U testresources numpy
        sudo -H pip3 install -U testresources
        sudo -H pip3 install -U numpy==1.16.1
        sudo pip3 install -U --no-deps future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
        sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
      ```
<u> <b> 2- Set up the Virtual Environment </b></u>

- Install the virtualenv package and create a new Python 3 virtual environment:

   ```
      sudo apt-get install virtualenv
      python3 -m virtualenv -p python3 <venv_name>
      source env1/bin/activate
   ```
- Activate the virtual environment:

   ```
      source <venv_name>/bin/activate
   ```

- Install the desired version of TensorFlow and its dependencies:

   ```
      pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta setuptools testresources
      sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow
   ```
- Install FEDn:

   ```
     git clone https://github.com/scaleoutsystems/fedn.git
     cd fedn
     git checkout develop
     cd fedn
  ```
- Modify the <i> setup.py</i> file in order to fit with the installed dependencies and python version.
  ```
  sudo apt install nano
  sudo nano setup.py
   ```
  - Change python version 
    ```
    python_requires='>=3.8,<3.9', to  python_requires='>=3.6,<3.9'
    ```
  - Change numpy version 
    ```
    "numpy~=1.22.2", to  "numpy~=1.19.1",
    ```
- Install FEDn
    ```
      pip install -e .
    ```
<u> <b> 3- Configure the example you want to run: </b></u>

   - Return to the main directory and clone the example
   
       ```bash
         cd ../..
         git clone https://github.com/saadiabadi/Jetson-Nano-Setup-Fedn.git
         cd Jetson-Nano-Setup-Fedn
       ```

- Set the reducer <i> <b> discover_host: </b></i>  to the right address in client.yaml file. 
    ```
      sudo nano client.yaml
    ```
- Add the Combiner/s IP and its corresponding name into <i> <b> /etc/hosts/ </b></i> file. 
    ```
      sudo nano /etc/hosts/
    ```
<u> <b> 3- Finally, attaching Jetson nano client to the federation: </b></u>

   ```bash
      fedn run client -in client.yaml --name Jetson
   ```


[comment]: <> (## Configuring the Reducer  )

[comment]: <> (Navigate to 'https://localhost:8090' &#40;or the url of your Reducer&#41; and follow instructions to upload the compute package in 'package/package.tar.gz' and the initial model in 'initial_model/initial_model.npz'. )

## Creating a compute package
Whenever you make updates to the client code (such as altering any of the settings in the above mentioned file), you need to re-package the compute package:

```bash
tar -czvf package.tar.gz client
```
To clear the system and set a new compute package, see: https://github.com/scaleoutsystems/fedn/blob/master/docs/FAQ.md

For an explaination of the compute package structure and content: https://github.com/scaleoutsystems/fedn/blob/develop/docs/tutorial.md
 
## Creating a new initial model
The baseline model (VGG16) is specified in the file 'client/init_model.py'. This script creates an untrained neural network and serializes that to a file.  If you wish to alter the initial model, edit 'init_model.py' and 'models/imdb_model.py' then regenerate the initial model file (install dependencies as needed, see requirements.txt):

```bash
python init_model.py 
```
### Configuring the client
We have made it possible to configure a couple of settings to vary the conditions for the training. These configurations are expsosed in the file 'settings.yaml': 

```yaml 
# Parameters for local training
test_size: 0.25
batch_size: 32
epochs: 1
#trained layers 0 means all layers in the model, otherwise just select the layers based on the identified number
trained_Layers: 6
```





## License
Apache-2.0 (see LICENSE file for full information).
