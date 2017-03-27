git clone https://github.com/smeb/camino
sudo apt install make
sudo apt install default-jdk
sudo apt install virtualenv
sudo apt install gcc
sudo apt install python-dev
sudo apt install python-tk
cd camino-ml
make install
cp src/config.py.example src/config.py
virtualenv -p /usr/bin/python2.7 ml_research
source ml_research/bin/activate
pip install -r requirements.txt
