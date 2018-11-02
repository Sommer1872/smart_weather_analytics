# Smart Weather Analytics

Apelt, Gosselke, Sommer & Tuchschmid


## Data
Unfortunately, this code uses proprietary data that we are not allowed to make widely accessible. If interested, please contact the owner of this repo directly.


## Installation instructions
(these steps have been tested on MacOS. For Linux/Windows, please adjust accordingly)

Use your terminal/command line to  navigate to the preferred location on your drive and execute the following command to download the whole project:
```bash
git clone https://github.com/Sommer1872/smart_weather_analytics.git
```

We highly encurage the use of anaconda as a package manager. If you have anaconda, you can simply `cd smart_weather_analytics` and then run:
```bash
conda env create -f requirements.yml
```
This creates a separate environment named `SWA` and automatically installs all the necessary packages.


## Run instructions

The main file is [master.py](master.py). You can run it to train a NN that tries to predict stock movements from weather data. It also runs different regressions to detect a relationship.
