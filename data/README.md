Load API key to your shell:
```
export $(cat .env | xargs)
```

The Python scripts `download...` and `catchment...` are to be ran locally with the API
key - they download the actual discharge data and catchment geo boundaries. 

We get the weather data from the Caravan dataset:
[GitHub](https://github.com/kratzert/Caravan/)

The are two notebooks we run for the weather data:

Part1
It's meant to be ran on Google Colab.
It does some magical things and then connects to Google
Earth Engine to execute actual export tasks there.
It downloads weather data for our catchments.
The process once started takes several hours (runs on the cloud)
and saves the data to your Google Drive.

Part2
It's meant to be ran locally. It transforms the raw ELA5 land time series
(hourly) into daily aggregates in local time and merges with our discharge data.

The `build_final_dataset.py` Python script is for building the final dataset we'll use 
in experiments. Merges the discharge data, USGS quality information and
some stuff from Caravan.

`event_descriptions.py` generates our prompts with the textual
weather context from the last 7 days. It has an automated leakage test (randomizing
some future values to make sure prompt doesn't change).

# Setup Order
1. `download_discharge.py`
2. `catchment_boundaries.py`
3. Caravan Part 1
4. Caravan Part 2
5. `build_final_dataset.py`