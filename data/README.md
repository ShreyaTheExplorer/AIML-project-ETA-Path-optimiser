# Data
Not committed to git. Download:
1. Trips: Kaggle competition "NYC Taxi Trip Duration" -> train.zip -> data/raw/train.csv
   (kaggle competitions download -c nyc-taxi-trip-duration, after accepting the rules)
2. Weather: written by preprocessing/weather.py (Open-Meteo archive API, hourly, free, no key) to data/raw/weather_nyc_2016.csv
3. Gazetteer & Multi-City Data:
   - `data/cities.csv`: Operational cities (Pune, Mumbai, Bengaluru, Delhi) with centers, map zoom, and service radius.
   - `data/gazetteer/*.csv`: 25 localities per city (Bengaluru, Pune, Mumbai, Delhi) within compact service areas.
   - `data/city_types.csv`: Mapping of cities to `Metropolitan` type for 27-feature ETA model contract.
   - `data/festivals_india.csv`: National and state-specific holidays with `verified` flag.
   - `data/loader.py`: Module validating and caching cities, gazetteers, and context resolution at startup.

