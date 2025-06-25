from pyopensky.trino import Trino
from datetime import datetime
import pandas as pd
import os
import time
import json

trino = Trino()

AIRCRAFT_DB = pd.read_csv("aircraft.csv")
AIRPORT_DB = pd.read_csv("airports.csv")
REQUESTED_COLUMNS:tuple[str] = ("lastposupdate", "lat", "lon", "velocity", "heading", "vertrate", "baroaltitude", "geoaltitude") # "time" ,lastcontact, "onground", "icao24"
FLIGHTLIST_FILE_NAME = "flights.csv"
FLIGHT_DATA_PATH = "FlightData"
FLIGHT_PATHS_SUBDIR = "Raw"
RUN_PARAMETERS_FILE = "run_parameters.json"


def find_bada_typecodes(bada_path:str = "./BADA") -> list[str]:
    typecodes:list[str] = []
    with open(os.path.join(bada_path, "ReleaseSummary")) as file:
        lines = file.readlines()
        lines = lines[9:]
        for line in lines:
            if line[7:10] == "PTF":
                typecode = line[:4].strip("_")
                typecodes.append(typecode)
    return typecodes


def get_flight_list(scanning_start:datetime, scanning_stop:datetime, region:str|None=None, limit:int|None=None) -> None:
    """
    Get a list of flights from the OpenSky Network API.
    :param scanning_start: Include flights that land after this time.
    :param scanning_stop: Include flights that land before this time.
    :param region: Optional region to limit the flights to (EU, US, AS, AF, OC, SA, AN).
    :param limit: Optional limit on the number of flights to return.
    :return: DataFrame with flight information.
    """
    BADA_TYPECODES = find_bada_typecodes()
    flight_list = trino.flightlist(scanning_start, scanning_stop, selected_columns=["origin", "destination", "icao24", "firstseen", "lastseen"])
    if flight_list is None:
        raise Exception("No flights found in the specified time range.")
    flight_list.dropna(inplace=True)
    flight_list.rename(columns={"departure": "origin", "arrival": "destination"}, inplace=True)
    before_region_filter = len(flight_list)
    if region is not None:
        if region == "US":
            airports_icao = AIRPORT_DB[AIRPORT_DB["iso_country"] == "US"]["icao"]
        elif region in ['OC', 'AS', 'AF', 'AN', 'EU', 'SA']:
            airports_icao = AIRPORT_DB[AIRPORT_DB["continent"] == region]["icao"]
        else:
            raise Exception(f"Unknown region: {region}. Use 'EU', 'US', 'AS', 'AF', 'OC', 'SA' or 'AN'.")
        flight_list = flight_list[flight_list["origin"].isin(airports_icao) & flight_list["destination"].isin(airports_icao)]
        print(f"Dropped {before_region_filter - len(flight_list)} of {before_region_filter} flights due to region filter.")

    flight_list = flight_list.merge(AIRCRAFT_DB[["icao24", "typecode"]], on="icao24", how="left")
    len_flight_list = len(flight_list)
    flight_list = flight_list[flight_list["typecode"].isin(BADA_TYPECODES)]
    if len(flight_list) < len_flight_list:
        print(f"Dropped {len_flight_list - len(flight_list)} of {len_flight_list} flights without corresponding BADA file.")

    if limit is not None:
        len_flight_list = len(flight_list)
        if limit < len_flight_list:
            flight_list = flight_list.head(limit)
            print(f"Dropped {len_flight_list - limit} of {len_flight_list} flights due to limit.")

    flight_list["origin_name"] = flight_list["origin"].map(AIRPORT_DB.set_index("icao")["name"])
    flight_list["destination_name"] = flight_list["destination"].map(AIRPORT_DB.set_index("icao")["name"])

    flight_list.reset_index(drop=True, inplace=True)
    flight_list.to_csv(os.path.join(FLIGHT_DATA_PATH, FLIGHTLIST_FILE_NAME), index=True)


def load_adsb_from_durations(flight_list:pd.DataFrame) -> None:
    """
    Load ADS-B data for each flight in the provided flight list.
    :param flight_list: DataFrame with flight information.
    """
    for i in range(len(flight_list)):
        start = time.time()
        flight_to_import = flight_list.iloc[i]
        print(f"Loading flight number {i+1}/{len(flight_list)} from {flight_to_import['origin_name']} to {flight_to_import['destination_name']}, typecode {flight_to_import['typecode']}")
        flight_path = trino.history(start=flight_to_import["firstseen"],
                        stop=flight_to_import["lastseen"],
                        icao24=flight_to_import["icao24"],
                        selected_columns=REQUESTED_COLUMNS)
        filename = f"""{flight_to_import["origin"]}_{flight_to_import["destination"]}_{flight_to_import["typecode"]}_{flight_to_import["icao24"]}_{i+1}.csv"""
        if flight_path is not None:
            flight_path.to_csv(os.path.join(FLIGHT_DATA_PATH, FLIGHT_PATHS_SUBDIR, filename), index=False)
            print(f"""Saved to {filename}. Time taken: {time.time() - start:.2f} seconds.\n""")
    print("Finished loading flights.\n")


def main(scanning_start, scanning_stop, region=None, limit=None) -> None:
    """
    Check if flight data directory exists, if not create it.
    If flight list exists, load it and fetch outstanding flights to download.
    :param scanning_start: Include flights that land after this time.
    :param scanning_stop: Include flights that land before this time.
    :param region: Optional region to limit the flights to (EU, US, AS, AF, OC, SA, AN).
    :param limit: Optional limit on the number of flights to return.
    """
    if region == "None":
        region = None
    flights_to_download = None
    run_parameters = {
        "Scanning start": scanning_start,
        "Scanning stop": scanning_stop,
        "Region": region if region else "None",
        "Limit": limit if limit else "None"
        }
    if not os.path.exists(FLIGHT_DATA_PATH):
        print("Creating flight data directory...", end="")
        os.makedirs(FLIGHT_DATA_PATH)
        with open(os.path.join(FLIGHT_DATA_PATH, ".gitignore"), "w") as gitignore_file:
            gitignore_file.write("*")
        os.makedirs(os.path.join(FLIGHT_DATA_PATH, FLIGHT_PATHS_SUBDIR))
        print("done.")
    if not os.path.exists(os.path.join(FLIGHT_DATA_PATH, RUN_PARAMETERS_FILE)):
        print("Creating run parameters file...", end="")
        with open(os.path.join(FLIGHT_DATA_PATH, RUN_PARAMETERS_FILE), "w") as f:
            json.dump(run_parameters, f, indent=4, default=str)
        print("done.")
    else:   # check we are downloading the same flights as before
        with open(os.path.join(FLIGHT_DATA_PATH, RUN_PARAMETERS_FILE), "r") as f:
            run_parameters_file = json.load(f)
            for key in run_parameters_file.keys():
                if str(run_parameters[key]) != str(run_parameters_file[key]):
                    raise Exception(f"{key} in run parameters file ({run_parameters_file[key]}) does not match current run parameters ({run_parameters[key]}).\n" + 
                                    f"Please delete the current folder: {FLIGHT_DATA_PATH} and try again.")
    if os.path.exists(os.path.join(FLIGHT_DATA_PATH, FLIGHTLIST_FILE_NAME)):
        flights_to_download = pd.read_csv(os.path.join(FLIGHT_DATA_PATH, FLIGHTLIST_FILE_NAME))
        if os.path.exists(os.path.join(FLIGHT_DATA_PATH, FLIGHT_PATHS_SUBDIR)):
            idx_already_downloaded = []
            for flight in os.listdir(os.path.join(FLIGHT_DATA_PATH, FLIGHT_PATHS_SUBDIR)):
                flight_idx = flight.split("_")[-1].split(".")[0]
                idx_already_downloaded.append(int(flight_idx))
        else:
            os.makedirs(os.path.join(FLIGHT_DATA_PATH, FLIGHT_PATHS_SUBDIR))
            idx_already_downloaded = []

        flights_to_download = flights_to_download[~flights_to_download.index.isin(idx_already_downloaded)]
        print(f"Resuming downloads from {os.path.join(FLIGHT_DATA_PATH, FLIGHTLIST_FILE_NAME)}.",
              f"{len(flights_to_download)} of {len(flights_to_download)+len(idx_already_downloaded)} to go.")
        input("Press Enter to continue or Ctrl+C to exit...")

    if flights_to_download is None:
        print(f"No flights to download. Fetching flight list from {scanning_start} to {scanning_stop}...")
        if region:
            print(f"Region limited to {region}.")
        if limit:
            print(f"Limit set to {limit} flights.")
        get_flight_list(scanning_start, scanning_stop, region=region, limit=limit)
        print(f"Flight list fetched and saved to {FLIGHTLIST_FILE_NAME}. Now loading flights...")
        flights_to_download = pd.read_csv(os.path.join(FLIGHT_DATA_PATH, FLIGHTLIST_FILE_NAME))

    load_adsb_from_durations(flights_to_download)

if __name__ == "__main__":
    scanning_stop = datetime(2024, 11, 2)
    scanning_start = datetime(2024, 11, 1)
    region = "EU" # Options: "EU", "US", "AS", "AF", "OC", "SA", "AN", "None"
    main(scanning_start, scanning_stop, region)