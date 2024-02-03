import schedule
import time
from functions import *


def data_pipeline_job():
    print("Job: Data Pipeline\nStatus: working...\n")
    new_data_found = collect_raw_data()

    if new_data_found:
        curated_data, season = collect_curated_data()

        if not curated_data.empty:
            create_model(players=curated_data, season=season)

    model = load_model()  # Don't know what to do with it now...

    print("\nJob: Data Pipeline\nStatus: done!\n\n")


if __name__ == "__main__":
    # Schedule the job every year
    schedule.every(10).seconds.do(data_pipeline_job)

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)
