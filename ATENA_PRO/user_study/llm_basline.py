import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

NETFLIX_FILENAME = "netflix.tsv"
NETFLIX_SCHEME = "show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description"
NETFLIX_NUMERIC_COLUMNS = "show_id, release_year"
FLIGHTS_FILENAME = "flights.tsv"
FLIGHTS_SCHEME = "flight_id, airline, origin_airport, destination_airport, flight_number, delay_reason, departure_delay, scheduled_trip_time, scheduled_departure, scheduled_arrival, day_of_week, day_of_month, month"
FLIGHTS_NUMERIC_COLUMNS = "no numeric columns"
PLAYSTORE_FILENAME = "1.tsv"
PLAYSTORE_SCHEME = "app_id, name, category, rating, reviews, app_size_kb, installs, type, price, content_rating, last_updated, min_android_ver"
PLAYSTORE_NUMERIC_COLUMNS = "rating, reviews, app_size_kb, installs, price"

def ChatGPT(prompt, should_print=False):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    res = completion.choices[0].message["content"]
    if should_print:
        print("ChatGPT answer:")
        print(res)
    return res


def createPrompt(filename, scheme, numeric_columns, task):
    print("--------------------------------")
    print(task)

    # External Knowledge: {external_knowledge}

    return f"""
Dataset file name: \'{filename}\'
Dataset scheme: {scheme}
Dataset numeric columns: {numeric_columns}

Write pandas code to answer the following task for the dataset provided above:
{task}"""



t1 = "find a country with different, atypical viewing habits, compared to the rest of the world"
t2 = "investigate the properties of “successful” TV shows that have more than one season"
t6 = "explore different reasons of flight delays"
t5 = "show interesting properties of flights in the summer"
t9 = "show properties of apps with at least one million installs"
t10 = "compare high-rated apps with low-rated apps"

# netflix_external_knowledge = "for TV shows use type equals \'TV Show\' and for movies \'Movie\'. these are the only values in the column \'type\'"

# print(ChatGPT(createPrompt(filename=NETFLIX_FILENAME, scheme=NETFLIX_SCHEME, numeric_columns=NETFLIX_NUMERIC_COLUMNS, task=t1)))
# print(ChatGPT(createPrompt(filename=NETFLIX_FILENAME, scheme=NETFLIX_SCHEME, numeric_columns=NETFLIX_NUMERIC_COLUMNS, task=t2)))
# print(ChatGPT(createPrompt(filename=FLIGHTS_FILENAME, scheme=FLIGHTS_SCHEME, numeric_columns=FLIGHTS_NUMERIC_COLUMNS, task=t5)))
# print(ChatGPT(createPrompt(filename=FLIGHTS_FILENAME, scheme=FLIGHTS_SCHEME, numeric_columns=FLIGHTS_NUMERIC_COLUMNS, task=t6)))
# print(ChatGPT(createPrompt(filename=PLAYSTORE_FILENAME, scheme=PLAYSTORE_SCHEME, numeric_columns=PLAYSTORE_NUMERIC_COLUMNS, task=t9)))
# print(ChatGPT(createPrompt(filename=PLAYSTORE_FILENAME, scheme=PLAYSTORE_SCHEME, numeric_columns=PLAYSTORE_NUMERIC_COLUMNS, task=t10)))