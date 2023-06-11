import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

NETFLIX_FILENAME = "netflix.tsv"
NETFLIX_SCHEME = "show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description"
NETFLIX_NUMERIC_COLUMNS = "show_id, release_year"
FLIGHTS_FILENAME = "flights.tsv"
FLIGHTS_SCHEME = "FLIGHT_NUMBER, AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY, DEPARTURE_DELAY, SCHEDULED_DEPARTURE, SCHEDULED_TIME, SCHEDULED_ARRIVAL, DAY_OF_WEEK, DAY_OF_YEAR, MONTH"
FLIGHTS_NUMERIC_COLUMNS = "DEPARTURE_DELAY"
PLAYSTORE_FILENAME = "play_store.tsv"
PLAYSTORE_SCHEME = "app_id, name, category, rating, reviews, app_size_kb, installs, type, price, content_rating, last_updated, min_android_ver"
PLAYSTORE_NUMERIC_COLUMNS = "rating, reviews, app_size_kb, installs, price"
SALARIES_FILENAME = "1.tsv"
SALARIES_SCHEME = "work_year, experience_level, employment_type, job_title, salary, salary_currency, salary_in_usd, employee_residence, remote_ratio, company_location"
SALARIES_NUMERIC_COLUMNS = "work_year, salary, salary_in_usd, remote_ratio"

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
t10 = "compare high-rated (4.7 and above) apps with low-rated (2.5 and below) apps"
t11 = "investigate the characteristics of employees in the 90th percentile of salaries (earning above 219,000 usd, according to this dataset)"

# print(ChatGPT(createPrompt(filename=NETFLIX_FILENAME, scheme=NETFLIX_SCHEME, numeric_columns=NETFLIX_NUMERIC_COLUMNS, task=t1)))
# print(ChatGPT(createPrompt(filename=NETFLIX_FILENAME, scheme=NETFLIX_SCHEME, numeric_columns=NETFLIX_NUMERIC_COLUMNS, task=t2)))
# print(ChatGPT(createPrompt(filename=FLIGHTS_FILENAME, scheme=FLIGHTS_SCHEME, numeric_columns=FLIGHTS_NUMERIC_COLUMNS, task=t5)))
# print(ChatGPT(createPrompt(filename=FLIGHTS_FILENAME, scheme=FLIGHTS_SCHEME, numeric_columns=FLIGHTS_NUMERIC_COLUMNS, task=t6)))
# print(ChatGPT(createPrompt(filename=PLAYSTORE_FILENAME, scheme=PLAYSTORE_SCHEME, numeric_columns=PLAYSTORE_NUMERIC_COLUMNS, task=t9)))
# print(ChatGPT(createPrompt(filename=PLAYSTORE_FILENAME, scheme=PLAYSTORE_SCHEME, numeric_columns=PLAYSTORE_NUMERIC_COLUMNS, task=t10)))
# print(ChatGPT(createPrompt(filename=SALARIES_FILENAME, scheme=SALARIES_SCHEME, numeric_columns=SALARIES_NUMERIC_COLUMNS, task=t11)))