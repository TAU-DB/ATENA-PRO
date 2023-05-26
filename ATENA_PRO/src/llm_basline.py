import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

NETFLIX = "dataset file name: \'netflix.tsv\'\ndataset scheme: show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description"
FLIGHTS = "dataset file name: \'flights.tsv\'\ndataset scheme: flight_id, airline, origin_airport, destination_airport, flight_number, delay_reason, departure_delay, scheduled_trip_time, scheduled_departure, scheduled_arrival, day_of_week, day_of_month, month"
PLAYSTORE = "dataset file name: \'play_store.tsv\'\ndataset scheme: app_id, name, category, rating, reviews, app_size_kb, installs, type, price, content_rating, last_updated, min_android_ver"
QUERY = "write pandas code to answer the following task for the dataset provided above:"


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


def createPrompt(dataset, task):
    print("--------------------------------")
    print(task)
    return f'{dataset}\n\n{QUERY}\n{task}'


t1 = "find a country with different, atypical viewing habits, compared to the rest of the world"
t2 = "investigate the properties of “successful” TV shows that have more than one season"
t6 = "explore different reasons of flight delays"
t5 = "show interesting properties of flights in the summer"
t10 = "compare high-rated apps with low-rated apps"
t9 = "show properties of apps with at least one million installs"

print(ChatGPT(createPrompt(NETFLIX, t1)))
# print(ChatGPT(createPrompt(NETFLIX, t2)))
# print(ChatGPT(createPrompt(FLIGHTS, t6)))
# print(ChatGPT(createPrompt(FLIGHTS, t5)))
# print(ChatGPT(createPrompt(PLAYSTORE, t10)))
# print(ChatGPT(createPrompt(PLAYSTORE, t9)))
