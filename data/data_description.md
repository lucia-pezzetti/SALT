# NYC Yellow Taxi Trip Data – April 2024

This dataset contains detailed records of individual trips made by yellow taxis in New York City during **April 2024**. The data is provided by the NYC Taxi and Limousine Commission (TLC).

## 📁 File Location

- `data/raw/yellow_tripdata_2024-04.csv`: Original, unmodified data downloaded from the TLC.

---

## 🧾 Data Dictionary

| Column Name               | Description |
|--------------------------|-------------|
| **VendorID**              | ID of the taxi company or provider. <br> Values: `1` or `2`. |
| **tpep_pickup_datetime** | Date and time when the trip started (local NYC time). |
| **tpep_dropoff_datetime**| Date and time when the trip ended (local NYC time). |
| **passenger_count**      | Number of passengers in the vehicle. |
| **trip_distance**        | Distance of the trip in miles. |
| **RatecodeID**           | Final rate code in effect at the end of the trip: <br> 1 = Standard rate <br> 2 = JFK <br> 3 = Newark <br> 4 = Nassau or Westchester <br> 5 = Negotiated fare <br> 6 = Group ride |
| **store_and_fwd_flag**   | Whether the trip record was stored before sending to the vendor: <br> `Y` = Stored and forwarded <br> `N` = Sent in real-time |
| **PULocationID**         | TLC Taxi Zone ID where the trip started. See NYC Taxi Zone shapefile for mapping. |
| **DOLocationID**         | TLC Taxi Zone ID where the trip ended. |
| **payment_type**         | Payment method used: <br> 1 = Credit card <br> 2 = Cash <br> 3 = No charge <br> 4 = Dispute <br> 5 = Unknown <br> 6 = Voided trip |
| **fare_amount**          | Base fare charged. |
| **extra**                | Extra charges (e.g., night surcharge, rush hour). |
| **mta_tax**              | $0.50 tax imposed by the MTA. |
| **tip_amount**           | Tip amount paid by passenger. |
| **tolls_amount**         | Total tolls paid during the trip. |
| **improvement_surcharge**| $0.30 fee added to all rides. |
| **total_amount**         | Total amount charged to the passenger. |
| **congestion_surcharge** | Additional surcharge for trips within Manhattan’s congestion zone. |
| **Airport_fee**          | Flat fee for trips to/from airports. Usually $1.25 or $2.50. |

---

## 📌 Notes

- **Time fields** are in local NYC time.
- **Location IDs** correspond to TLC Taxi Zones. You can cross-reference these using the official zone lookup or shapefile.
- **Data cleaning** may be required to remove rows with invalid values (e.g., `passenger_count = 0`).

---

## 📚 Data Description Source

The data dictionary and column descriptions are adapted from the official NYC TLC documentation:

🔗 [https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
