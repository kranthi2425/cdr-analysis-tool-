import csv
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

def generate_fake_call_record(index):
    call_id = f"CALL{index:03}"
    caller_number = fake.phone_number()
    callee_number = fake.phone_number()
    call_start_time = fake.date_time_between(start_date='-1y', end_date='now')
    call_end_time = call_start_time + timedelta(seconds=random.randint(60, 3600))
    call_duration = int((call_end_time - call_start_time).total_seconds())
    call_type = random.choice(['incoming', 'outgoing', 'missed', 'voicemail'])
    call_notes = fake.text(max_nb_chars=100) if random.choice([True, False]) else ""
    return {
        'call_id': call_id,
        'caller_number': caller_number,
        'callee_number': callee_number,
        'call_start_time': call_start_time.isoformat(),
        'call_end_time': call_end_time.isoformat(),
        'call_duration': call_duration,
        'call_type': call_type,
        'call_notes': call_notes
    }

def generate_csv(num_records=50, filename='cdr_records.csv'):
    fieldnames = [
        'call_id',
        'caller_number',
        'callee_number',
        'call_start_time',
        'call_end_time',
        'call_duration',
        'call_type',
        'call_notes'
    ]

    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_records):
            call_record = generate_fake_call_record(i + 1)
            # Ensure call_notes is not None (though it's already handled in generate_fake_call_record)
            call_record['call_notes'] = call_record.get('call_notes', '')
            writer.writerow(call_record)

    print(f"Generated {num_records} call records in {filename}")

if __name__ == '__main__':
    generate_csv()