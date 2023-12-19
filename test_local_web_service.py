import requests

url = 'http://localhost:9696/predict'

employee_id = 'xyz-123'

employee = {
    "department": 'support',
    "promoted": 0,
    "review": 0.65,
    "projects": 2,
    "salary": 'low',
    "tenure": 2.0,
    "satisfaction": 0.20,
    "bonus": 0,
    "avg_hrs_month": 190.0
}


response = requests.post(url, json=employee).json()
print(response) 

if response['leaving_the_company'] == True:
    print('sending cautionary email to %s' % employee_id)
else:
    print('not sending supportive email to  %s' % employee_id)