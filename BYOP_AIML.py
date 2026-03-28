import pandas as pd
import random
from sklearn.linear_model import LogisticRegression


def generate_dataset(n_samples=100):
    data = []
    for _ in range(n_samples):
        age = random.randint(15, 85)
        
        height = random.randint(135, 272)
        
        weight = random.randint(40, 250)
        
        sleep = random.randint(1, 16)
        
        pain = random.randint(0, 10)
        
        intensity = random.randint(1, 3)
        
        prev_injury = random.randint(0, 1)
        
        swelling = random.randint(0, 1)
        
        mobility_issue = random.randint(0, 1)
        
        location = random.randint(0, 9)
        
        cause = random.randint(0, 4)

        bmi = weight / ((height / 100) ** 2)

        if pain > 7 or swelling == 1 or mobility_issue == 1:
            risk = 2
        elif pain > 4 or sleep < 5 or prev_injury == 1:
            risk = 1
        else:
            risk = 0

        data.append([
            age,
            height,
            weight,
            sleep,
            pain,
            intensity,
            prev_injury,
            swelling,
            mobility_issue,
            location,
            cause,
            bmi,
            risk,
        ])

    columns = ['age', 'height', 'weight', 'sleep', 'pain', 'intensity',
        'prev_injury', 'swelling', 'mobility_issue', 'location', 'cause',
        'bmi', 'risk']
    
    return pd.DataFrame(data, columns=columns)


def train_model(df):
    x = df.drop('risk', axis=1)
    y = df['risk']
    model = LogisticRegression(max_iter=300)
    model.fit(x, y)
    return model


def get_user_input():
    try:
        age = int(input('Enter age: '))
        
        height = float(input('Enter height in cm: '))
        
        weight = float(input('Enter weight in kg: '))
        
        sleep = int(input('Enter hours of sleep: '))
        
        pain = int(input('Enter pain level (0-10): '))
        
        intensity = int(input('Enter intensity level (1-3): '))
        
        prev_injury = int(input('Enter previous injury (0 for no, 1 for yes): '))
        
        swelling = int(input('Enter swelling (0 for no, 1 for yes): '))
        
        mobility_issue = int(input('Enter mobility issue (0 for no, 1 for yes): '))
        
        location = int(input('Enter location of injury (0=head,1=neck,2=shoulder,3=knee,4=ankle,5=back,6=hip,7=wrist,8=elbow,9=chest): '))
        
        cause = int(input('Enter cause of injury (0=sports,1=gym,2=accident,3=fall,4=other): '))
    
    except ValueError:
        print('Invalid input type. Please enter numeric values where required.')
        return None

    bmi = weight / ((height / 100) ** 2)
    return [age, height, weight, sleep, pain, intensity, prev_injury, swelling, mobility_issue, location, cause, bmi]


def print_bmi(bmi):
    print(f"\nYour BMI is: {round(bmi, 2)}")
    if bmi < 18.5:
        print('You are underweight.')
    elif 18.5 <= bmi < 25:
        print('You have a normal weight.')
    elif 25 <= bmi < 30:
        print('You are overweight.')
    else:
        print('You are obese.')


def advice_gen(location, pain, swelling, mobility_issue, sleep, prev_injury):
    if pain > 7 or swelling == 1 or mobility_issue == 1 or sleep < 5 or prev_injury == 1:
        print('Consult a doctor or physiotherapist.')
    else:
        print('Safe to continue training with caution.')

    location_advice = {
        0: 'Avoid all activities.',
        1: 'Avoid neck exercises, heavy lifting.',
        2: 'Avoid shoulder exercises, heavy lifting.',
        3: 'Avoid knee exercises, running, jumping.',
        4: 'Avoid ankle exercises, running, jumping.',
        5: 'Avoid back exercises, heavy lifting.',
        6: 'Avoid hip exercises, heavy lifting.',
        7: 'Avoid wrist exercises, heavy lifting.',
        8: 'Avoid elbow exercises, heavy lifting.',
        9: 'Avoid chest exercises, heavy lifting.',
    }
    print(location_advice.get(location, 'Use common sense and consult a professional.'))


def main():
    df = generate_dataset(100)
    model = train_model(df)

    user_data = get_user_input()
    if user_data is None:
        return

    age, height, weight, sleep, pain, intensity, prev_injury, swelling, mobility_issue, location, cause, bmi = user_data

    print_bmi(bmi)

    prediction = model.predict([user_data])[0]
    label = {0: 'low', 1: 'moderate', 2: 'high'}.get(prediction, 'unknown')
    print(f"\nYou are at {label} risk of injury.")

    advice_gen(location, pain, swelling, mobility_issue, sleep, prev_injury)


if __name__ == '__main__':
    main()
