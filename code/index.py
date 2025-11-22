# PROBABILITY, GUESSING GAME

import random

number = random.randint(1,100)

attempts = 0
max_attempts = 5
while attempts < max_attempts:
    guess = int(input("enter your guess (1-100):"))
    attempts += 1
    if guess < number:
        print("too low!")
    elif guess > number:
        print("too high!")
    else:
        print(f"congratulations! you guessed the number {number} in {attempts} attempts.")
        break
else:
    print(f"you used all attempts. the number was {number}. better luck next time!")
    