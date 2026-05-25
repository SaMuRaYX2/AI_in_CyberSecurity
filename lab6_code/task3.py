# Лабораторна робота №6
# Завдання 3
# Наївний Байєс: прогноз проведення матчу

# Загальна кількість прикладів
total = 14

# Кількість класів
yes = 9
no = 5

# Апріорні ймовірності
P_yes = yes / total
P_no = no / total

# Варіант 12:
# Outlook = Overcast
# Humidity = High
# Wind = Strong

# Частотні значення з таблиці методички
P_outlook_yes = 4 / 9      # Overcast при Yes
P_outlook_no = 0 / 5       # Overcast при No

P_humidity_yes = 3 / 9     # High при Yes
P_humidity_no = 4 / 5      # High при No

P_wind_yes = 3 / 9         # Strong при Yes
P_wind_no = 3 / 5          # Strong при No

# Обчислення ненормалізованих ймовірностей
prob_yes = P_outlook_yes * P_humidity_yes * P_wind_yes * P_yes
prob_no = P_outlook_no * P_humidity_no * P_wind_no * P_no

print("Ненормалізована ймовірність Yes:", prob_yes)
print("Ненормалізована ймовірність No:", prob_no)

# Нормалізація
if prob_yes + prob_no != 0:
    norm_yes = prob_yes / (prob_yes + prob_no)
    norm_no = prob_no / (prob_yes + prob_no)
else:
    norm_yes = 0
    norm_no = 0

print("P(Yes) =", round(norm_yes, 4))
print("P(No) =", round(norm_no, 4))

if norm_yes > norm_no:
    print("Висновок: матч відбудеться.")
else:
    print("Висновок: матч не відбудеться.")