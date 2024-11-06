import time


last_time = 0

while True:
    now = time.time() 
    # print(f"time: {now}")
    if now- last_time > 3 :
        last_time = now
        print(f"break: {now}")
        