import msvcrt
from fsae.envs import RandomTrackEnv

def get_key(): 
    key = None
    # Check if there's data available to read from sys.stdin without blocking
    if msvcrt.kbhit():
        key = msvcrt.getch().decode()
    return key

def main():
    env = RandomTrackEnv(render_mode="tp_camera")
    done = False
    print("started")
    action = [0,0]

    while not done:
        key = get_key() # This is a non-blocking call now
        if key is not None:
            if key == "q":  # Move forward-left
                action = [1, -0.6]
            elif key == "w":  # Move forward
                action = [1, 0.0]
            elif key == "e":  # Move forward-right
                action = [1, 0.6]
            elif key == "a":  # Steer left (no throttle)
                action = [0, -0.6]
            elif key == "s":  # No steer / throttle
                action = [0, 0]
            elif key == "d":  # Steer right (no throttle)
                action = [0, 0.6]
            elif key == "z":  # Move back-left
                action = [-1, -0.6]
            elif key == "x":  # Move back
                action = [-1, 0.0]
            elif key == "c":  # Move back-right
                action = [-1, 0.6]
            elif key == "p":  # quit application
                break
            print(f"{key} recognised. Action: ", action)
        
        #If no key is pressed, pass the same command
        state_, reward, done, _info = env.step(action)
        _ = env.render("fp_camera")
        print("tick")

    env.close()

if __name__ == "__main__":
    main()