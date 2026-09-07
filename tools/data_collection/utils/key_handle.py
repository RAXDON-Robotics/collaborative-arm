import traceback
import logging
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)

@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True
    
def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["start_recording"] = False
    events["finish_current_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:            
            if hasattr(key, "char"):
                if key.char.lower() == "e":
                    # finish recording current episode
                    events["finish_current_recording"] = True
                elif key.char.lower() == "r":
                    # Exiting loop and rerecord the last episode...
                    events["finish_current_recording"] = True
                    events["rerecord_episode"] = True
                elif key.char.lower() == "q":
                    # stop recording
                    events["stop_recording"] = True
                elif key.char.lower() == "s":
                    # start recording
                    events["start_recording"] = True
                    
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events