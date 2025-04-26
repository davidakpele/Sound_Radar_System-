
from matplotlib import pyplot as plt
from sonar_simulation import RadarSystem


if __name__ == '__main__':
    radar = RadarSystem()
    try:
        radar.start()
        while True:
            plt.pause(0.1)
    except KeyboardInterrupt:
        radar.audio.stop()
        radar.visualizer.stop_animation() 
        print("Radar system stopped.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close('all')
