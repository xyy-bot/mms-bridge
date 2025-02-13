import threading
import serial
import time
import cv2
import os
import matplotlib.pyplot as plt
import csv
import sys

# Arduino Reader Class (intended 1 kHz sampling)
class ArduinoReader(threading.Thread):
    def __init__(self, serial_port, baud_rate, log_dir, experiment_name, sampling_interval):
        super().__init__()
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.sampling_interval = sampling_interval  # intended: 0.001 sec for 1 kHz
        self.data = []  # List to hold (timestamp, force) tuples
        self.running = True
        self.serial_port = None
        # We'll set first_sample_time when we get the first valid sample
        self.first_sample_time = None

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"{self.experiment_name}.csv")

        # Open the serial port
        try:
            self.serial_port = serial.Serial(serial_port, baud_rate, timeout=0.01)
        except Exception as e:
            print(f"Failed to open serial port {serial_port}: {e}")
            sys.exit(1)

    def run(self):
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "ForceReading"])

            # Wait for the first valid reading to establish time zero.
            while self.running and self.first_sample_time is None:
                try:
                    if self.serial_port and self.serial_port.is_open:
                        raw_data = self.serial_port.readline()
                        if not raw_data:
                            continue
                        try:
                            force_value = float(raw_data.decode('utf-8').strip())
                        except Exception:
                            continue
                        # Set the first sample time now and log the first sample with timestamp 0.
                        self.first_sample_time = time.time()
                        timestamp = 0.0
                        self.data.append((timestamp, force_value))
                        writer.writerow([timestamp, force_value])
                        print(f"First valid sample: {force_value} at time 0")
                    else:
                        break
                except Exception as e:
                    print(f"Arduino Error (first sample): {e}")

            # Start sample_count from 1 because the first sample is already logged.
            sample_count = 1

            # Main loop: use the ideal timestamp based on sample count.
            while self.running:
                # Calculate the intended absolute time for this sample.
                intended_time = self.first_sample_time + sample_count * self.sampling_interval

                try:
                    if self.serial_port and self.serial_port.is_open:
                        raw_data = self.serial_port.readline()
                        if not raw_data:
                            continue
                        try:
                            force_value = float(raw_data.decode('utf-8').strip())
                        except Exception:
                            continue
                    else:
                        break
                except Exception as e:
                    print(f"Arduino Error during reading: {e}")
                    continue

                # Instead of using the actual elapsed time, assign an ideal timestamp.
                timestamp = sample_count * self.sampling_interval
                self.data.append((timestamp, force_value))
                writer.writerow([timestamp, force_value])
                sample_count += 1

                # Calculate how long to sleep until the next intended sample time.
                sleep_time = (intended_time + self.sampling_interval) - time.time()
                if sleep_time > 0 and self.running:
                    try:
                        time.sleep(sleep_time)
                    except Exception as e:
                        print(f"Arduino Sleep error: {e}")
                        break

    def stop(self):
        self.running = False
        time.sleep(self.sampling_interval)  # Allow thread to exit cleanly
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("Arduino serial port closed.")

    def plot_data(self):
        if not self.data:
            print("No data available for plotting.")
            return
        try:
            timestamps, force_readings = zip(*self.data)
            plt.figure()
            plt.plot(timestamps, force_readings, label="Force Reading")
            plt.xlabel("Time (s)")
            plt.ylabel("Force")
            plt.title(f"Force Data - {self.experiment_name}")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_file = os.path.join(self.log_dir, f"{self.experiment_name}.png")
            plt.savefig(plot_file)
            plt.close('all')
            print(f"Arduino plot saved to: {plot_file}")
        except Exception as e:
            print(f"Error while plotting: {e}")

# GelSight Mini Camera Reader Class (intended 20 Hz sampling)
class GelSightReader(threading.Thread):
    def __init__(self, video_source, log_dir, experiment_name, sampling_interval):
        super().__init__()
        self.video_source = video_source
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.sampling_interval = sampling_interval  # intended: 0.05 sec for 20 Hz
        self.running = True
        self.experiment_start_time = None  # To be set externally

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        self.video_file = os.path.join(self.log_dir, f"{self.experiment_name}.mov")
        self.video_stream = cv2.VideoCapture(self.video_source)

        # Force the resolution to 640x480.
        self.video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        # Allow the camera to adjust
        time.sleep(0.5)

        # For output, we force the frame size to be 640x480.
        self.frame_size = (480, 640)
        print(f"Forced frame size: {self.frame_size}")

        # Set FPS based on the sampling interval
        self.fps = int(1 / self.sampling_interval)

        # Initialize video writer (using mp4v codec for .mov files)
        self.video_writer = cv2.VideoWriter(
            self.video_file,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            self.frame_size
        )

        # Initialize last_frame to store the final frame
        self.last_frame = None

    def run(self):
        if self.experiment_start_time is None:
            self.experiment_start_time = time.time()
        frame_number = 0

        while self.running:
            intended_time = self.experiment_start_time + frame_number * self.sampling_interval
            current_time = time.time()
            sleep_time = intended_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            ret, frame = self.video_stream.read()
            if ret:
                # Rotate and resize the frame
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = cv2.resize(frame, self.frame_size)

                # Save the current frame as the last frame captured.
                self.last_frame = frame

                self.video_writer.write(frame)
                frame_number += 1
            else:
                print("Failed to read frame from video source.")
                # You can choose to break or continue here.
        self.stop()

    def stop(self):
        if not self.running:
            return  # Prevent multiple stop calls
        self.running = False
        time.sleep(self.sampling_interval)  # Allow loop to exit cleanly
        if self.video_stream.isOpened():
            self.video_stream.release()
        self.video_writer.release()
        print(f"GelSight video saved to: {self.video_file}")


# Main function
if __name__ == "__main__":
    base_filename = "aluminum cube_0"

    # Create directories if they do not exist
    try:
        os.makedirs('Force_data', exist_ok=True)
        os.makedirs('Video_data', exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)

    # Define a common experiment start time
    experiment_start_time = time.time()

    # Setup Arduino Reader (1 kHz)
    arduino_port = "/dev/ttyACM0"
    arduino_baud_rate = 115200
    arduino_sampling_interval = 0.001  # intended 1 kHz
    arduino_reader = ArduinoReader(
        serial_port=arduino_port,
        baud_rate=arduino_baud_rate,
        log_dir='Force_data',
        experiment_name=base_filename,
        sampling_interval=arduino_sampling_interval
    )
    arduino_reader.experiment_start_time = experiment_start_time

    # Setup GelSight Reader (20 Hz)
    gelsight_source = 8  # change to your correct camera index
    gelsight_sampling_interval = 0.05  # intended 20 Hz
    gelsight_reader = GelSightReader(
        video_source=gelsight_source,
        log_dir='Video_data',
        experiment_name=base_filename,
        sampling_interval=gelsight_sampling_interval
    )
    gelsight_reader.experiment_start_time = experiment_start_time

    try:
        arduino_reader.start()
        gelsight_reader.start()

        print("Logging started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping logging...")
    finally:
        print("Stopping threads...")
        try:
            arduino_reader.stop()
            print("ArduinoReader stopped.")
        except Exception as e:
            print(f"Error stopping ArduinoReader: {e}")

        try:
            gelsight_reader.stop()
            print("GelSightReader stopped.")
        except Exception as e:
            print(f"Error stopping GelSightReader: {e}")

        # Join threads
        try:
            arduino_reader.join()
            print("ArduinoReader joined.")
        except Exception as e:
            print(f"Error joining ArduinoReader: {e}")

        try:
            gelsight_reader.join()
            print("GelSightReader joined.")
        except Exception as e:
            print(f"Error joining GelSightReader: {e}")

        # Plot the Arduino data for visual verification
        try:
            print(f"Collected {len(arduino_reader.data)} data points from Arduino.")
            arduino_reader.plot_data()
        except Exception as e:
            print(f"Error while plotting Arduino data: {e}")

        print("Logging stopped.")
