import threading
import serial
import time
import cv2
import datetime
import os
import matplotlib.pyplot as plt
import csv
import sys


class ArduinoReader(threading.Thread):
    def __init__(self, serial_port, baud_rate, log_dir, experiment_name, sampling_interval):
        super().__init__()
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.sampling_interval = sampling_interval
        self.data = []
        self.running = True
        self.serial_port = None  # Initialize to None

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
            writer.writerow(["Timestamp", "ForceReading"])
            while self.running:
                try:
                    if self.serial_port and self.serial_port.is_open:
                        data = self.serial_port.readline().decode('utf-8').strip()
                        if data and data.replace('.', '', 1).isdigit():
                            timestamp = datetime.datetime.now().isoformat()
                            self.data.append((timestamp, float(data)))  # Save for plotting
                            writer.writerow([timestamp, data])
                            # print(f"Arduino: {timestamp}, {data}")
                except Exception as e:
                    print(f"Arduino Error: {e}")
                time.sleep(self.sampling_interval)

    def stop(self):
        self.running = False
        time.sleep(self.sampling_interval)  # Allow thread loop to exit cleanly
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("Arduino serial port closed.")

    def plot_data(self):
        if not self.data:
            print("No data available for plotting.")
            return

        try:
            # Print the size of the dataset
            # print(f"Plotting {len(self.data)} data points.")

            # Extract timestamps and force readings
            timestamps, force_readings = zip(*self.data)

            # Convert timestamps to seconds relative to the start time
            start_time = datetime.datetime.fromisoformat(timestamps[0])
            elapsed_seconds = [
                (datetime.datetime.fromisoformat(ts) - start_time).total_seconds()
                for ts in timestamps
            ]

            # Plot the data
            plt.figure()
            plt.plot(elapsed_seconds, force_readings, label="Force Reading")
            plt.xlabel("Time (s)")  # X-axis in seconds
            plt.ylabel("Force")
            plt.title(f"Force Data - {self.experiment_name}")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot
            plot_file = os.path.join(self.log_dir, f"{self.experiment_name}.png")
            plt.savefig(plot_file)
            # print(f"Arduino plot saved to: {plot_file}")
            plt.close('all')  # Ensure all figures are closed

        except Exception as e:
            print(f"Error while plotting: {e}")


# GelSight Mini Camera Reader Class
class GelSightReader(threading.Thread):
    def __init__(self, video_source, log_dir, experiment_name, sampling_interval):
        super().__init__()
        self.video_source = video_source
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.sampling_interval = sampling_interval
        self.running = True

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Set the video file to .mov format
        self.video_file = os.path.join(self.log_dir, f"{self.experiment_name}.mov")
        self.video_stream = cv2.VideoCapture(self.video_source)

        # Determine FPS and frame size
        self.fps = int(1 / self.sampling_interval)
        self.frame_size = (
            int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640,
            int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480,
        )

        # Initialize the video writer with mp4v codec for .mov format
        self.video_writer = cv2.VideoWriter(
            self.video_file,
            cv2.VideoWriter_fourcc(*'mp4v'),  # Codec for .mov files
            self.fps,
            self.frame_size
        )

    def run(self):
        try:
            while self.running:
                try:
                    ret, frame = self.video_stream.read()
                    if ret:
                        self.video_writer.write(frame)
                        # print(f"GelSight: Frame written to video")
                    time.sleep(self.sampling_interval)
                except Exception as e:
                    print(f"GelSight Error in run loop: {e}")
        except Exception as e:
            print(f"Unhandled exception in GelSightReader.run: {e}")
        finally:
            self.stop()

    def stop(self):
        if not self.running:
            return  # Prevent multiple stop calls
        self.running = False
        time.sleep(self.sampling_interval)  # Allow thread loop to exit cleanly
        if self.video_stream.isOpened():
            self.video_stream.release()
        self.video_writer.release()
        print(f"GelSight video saved to: {self.video_file}")




# Main Function
if __name__ == "__main__":
    base_filename = "soft_roller"

    # Ensure directories exist
    try:
        os.makedirs('Force_data', exist_ok=True)
        os.makedirs('Video_data', exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)

    # Arduino Reader Setup

    arduino_port = "/dev/ttyACM0"
    arduino_baud_rate = 115200
    arduino_sampling_interval = 0.001  # 1 kHz
    arduino_reader = ArduinoReader(
        serial_port=arduino_port,
        baud_rate=arduino_baud_rate,
        log_dir='Force_data',
        experiment_name=base_filename,
        sampling_interval=arduino_sampling_interval
    )

    # GelSight Mini Setup
    gelsight_source = 2  # Video source index
    gelsight_sampling_interval = 0.05  # 20 Hz
    gelsight_reader = GelSightReader(
        video_source=gelsight_source,
        log_dir='Video_data',
        experiment_name=base_filename,
        sampling_interval=gelsight_sampling_interval
    )

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

        # Plot Arduino data
        try:
            print(f"Collected {len(arduino_reader.data)} data points from Arduino.")
            arduino_reader.plot_data()
        except Exception as e:
            print(f"Error while plotting Arduino data: {e}")

        print("Logging stopped.")


