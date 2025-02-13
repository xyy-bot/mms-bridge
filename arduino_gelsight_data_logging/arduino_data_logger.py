import threading
import serial
import time
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
        self.sampling_interval = sampling_interval  # 0.001 sec for 1 kHz sampling
        self.data = []  # List to hold (timestamp, force) tuples
        self.running = True
        self.serial_port = None
        self.first_sample_time = None  # To mark time zero

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
                        # Set the first sample time and log the first sample with timestamp 0.
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

            # Main loop: assign ideal timestamps based on sample count.
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

                # Use the ideal timestamp rather than the actual elapsed time.
                timestamp = sample_count * self.sampling_interval
                self.data.append((timestamp, force_value))
                writer.writerow([timestamp, force_value])
                sample_count += 1

                # Sleep until the next intended sample time.
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
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps, force_readings, label="Force Reading")
            plt.xlabel("Time (s)")
            plt.ylabel("Force")
            plt.title(f"Force Data - {self.experiment_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_file = os.path.join(self.log_dir, f"{self.experiment_name}.png")
            plt.savefig(plot_file)
            plt.show()  # Display the plot interactively
            plt.close('all')
            print(f"Force plot saved to: {plot_file}")
        except Exception as e:
            print(f"Error while plotting: {e}")

if __name__ == "__main__":
    # Set parameters for the experiment. irregular_deformation_4
    experiment_name = "test_connection"
    log_dir = "Force_for_vqvae"
    arduino_port = "/dev/ttyACM0"  # Adjust this to your system's port
    baud_rate = 115200
    sampling_interval = 0.001  # 1 kHz sampling

    # Ensure the logging directory exists.
    os.makedirs(log_dir, exist_ok=True)

    # Setup the ArduinoReader.
    arduino_reader = ArduinoReader(
        serial_port=arduino_port,
        baud_rate=baud_rate,
        log_dir=log_dir,
        experiment_name=experiment_name,
        sampling_interval=sampling_interval
    )

    try:
        print("Starting force data logging. Press Ctrl+C to stop.")
        arduino_reader.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping force data logging.")
    finally:
        arduino_reader.stop()
        arduino_reader.join()
        print("Force data logging stopped.")
        print(f"Collected {len(arduino_reader.data)} data points.")
        arduino_reader.plot_data()
