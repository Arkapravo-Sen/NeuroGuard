import time
import board
import busio
import adafruit_adxl37x
from backends import BleakBackend
from functools import partial
import mne_lsl.lsl
import backends
from muse import Muse
from multiprocessing import Process
import numpy as np
import subprocess
import torch
from torch import nn
import smtplib
from email.message import EmailMessage
import ssl
import matplotlib.pyplot as plt

#####################
#ACCELEROMETER CHUNK#
#####################

# Create I2C interface
i2c = board.I2C()

# Initialize the ADXL375
accelerometer = adafruit_adxl37x.ADXL375(i2c)

threshold = 5

def offset():
    
    """
    This function calibrates the accelerometer by returning an offset value that, when subtracted from the raw reading, results in a zero output.
    """
    
    x_off, y_off, z_off, counter = 0,0,0,0
    while counter<100:
        x,y,z = accelerometer.acceleration
        x_off += x
        y_off += y
        z_off += z
        counter += 1
        time.sleep(0.1)
    x_off /= 100
    y_off /= 100
    z_off /= 100
    
    return x_off, y_off, z_off

x_offset, y_offset, z_offset = offset()

#############
#MODEL CHUNK#
#############

# Define the model
class MyFourConvModel(nn.Module):
    def __init__(self):
        super(MyFourConvModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size= 8, stride=2, padding=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(p=0.5)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 2)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

model = MyFourConvModel()
model.load_state_dict(torch.load("/home/raspberrypi/Muse/lib64/python3.11/site-packages/best_model.pth", map_location=torch.device('cpu')))

##########################
#CONNECT TO HEADSET CHUNK#
##########################

def run_muse_lsl():
    # Use the absolute path to bash and ensure the shell is executed properly
    command = ["/bin/bash", "-i", "-c", "cd /home/raspberrypi/fina_project/lib64/python3.11/site-packages/ && python connect_to_muse.py"]

    try:
        # Running the command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        
##############
#ACQUIRE DATA#
##############

def view():
    print("Looking for an EEG stream...")
    
    # Find EEG stream
    eeg = mne_lsl.lsl.resolve_streams(stype="EEG", timeout=5)

    if len(eeg) == 0:
        raise RuntimeError("Can't find EEG stream.")
    else:
        eeg = mne_lsl.lsl.StreamInlet(eeg[0])

    print("Start acquiring EEG data.")

    # Get channel names to identify the index of AF7
    eeg_info = _view_info(eeg)
    ch_names = eeg_info["ch_names"]
    
    # Check if AF7 exists in the channel names
    if "AF7" not in ch_names:
        raise ValueError("AF7 electrode not found in the EEG stream.")
    else:
        af7_index = ch_names.index("AF7")
        print(f"AF7 electrode found at index {af7_index}.")

    # Start acquiring data and store it in an array
    data = store_data(eeg, af7_index)
    
    return data

def store_data(eeg, af7_index):
    print("Acquiring data...")

    # Create an empty list to store the AF7 data
    af7_data = []

    start_time = time.time()
    
    # Start recording for 10 seconds
    while time.time() - start_time < 10:
        # Pull EEG data
        samples, timestamp = eeg.pull_chunk(timeout=1, max_samples=100)  # Renamed 'time' to 'timestamp'

        if len(samples) > 0:
            # Extract AF7 data (using the index of AF7)
            af7_samples = samples[:, af7_index]
            
            # Append AF7 data to the list
            af7_data.extend(af7_samples)
    
    # Convert the list to a NumPy array
    af7_data = [[af7_data[:2501]]]
    af7_data_array = np.array(af7_data)
    af7_data_array = af7_data_array/1000000

    return af7_data_array

def _view_info(inlet):
    """Get info from stream"""
    inlet.open_stream()

    info = {}  # Initialize a container
    info["info"] = inlet.get_sinfo()
    info["description"] = info["info"].desc

    info["window"] = 10  # 10-second window showing the data.
    info["sfreq"] = info["info"].sfreq
    info["n_samples"] = int(info["sfreq"] * info["window"])
    info["ch_names"] = info["info"].get_channel_names()
    info["n_channels"] = len(info["ch_names"])
    info["inlet"] = inlet
    return info


###############
#SENDING EMAIL#
###############

email_sender = 'sender@gmail.com'
email_reciever = 'receiver@gmail.com'
email_psswd = 'psswd'

subject = 'Player Report'

try:
    if __name__ == "__main__":
        p1 = Process(target=run_muse_lsl)
        p1.start()
        time.sleep(20)
        
        last_view_time = time.time()
        
        while True:
            if time.time() - last_view_time >= 30:
                print("Looking for an EEG stream...")
                # Find EEG stream
                eeg = mne_lsl.lsl.resolve_streams(stype="EEG", timeout=5)
                eeg = mne_lsl.lsl.StreamInlet(eeg[0])
                current_time = time.time()
                while time.time() - current_time < 0.1:
                    # Pull EEG data
                    samples, timestamp = eeg.pull_chunk(timeout=1, max_samples=100)  # Renamed 'time' to 'timestamp'
                    last_view_time = time.time()

            x, y, z = accelerometer.acceleration
            x = (x - x_offset) / 9.81
            y = (y - y_offset) / 9.81
            z = (z - z_offset + 9.81) / 9.81
            
            print(f"X: {x:.2f} G, Y: {y:.2f} G, Z: {z:.2f} G")
            
            if x >= threshold or x <= -threshold or y >= threshold or y <= -threshold or z >= threshold or z <= -threshold:
                print("Player gone through extreme stress")
                data_array = view()
                print(data_array.shape)
                data_array = torch.tensor(data_array, dtype=torch.float32)
                with torch.no_grad():
                    output = model(data_array)
                    softmaxing = nn.Softmax(dim=1)
                    output = softmaxing(output)
                    
                data_array = data_array.squeeze()
                data_array = data_array.squeeze()
                plt.figure(figsize=(19.2,10.8))
                plt.plot(data_array)
                plt.xlim(-10,2511)
                plt.ylim(-0.01,0.01)
                plt.savefig('data.png')
                if output[0][0] > output[0][1]:
                        print("Model Prediction: Normal")
                        # The body of the email
                        player_impact = f"""Your Player seems to have gone through a lot of stress.
                        We have detected g forces beyond safe levels. 
                        After running the brain data through the model, our prediction is that the player seems to be fine. 
                        Please consider seeing a doctor if the player shows abnormal behavior."""

                        # Create the email message
                        em = EmailMessage()
                        em['From'] = email_sender
                        em['To'] = email_reciever
                        em['Subject'] = subject
                        em.set_content(player_impact)

                        # Attach the image
                        file_path = '/home/raspberrypi/fina_project/lib64/python3.11/site-packages/data.png'

                        try:
                            with open(file_path, 'rb') as file:
                                em.add_attachment(file.read(), maintype='image', subtype='png', filename='plot.png')
                        except FileNotFoundError:
                            print(f"Error: The file at {file_path} was not found.")
                            exit()

                        # Set up SSL context and send the email
                        context = ssl.create_default_context()
                        try:
                            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                                smtp.login(email_sender, email_psswd)
                                smtp.sendmail(email_sender, email_reciever, em.as_string())
                            print("Email sent successfully")
                        except Exception as e:
                            print(f"Error sending email: {e}")
                else:
                        print("Model prediction: Abnormal")
                        player_impact = f"""Your Player seems to have gone through a lot of stress.
                        We have detected g forces beyond safe levels.
After running the brain data through the model, our prediction is that the player seems to have some sort of brain damage or abnormality. 
Please consider seeing a doctor if the player shows abnormal behavior."""

                        # Create the email message
                        em = EmailMessage()
                        em['From'] = email_sender
                        em['To'] = email_reciever
                        em['Subject'] = subject
                        em.set_content(player_impact)

                        # Attach the image
                        file_path = '/home/raspberrypi/fina_project/lib64/python3.11/site-packages/data.png'

                        try:
                            with open(file_path, 'rb') as file:
                                em.add_attachment(file.read(), maintype='image', subtype='png', filename='plot.png')
                        except FileNotFoundError:
                            print(f"Error: The file at {file_path} was not found.")
                            exit()

                        # Set up SSL context and send the email
                        context = ssl.create_default_context()
                        try:
                            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                                smtp.login(email_sender, email_psswd)
                                smtp.sendmail(email_sender, email_reciever, em.as_string())
                            print("Email sent successfully")
                        except Exception as e:
                            print(f"Error sending email: {e}")
            time.sleep(0.01)
except KeyboardInterrupt:
    print("Exiting...")
