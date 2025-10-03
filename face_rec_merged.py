import pandas as pd
import numpy as np
import cv2
import redis
import time
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# --- Configuration and Initialization ---

# Connect to Redis Client
# Replace with your actual Redis credentials
HOSTNAME = 'redis-17081.c8.us-east-1-2.ec2.redns.redis-cloud.com'
PORTNUMBER = 17081
PASSWORD = '6M5UQmyCRI0sPQ7n23oCwAgfh48CgGYR'

try:
    r = redis.StrictRedis(host=HOSTNAME, port=PORTNUMBER, password=PASSWORD, db=0)
    r.ping()
    print("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis: {e}")
    # Handle connection error appropriately in a real application
    r = None

# Initialize InsightFace FaceAnalysis Model
# The model files will be downloaded automatically to the specified root directory.
faceapp = FaceAnalysis(name='buffalo_l', root='.', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)


# --- Database Functions ---

def retrieve_data(name):
    """Retrieves and decodes facial feature data from Redis."""
    if r is None:
        return pd.DataFrame(columns=['Name', 'Role', 'facial_feature'])

    retrieved_dict = r.hgetall(name)
    retrieved_series = pd.Series(retrieved_dict)

    # Decode keys and values
    retrieved_series.index = retrieved_series.index.map(lambda x: x.decode())
    retrieved_series = retrieved_series.map(lambda x: np.frombuffer(x, dtype=np.float32))

    retrieved_df = retrieved_series.to_frame().reset_index()
    retrieved_df.columns = ['name_role', 'facial_feature']

    # Split 'name_role' into 'Name' and 'Role'
    retrieved_df[['Name', 'Role']] = retrieved_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)

    return retrieved_df[['Name', 'Role', 'facial_feature']]


def retrieve_logs(name):
    """Retrieves and decodes attendance logs from Redis."""
    if r is None:
        return pd.DataFrame(columns=['Name', 'Role', 'Time'])

    log_list = r.lrange(name, 0, -1)
    decoded_logs = [log.decode() for log in log_list]

    # Parse log strings into a structured format
    parsed_logs = []
    for log in decoded_logs:
        try:
            name, role, timestamp = log.split('@')
            parsed_logs.append({'Name': name, 'Role': role, 'Time': timestamp})
        except ValueError:
            # Handle logs with incorrect format if necessary
            continue

    if not parsed_logs:
        return pd.DataFrame(columns=['Name', 'Role', 'Time'])

    return pd.DataFrame(parsed_logs)


# --- Core Machine Learning Algorithm ---

def ml_search_algorithm(dataframe, feature_column, test_vector, thresh=0.62):
    """
    Searches for the best match for a face embedding in the database using cosine similarity.
    """
    dataframe = dataframe.copy()

    # Ensure all embeddings are correctly shaped arrays
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

    # Calculate cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # Filter based on threshold
    data_filter = dataframe[dataframe['cosine'] >= thresh]

    if not data_filter.empty:
        # Get the person with the highest similarity score
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].idxmax()
        person_name = data_filter.loc[argmax]['Name']
        person_role = data_filter.loc[argmax]['Role']
    else:
        person_name = 'Unknown Person'
        person_role = 'Unknown Role'

    return person_name, person_role


# --- Classes for Streamlit App ---

class RegistrationForm:
    """Handles the logic for registering new users."""

    def __init__(self):
        self.sample = 0
        self.embeddings = []

    def reset(self):
        self.sample = 0
        self.embeddings = []

    def get_embedding(self, frame):
        """
        Processes a video frame to extract face embeddings for registration.
        """
        results = faceapp.get(frame, max_num=1)
        embedding = None

        if results:
            self.sample += 1
            res = results[0]
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display sample count on the frame
            text = f"Samples: {self.sample}/20"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)

            embedding = res['embedding']
            self.embeddings.append(embedding)

        return frame, embedding

    def save_data_redis(self, person_name, role):
        """
        Averages collected embeddings and saves the result to Redis.
        """
        if not self.embeddings or not person_name or not role:
            return 'failed', 'Embeddings list, name, or role is empty.'

        # Calculate the mean of all collected embeddings for a robust representation
        avg_embedding = np.mean(self.embeddings, axis=0)

        # Prepare data for Redis
        key = f"{person_name}@{role}"
        value = avg_embedding.astype(np.float32).tobytes()

        # Save to Redis Hash
        if r:
            r.hset(name='academy:register', key=key, value=value)
            self.reset()
            return 'success', f"Successfully registered {person_name}."
        else:
            return 'failed', 'Could not connect to Redis.'


class RealTimePred:
    """Handles the logic for real-time prediction and attendance logging."""

    def __init__(self):
        self.logs = {'name': [], 'role': [], 'current_time': []}
        self.log_interval = 30  # seconds
        self.last_log_time = time.time()

    def reset_dict(self):
        self.logs = {'name': [], 'role': [], 'current_time': []}

    def save_logs_redis(self):
        """Saves unique attendance logs to Redis."""
        if r is None:
            print("Cannot save logs, Redis not connected.")
            return

        dataframe = pd.DataFrame(self.logs)
        if dataframe.empty:
            return

        # Keep only the first record for each unique name in this interval
        dataframe.drop_duplicates('name', inplace=True)

        encoded_data = []
        for _, row in dataframe.iterrows():
            name, role, ctime = row['name'], row['role'], row['current_time']
            if name != 'Unknown Person':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if encoded_data:
            r.lpush('attendance:logs', *encoded_data)
            print('Saved data to Redis database.')

        self.reset_dict()

    def face_prediction(self, test_image, dataframe):
        """
        Performs face recognition on a single image and draws results.
        """
        current_time = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        results = faceapp.get(test_image)

        # Check if it's time to save logs
        if time.time() - self.last_log_time >= self.log_interval:
            self.save_logs_redis()
            self.last_log_time = time.time()

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embedding = res['embedding']

            person_name, person_role = ml_search_algorithm(
                dataframe,
                feature_column='facial_feature',
                test_vector=embedding,
                thresh=0.62
            )

            color = (0, 0, 255) if person_name == 'Unknown Person' else (0, 255, 0)  # Red for Unknown, Green for Known

            # Draw bounding box and text
            cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(test_image, person_name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            # Log the detected person for this frame
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_image