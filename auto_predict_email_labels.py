import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only

import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Terminal color formatting
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Load the pre-trained model and tokenizer
model = load_model('Models/LSTM_no1n2.h5')
with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Labels for classification categories
labels = ['Weather/Natural', 'Sent Mail', 'Random/NA', 'Financial/Logistics', 'Related to Other People',
          'Places', 'Legal', 'Business', '2-Letter/Random', 'Other Firms', 'HR/Recruiting/MBA']

max_length = 250

# Sample test emails
test_emails = [
    "Please find attached the latest profit and loss statement for Q4. Let me know if you need clarification.",
    "We need to reschedule the gas pipeline maintenance to next Monday due to unexpected delays at the Houston site.",
    "We have shortlisted candidates for the open analyst role. Please review their resumes and confirm interview availability.",
    "Please advise if the attached contract complies with the revised FERC regulations effective from this quarter.",
    "I will be in Chicago next week. Let’s try to meet and finalize the pricing model for the new deal.",
    "Due to heavy storms, our field offices will remain closed tomorrow. Employees should work remotely."
]

# Output header
print("\n" + "="*80)
print(f"{bcolors.BOLD}{bcolors.OKCYAN}Running Automated Email Classification...{bcolors.ENDC}")
print("="*80)

# Process and predict for each email
for idx, text in enumerate(test_emails, 1):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(padded, verbose=0)
    pred_label = labels[np.argmax(pred)]

    print(f"\n{bcolors.BOLD}Email #{idx}:{bcolors.ENDC} {text}")
    print(f"{bcolors.OKGREEN}Predicted Label:{bcolors.ENDC} {bcolors.BOLD}{pred_label}{bcolors.ENDC}")

print("\n" + "="*80)
print(f"{bcolors.OKBLUE}All emails classified successfully.{bcolors.ENDC}")
print("="*80)
