# Create your views here.
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from mne.filter import filter_data


from math import log2
from xml.sax.handler import DTDHandler
import pandas as pd 
import numpy as np 
import scipy
import scipy.signal
from scipy.stats import skew
from spectrum import Criteria
from scipy.stats import kurtosis
import pickle
import matplotlib.pyplot as plt
import json


@csrf_exempt
def index(request):
    if request.method == 'POST':

        json_data = json.loads(request.body)
            
        #df = json_data.to_dict()
        train = pd.DataFrame.from_dict(json_data, orient='index')
        #train.reset_index(level=0, inplace=True)
        df = train

        data = df.to_numpy()
        eye_oc = Eye_OC_Detection()
        formatted_data = eye_oc.data_formating(df = data, Fs= 256, Ch = 4)
        X_test = eye_oc.filter_InData(X = formatted_data, fs = 256)
        y_pred = eye_oc.ml_model_predict(X_test)


        df = pd.DataFrame(y_pred, columns = ['PredictedLabel'])
        df = df.to_dict()
        print("Yine yapiyorum biseyler")

        json_object = json.dumps(df, indent=4)
        print(json_object)
    return HttpResponse(json_object,content_type="application/json")



class Eye_OC_Detection():
  def filter_InData(self, X, fs = 256):
    filtered_eeg = filter_data(data = X,sfreq = fs,l_freq = 2.0, h_freq = 30.0, method='iir',verbose=0)
    return filtered_eeg

  def data_formating(self, df, Fs= 256, Ch = 4):
      duration = int(len(df)/Fs)
      newdf = np.empty((duration,4))
      data = newdf
      once =True
      for sheet in range(Fs):
          for i in range(Ch): 
              for t in range(duration):
                  ch = df[:,i]
                  selection = np.arange(duration)
                  selection = selection*Fs+sheet
                  newdf[:,i] = ch[selection]
          data = np.dstack((data, newdf))
      data = np.delete(data, 0, axis=2)
      return data

  def ml_model_predict(self, In_data):
    with open('ml_model','rb') as f:
      mp = pickle.load(f)
    y_pred = mp.predict(In_data)
    return y_pred