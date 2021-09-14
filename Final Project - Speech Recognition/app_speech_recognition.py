#Imports
import streamlit as st
import numpy as np
import speech_recognition as s_r
import librosa   #for audio processing
import matplotlib.pyplot as plt
from scipy.io import wavfile #for audio processing
from scipy.io.wavfile import write
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


#----------------------
# functions definition
#----------------------
def file_selector():
    file = st.file_uploader('Upload wav file',type='wav')
    return file

def record_voice():
    r = s_r.Recognizer()
    my_mic = s_r.Microphone(device_index=1) #my device index is 1, you have to put your device index
    with my_mic as source:
        r.adjust_for_ambient_noise(source) #reduce noise
        audio = r.listen(source) #take voice input from the microphone
    # write audio to a WAV file
    audio_test  = 'test.wav'
    with open(audio_test, 'wb') as f:
        f.write(audio.get_wav_data())
    return audio_test, r

#Cut audio file to 1second (assume that person starting to speaking at 3ms)
def cut_record_voice(audio_test,r):
    with s_r.AudioFile(audio_test) as source:
        audio_short = r.record(source, offset=0.3, duration=1.3) 
    audio_cut = 'test_proc.wav'
    with open(audio_cut, 'wb') as f:
        f.write(audio_short.get_wav_data())
    return audio_cut

#force audio file to be 1second and fix lenght if inferior to 8000 even after resample to 8000
def pre_process(audio):
    samples, sample_rate = librosa.load(audio, duration=1)
    samples_8k = librosa.core.resample(samples, sample_rate, 8000)
    shape_audio = samples_8k.shape[0]
    if shape_audio < 8000:
        samples_8k = librosa.util.fix_length(samples_8k, 8000, mode='edge')
    return samples_8k

#transform array of sound to play it
def play_array(samples_8k):
    scaled = np.int16(samples_8k/np.max(np.abs(samples_8k)) * 32767)
    write('test_reverse.wav', 8000, scaled)
    return 'test_reverse.wav'

#Create and display graph with file as input
def create_graph(audiofile):
    samples, sample_rate = librosa.load(audiofile, sr = 16000)
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave')
    ax1.set_xlabel('time')
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, 16000/len(samples), 16000), samples)
    return st.pyplot(fig)

#Create and display graph with array as input
def create_graph_arr(samples):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Pre-processign audio spectrogram')
    ax1.set_xlabel('time')
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, 8000/len(samples), 8000), samples)
    return st.pyplot(fig)

#Display spectrogram graph with librosa - not working
def spectro(x):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

#To compare with google librairy - if times 
@st.cache
def recognition(audio):
    try:
        # using google speech recognition
        st.write('Text: '+r.recognize_google(audio, language = 'fr-FR'))
    except:
        st.write('Sorry, I did not get that')  

 #Define the function that predicts text for the given audio:
def predict_audio(audio_pred):
    labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    le = LabelEncoder()
    y=le.fit_transform(labels)
    classes= list(le.classes_)
    model=load_model('models/model.h5')
    prob=model.predict(audio_pred.reshape(1,8000,1))   
    index=np.argmax(prob[0])
    return classes[index], max(prob[0])       

def main():

    audiofile = ''
    sample = ''
    proceed = ''
    score_txt = ''
    title = 'Speech recognition for connected object'

    #------------
    #section side
    #------------
    menu_one = 'Purpose'
    menu_sec = 'Play'
    menu_third = 'About'
    optionSide = st.sidebar.selectbox(
        'Contents',
        [menu_one, menu_sec, menu_third])


    #---------
    #section 0
    #---------
    st.title(title)


    #---------
    #section 1
    #---------
    if (optionSide == menu_one):

        st.subheader(menu_one)
        st.write("Construction of a voice recognition system that understands simple voice commands. \n The idea is simple: Faced with so many connected objects around us: screens, TVs, refrigerators, cars etc, the idea to get out of the dependence of screens is this voice recognition system which understands simple voice commands.")
        imageIntro = Image.open('images/bg.png')
        st.image(imageIntro)

    #---------
    #section 2
    #---------
    elif (optionSide == menu_sec):

        chosen = st.selectbox(
            menu_sec,
            ['','Upload File','Recording now'])

        if chosen == 'Recording now':
            imgMic = Image.open('images/mic.png')
            st.image(imgMic,width=50, caption='Recording ...')
            audiofile, r = record_voice()
            left_column, right_column = st.columns(2)
            left_column.write('Initial :')
            left_column.audio(audiofile)
            audiofile_cut = cut_record_voice(audiofile, r)
            samples_arr = pre_process(audiofile_cut)
            right_column.write('Pre-processing :')
            audiofile_proc = play_array(samples_arr)
            right_column.audio(audiofile_proc)
            create_graph_arr(samples_arr)
            proceed = 'TRUE'

        elif chosen == 'Upload File':
            audiofile = file_selector()
            if (audiofile != '') and (audiofile != None):
                left_column, right_column = st.columns(2)
                left_column.write('Initial :')
                left_column.audio(audiofile)
                samples_arr = pre_process(audiofile)
                right_column.write('Pre-process :')
                audiofile_proc = play_array(samples_arr)
                right_column.audio(audiofile_proc)
                create_graph_arr(samples_arr)
                proceed = 'TRUE'

        else:
            proceed = 'FALSE'


        if proceed == 'TRUE':

            st.info("starting recognition ... ")
            #my_bar = st.progress(20)

            try:
                reco, index = predict_audio(samples_arr)           
                score_pred = int(index*100)
                if score_pred > 75:
                    imgMic = Image.open('images/very_good.png')
                    st.success(reco)
                elif score_pred < 75 and score_pred > 50:
                    imgMic = Image.open('images/good.png')
                    st.success(reco)
                elif score_pred < 50 and score_pred > 35:
                    imgMic = Image.open('images/neutral.png')
                    st.success(reco)
                elif score_pred < 35:
                    imgMic = Image.open('images/bad.png')
                    st.error('Sorry, can you try again')
                st.image(imgMic,width=30)

            except:
                st.error('Sorry, I did not get that')

           # my_bar.progress(100)


    #---------
    #section 3
    #---------
    elif (optionSide == menu_third):

        st.subheader(menu_third)
        st.write('')
        st.markdown('__Dataset for training__ : 65000 audio files of 1second, 30 word utterances by 1000 of different people')
        st.markdown('__Labels used__ : yes, no, up, down, left, right, on, off, stop, go')
        st.markdown('__Model Accuracy__ : 94%')
        st.write('')
        st.write('')
        st.text('by Naïmé D., Fabrice T., Soufyen T.')

if __name__ == '__main__':
    main()
    #app.run_server(debug=True)
    