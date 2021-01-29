import random
from flask import Flask, request
from pymessenger.bot import Bot

from keras.models import load_model, Model
from keras.layers import Input, LSTM, Dense
import re
import numpy as np

app = Flask(__name__)       # Initializing our Flask application
ACCESS_TOKEN = 'EAAJvluWmu6sBAKa8cjKqgPg3gG3ZAQ3pTv5gHSxrkD2gzQM99ZCCOR0VOexb7xLnKE9cZBQZB9WGa7XqE2ZB7FlLCTDwTItRndRsUWAwoJIZA8nwo4xraErSKZBX15BdoPPdIOvLt04KPAVv8PKruUF0qbd5uKiNh4QQkzymnpYYQZDZD'
VERIFY_TOKEN = 'EAAJvluWmu6sBAKa8cjKqgPg3gG3ZAQ3pTv5gHSxrkD2gzQM99ZCCOR0VOexb7xLnKE9cZBQZB9WGa7XqE2ZB7FlLCTDwTItRndRsUWAwoJIZA8nwo4xraErSKZBX15BdoPPdIOvLt04KPAVv8PKruUF0qbd5uKiNh4QQkzymnpYYQZDZDAS'
bot = Bot(ACCESS_TOKEN)

# Importing standard route and two requst types: GET and POST.
# We will receive messages that Facebook sends our bot at this endpoint
@app.route('/', methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        # Before allowing people to message your bot Facebook has implemented a verify token
        # that confirms all requests that your bot receives came from Facebook.
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    # If the request was not GET, it  must be POSTand we can just proceed with sending a message
    # back to user
    else:
            # get whatever message a user sent the bot
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if message.get('message'):
                    # Facebook Messenger ID for user so we know where to send response back to
                    recipient_id = message['sender']['id']
                    if message['message'].get('text'):
                        response_sent_text = get_message(message['message'].get('text'))
                        send_message(recipient_id, response_sent_text)
                    # if user send us a GIF, photo, video or any other non-text item
                    if message['message'].get('attachments'):
                        response_sent_text = "gif"
                        send_message(recipient_id, response_sent_text)
    return "Message Processed"


def verify_fb_token(token_sent):
    # take token sent by Facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


def get_message(mensaje):
    return generate_response(mensaje)


# Uses PyMessenger to send response to the user
def send_message(recipient_id, response):
    # sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"


def load_full_model(training_model):
    
    data_path = "preguntas7.txt"
    data_path2 = "respuestas7.txt"
    with open(data_path.encode('utf-8'), 'r') as f:
      lines = f.read().split('\n')
    with open(data_path2.encode('utf-8'), 'r') as f:
      lines2 = f.read().split('\n')
    lines = [re.sub(r"\[\w+\]",'hi',line) for line in lines]
    lines = [" ".join(re.findall(r"\w+",line)) for line in lines]
    lines2 = [re.sub(r"\[\w+\]",'',line) for line in lines2]
    lines2 = [" ".join(re.findall(r"\w+",line)) for line in lines2]
    # Grouping lines by response pair
    pairs = list(zip(lines,lines2))
    #random.shuffle(pairs)

    input_docs = []
    target_docs = []
    input_tokens = set()
    target_tokens = set()
    for line in pairs[:]:
      input_doc, target_doc = line[0], line[1]
      # Appending each input sentence to input_docs
      input_docs.append(input_doc)
      # Splitting words from punctuation  
      target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
      # Redefine target_doc below and append it to target_docs
      target_doc = '<START> ' + target_doc + ' <END>'
      target_docs.append(target_doc)
    
      # Now we split up each sentence into words and add each unique word to our vocabulary set
      for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
          input_tokens.add(token)
      for token in target_doc.split():
        if token not in target_tokens:
          target_tokens.add(token)
    
    input_tokens = sorted(list(input_tokens))
    target_tokens = sorted(list(target_tokens))
    
    global num_encoder_tokens
    global num_decoder_tokens

    num_encoder_tokens = len(input_tokens)
    num_decoder_tokens = len(target_tokens)
    
    global input_features_dict
    global target_features_dict

    input_features_dict = dict(
        [(token, i) for i, token in enumerate(input_tokens)])
    target_features_dict = dict(
        [(token, i) for i, token in enumerate(target_tokens)])
    
    global reverse_input_features_dict
    global reverse_target_features_dict

    reverse_input_features_dict = dict(
        (i, token) for token, i in input_features_dict.items())
    reverse_target_features_dict = dict(
        (i, token) for token, i in target_features_dict.items())
    
    
    global max_decoder_seq_length
    global max_encoder_seq_length
    max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
    max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])
    
    encoder_input_data = np.zeros(
        (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
        for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
            #Assign 1. for the current line, timestep, & word in encoder_input_data
            encoder_input_data[line, timestep, input_features_dict[token]] = 1.

        for timestep, token in enumerate(target_doc.split()):
            decoder_input_data[line, timestep, target_features_dict[token]] = 1.
            if timestep > 0:
                decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.
    
    dimensionality = 256

    #Encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_lstm = LSTM(dimensionality, return_state=True)
    encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
    encoder_states = [state_hidden, state_cell]    
    
    #Decoder
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
    decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    #Load the model
    encoder_inputs = training_model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]

    global encoder_model

    encoder_model = Model(encoder_inputs, encoder_states)

    latent_dim = 256
    decoder_state_input_hidden = Input(shape=(latent_dim,))
    decoder_state_input_cell = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    global decoder_model
    
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

#Method to convert user input into a matrix
def string_to_matrix(user_input):
  tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
  user_input_matrix = np.zeros(
    (1, 12, 356),
    dtype='float32')
  for timestep, token in enumerate(tokens):
    if token in input_features_dict:
      user_input_matrix[0, timestep, input_features_dict[token]] = 1.
  return user_input_matrix
  
#Method that will create a response using seq2seq model we built
def generate_response(user_input):
  input_matrix = string_to_matrix(user_input)
  chatbot_response = decode_response(input_matrix)
  #Remove <START> and <END> tokens from chatbot_response
  chatbot_response = chatbot_response.replace("<START>",'')
  chatbot_response = chatbot_response.replace("<END>",'')
  return chatbot_response


#KERAS MODEL
def decode_response(test_input):
    #Getting the output states to pass into the decoder
    states_value = encoder_model.predict(test_input)
    #Generating empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    #Setting the first token of target sequence with the start token
    target_seq[0, 0, target_features_dict['<START>']] = 1.
    
    #A variable to store our response word by word
    decoded_sentence = ''
    
    stop_condition = False
    while not stop_condition:
        #Predicting output tokens with probabilities and states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
        #Choosing the one with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += " " + sampled_token
        #Stop if hit max length or found the stop token
        if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        #Update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        #Update states
        states_value = [hidden_state, cell_state]
    return decoded_sentence

model = load_model('model.h5')
load_full_model(model)

# Add description here about this if statement.
if __name__ == "__main__":
    #loading the model
    
    app.run()
