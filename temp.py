tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(email_texts)
sequences = tokenizer.texts_to_sequences(email_texts)

padded_sequences = pad_sequences(sequences, maxlen=250, padding='post', truncating='post')

max_length

pad_sequences