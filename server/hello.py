from flask import Flask
from flask import request
from flask import jsonify
from google.cloud import translate

app = Flask(__name__)

@app.route('/')
def hello_world():
	return 'Hello, World!'


@app.route('/predictPersonality', methods=['POST'])
def predictPersonality():
	print(request.data)
	#user_posts = request.form['posts']
	user_posts = request.form.get('posts')
	user_posts = translateToEnglish(user_posts)
	predict_type = predict(user_posts)
	resp = jsonify({'type': user_posts})
	return resp



def translateToEnglish(text):
	translate_client = translate.Client()
	results = translate_client.get_languages()

	for language in results:
		print(u'{name} ({language})'.format(**language))


	text = u'Hello, world!'
	target = 'zh'

	translation = translate_client.translate(text,target_language=target)

	print(u'Text: {}'.format(text))
	print(u'Translation: {}'.format(translation['translatedText']))
	return translation


def predict(posts):
	return None
