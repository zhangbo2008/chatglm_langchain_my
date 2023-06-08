from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
				"Howdy!",	# str representing string value in 'parameter_9' Textbox component
				"",	# Any representing  in 'parameter_1' State component
				"null",	# str representing filepath to JSON file in 'parameter_8' Chatbot component
				"LLM 对话",	# str representing string value in '请选择使用模式' Radio component
				api_name="/tiwen"
)
print(result)