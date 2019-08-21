data_dir = 'data'
output_dir = 'results'
PORT = 8080
gen_file = "./data/squad_input.json"

http_proxy = ""
https_proxy = ""

if http_proxy or https_proxy:
    proxyDict = {
                  "http": http_proxy,
                  "https": https_proxy,
                }
else:
    proxyDict = {}
