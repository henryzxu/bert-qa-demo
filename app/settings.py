PORT = 8080

http_proxy = ""
https_proxy = ""

if http_proxy or https_proxy:
    proxyDict = {
                  "http": http_proxy,
                  "https": https_proxy,
                }
else:
    proxyDict = {}
