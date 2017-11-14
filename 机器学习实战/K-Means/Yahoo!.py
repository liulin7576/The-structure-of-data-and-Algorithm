import urllib
import urllib.request as r
import json

def getGrab(stAddress, city):  #对地址进行地理解码
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' %(stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params
    print(yahooApi)
    c = r.urlopen(yahooApi)
    return json.loads(c.read())

#爬虫不出来。。。
