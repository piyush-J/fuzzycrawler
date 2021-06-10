import requests
from bs4 import BeautifulSoup as bs
s = requests.Session()
s.headers['user-agent'] ="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36"

def get_URL_Input():
    url=input("Enter URL : ")
    return url

def extractResponseCode(k):
    val=''
    for i in range(0,len(k)):
        if k[i]=='[':
            i=i+1
            while k[i]!=']':
                val=val+k[i]
                i=i+1
            break
    val=int(val)
    return val

def get_all_forms(url):
    soup = bs(s.get(url).content, "html.parser")
    return soup.find_all("form")

def get_form_details(form):
    details = {}
    try:
        action = form.attrs.get("action").lower()
    except:
        action = None
    method = form.attrs.get("method", "get").lower()
    inputs = []
    for input_tag in form.find_all("input"):
        input_type = input_tag.attrs.get("type", "text")
        input_name = input_tag.attrs.get("name")
        input_value = input_tag.attrs.get("value", "")
        inputs.append({"type": input_type, "name": input_name, "value": input_value})
    return inputs

def form_input_extraction(input):
    keys=[]
    for variable in input:
        if variable['type']!='submit':
            keys.append(variable['name'])
    return keys

def preprocessing_Form_Fields(url):
    #Getting form details and its keys of given url
    form=get_all_forms(url)
    form_details = get_form_details(form[0])
    keys=form_input_extraction(form_details)
    #print(keys)
    return form_details,keys

def get_Values(keys):
    #Getting values for the retrieved keys
    values=[]
    for i in keys:
        print("Enter "+i+" :",end=" ")
        k=input()
        values.append(k)
    #To be modified by fuzzing inputs
    #values=["john.doe@example.com ( ' OR 1=1;-- )",'Doe@123']
    #values=["john.doe@example.com",'Doe@123']
    return values

def form_input_feeding(keys,values,input):
    logindata={}
    for i in range(len(keys)):
        logindata[keys[i]]=values[i]
    for variable in input:
        if variable['type']=='submit':
            logindata[variable['name']]=variable['value']
    return logindata

def validation(logindata,keys,receive):
    send=s.post(url,data=logindata)
    status=0
    #print(send.url)
    if receive.content==send.content:
        #Check for same content after post operation
        print("Feeding Credentials Failed") 
    elif extractResponseCode(str(send))>=400 or (url==send.url):
        #Check for invalid response code or urls identity after post operation
        print("Fuzzed Credentials Failed")
        status=1
    else:
        #Checks if url is changed after post operation
        newform=get_all_forms(send.url)
        flag=0
        if len(newform)!=0:
            flag=0
            newkeys=form_input_extraction(get_form_details(newform[0]))
            for i in newkeys:
                if i in keys:
                    flag=flag+1
        if flag!=0:
            print("Fuzzed Credentials Failed")
            status=1
        else:
            print("Fuzzed Credentials Passed")
            status=2
    return status



if __name__=="__main__":
    url=get_URL_Input()
    #url="http://localhost/demo/example_mysql_injection_login.php" 
    try:
        receive=s.get(url)
    except:
        print("Invalid URL")
        exit(0)
    #Getting form details and its keys of given url
    form_details,keys=preprocessing_Form_Fields(url)
    print("Form inputs in give url are : ")
    for i in range(len(keys)):
        print(str(i+1)+")"+keys[i])
    #Getting values for the retrieved keys
    #Write loop here to feed inputs
    #if status is 0 or 1 continue fuzzing, if status is 2 sql injection passed so stop fuzzing
    values=get_Values(keys)
    #Creating the login data
    logindata=form_input_feeding(keys,values,form_details)
    #Checking for SQL injection
    status=validation(logindata,keys,receive)