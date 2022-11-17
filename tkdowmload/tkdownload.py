def rbVideo():
    global getvideo
    labelMsg.config(text="")
    getvideo = videorb.get()

def clickDown()
    global getvideo, strftype, listradio
    labelMsg.config(text="")
    if(url.get()==""):
        labelMsg.config(text="網址欄位必需輸入")
        return

    if(path.get()=="")
        pathdir = "download"
    else:
        pathdir = path.gey()
        pathdir = pathdir.replace("\\","\\\\")


    try:
        yt = YouTube(url.get())