from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from flask import Flask, render_template, request, jsonify, make_response
import filetype
from pdf2image import convert_from_path, pdf2image
import pandas as pd
from PIL import Image
import PIL.ImageOps
from pytesseract import pytesseract, Output
import pytesseract
import platform
import json
import re
import string
import urllib.parse as urlparse
from urllib.parse import parse_qs
import pandas as pd
import pymysql
import tabula
import cv2
import datefinder
from price_parser import Price
import math
import numpy as np
import csv
import time
start_time = time.time()


# First
# 1. Change the last line to your local IP address
# 2. Comment/Uncomment the file path of your OS

# Linux path to tesseract
#pytesseract.pytesseract.tesseract_cmd = r"//usr//local//Cellar//tesseract//4.1.0//bin//tesseract"
# Hendra's Mac
# IMAGE_SAVE_FOLDER = "//Users//hendrara//Workspace//phyton//static//"
# DOWNLOAD_PATH = '//Users//hendrara//Downloads//'

# Windows path to tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# Althony's Windows
TVEXTRACT_FOLDER = "C:\\Users\\Althony\\Documents\\tvextract-deploy"
IMAGE_SAVE_FOLDER = TVEXTRACT_FOLDER+"\\static\\"
PROCESS_FOLDER_PATH = TVEXTRACT_FOLDER+"\\process\\"
EXPORT_FOLDER_PATH = TVEXTRACT_FOLDER+"\\Exports\\"


# define a folder to store and later serve the images
UPLOAD_FOLDER = '/static/'

upload_filename = ""

app = Flask(__name__)


# preprocessing
# gray scale


def grey(img):
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"./preprocess/img_gray.png", img)
    return img

# blur


def blur(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(r"./preprocess/img_blur.png", img)
    return img_blur

# threshold


def threshold(img):
    # pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    cv2.imwrite(r"./preprocess/img_threshold.png", img)
    return img

# route and function to handle the upload page


def invert(img):
    img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8))  # RGB image
    img = PIL.ImageOps.invert(np.float32(img))
    cv2.imwrite(r"./preprocess/img_inverted.png", img)
    return img


@app.route('/', methods=['GET', 'POST'])
def processing():
    start_time = time.time()
    process_folder = ""
    index_folder = 0
    if platform.system() == 'Windows':
        index_folder = 0
    elif platform.system() == 'Darwin':
        index_folder = 1
    else:
        index_folder = 0

    if len(os.listdir(PROCESS_FOLDER_PATH)) == index_folder:
        process_folder = "process folder is empty"
        return jsonify({"status": "failed", "message": process_folder})
    else:

        upload_filename = os.listdir(PROCESS_FOLDER_PATH)[index_folder]
        #upload_filename = request.args.get('filename')
        file = PROCESS_FOLDER_PATH + upload_filename
        print(file)

        if file.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'pdf')):

            tempUpFileName = upload_filename
            if file.lower().endswith('pdf'):

                pages = convert_from_path(file, 500)

                image_counter = 1

                for page in pages:
                    upload_filename = tempUpFileName[:-
                                                     4] + "-"+str(image_counter)
                    page.save(IMAGE_SAVE_FOLDER +
                              upload_filename+".jpg", 'JPEG')

                    gray = page.convert('L')
                    gray_image = gray.point(
                        lambda x: 0 if x < 200 else 255, '1')
                    gray_image.save(IMAGE_SAVE_FOLDER +
                                    upload_filename+"_gray.jpg", 'JPEG')

                    image = Image.open(IMAGE_SAVE_FOLDER +
                                       upload_filename+"_gray.jpg")
                    invert_img = PIL.ImageOps.invert(image)

                    invert_img.save(IMAGE_SAVE_FOLDER +
                                    upload_filename+"_inverted.jpg", 'JPEG')

                    image_counter = image_counter + 1

                filelimit = image_counter-1

                # Create an completely empty Dataframe without any column names, indices or data
                img_data = pd.DataFrame()
                extracted_text = ""
                for i in range(1, filelimit + 1):

                    upload_filename = tempUpFileName[:-4] + \
                        "-"+str(i)+"_inverted.jpg"

                    extracted_text += " \n"+pytesseract.image_to_string(
                        Image.open(IMAGE_SAVE_FOLDER + upload_filename), lang='eng+ind')

                    print(extracted_text)

                    img_data_pg = pytesseract.image_to_data(
                        IMAGE_SAVE_FOLDER + upload_filename, lang='eng+ind',  output_type='data.frame')

                    img_data = img_data.append(img_data_pg, ignore_index=True)
                    print(img_data)

                # only take first page, need fix
                img_src = UPLOAD_FOLDER + tempUpFileName[:-4]+"-1_inverted.jpg"

            elif(file.lower().endswith('png')):
                im = Image.open(file)
                rgb_im = im.convert('RGB')
                rgb_im.save(IMAGE_SAVE_FOLDER + tempUpFileName[:-4]+".jpg")
                img_src = UPLOAD_FOLDER + tempUpFileName[:-4]+".jpg"
            else:
                print(file)
                image = Image.open(file)
                image.save(IMAGE_SAVE_FOLDER + tempUpFileName[:-4]+".jpg")
                img_src = UPLOAD_FOLDER + tempUpFileName[:-4]+".jpg"
                print(img_src)

            #img = cv2.resize(img, (1000, 1000))
            # image pre-processing
            #im = cv2.imread(img)
            #im_gray = gray_b(img)
            # im_blur = blur(img)
            # im_thresh = threshold(im_blur)
            # image = im_thresh

            img = cv2.imread("."+img_src, 0)
            # alpha = 4.0
            # beta = -160

            # new = alpha * img + beta
            # new = np.clip(new, 0, 255).astype(np.uint8)

            # cv2.imwrite("."+img_src, new)

            image = Image.open("."+img_src)

            extracted_text = pytesseract.image_to_string(
                image, lang='eng+ind+msa', config="--psm 6")
            img_data = pytesseract.image_to_data(
                image, lang='eng+ind+msa',  output_type='data.frame')
            img_src = "."+img_src

            print(extracted_text)

            # Calculate total average confidence level
            img_data = pd.DataFrame.from_dict(img_data)
            img_data = img_data.dropna()
            sum = img_data.sum(axis=0)
            count = img_data.count(axis=0)
            avgConf = sum.conf / count.conf
            avgConf_str = str(("%.2f" % avgConf) + " %")

            # create index column for data frame view
            n_boxes = len(img_data['text'])
            pd_index = []
            for i in range(n_boxes):
                pd_index.append(i)
            s = pd.Series(pd_index)
            img_data = img_data.set_index([s])

            img_data.to_csv(EXPORT_FOLDER_PATH +
                            upload_filename+"-textcoordinates.csv")

            # Arrange Text
            num = 2
            num2 = 2
            num3 = 2
            text = ""
            for i in range(n_boxes):
                if int(img_data.loc[i, 'block_num']) < num:
                    if int(img_data.loc[i, 'line_num']) < num2:
                        if int(img_data.loc[i, 'par_num']) < num3:
                            text += img_data.loc[i, 'text'] + " "
                        else:
                            num3 += 1
                            text += "\n"
                            text += img_data.loc[i, 'text'] + " "
                    else:
                        num2 += 1
                        text += "\n"
                        text += img_data.loc[i, 'text'] + " "

                else:
                    num += 1
                    text += "\n"
                    text += img_data.loc[i, 'text'] + " "
            before_correction = text
            print(before_correction)

            # get low confidence text
            print("LOW CONFIDENCE TEXT")
            corrected_word = ""
            low_conf_text_lst = []
            for i in range(n_boxes):
                if int(img_data.loc[i, 'conf']) < 60:
                    low_conf_text = img_data.loc[i, 'text']
                    low_conf_text = low_conf_text.lower()
                    low_conf_text_lst.append(low_conf_text)
                    print(low_conf_text)
                    # corrected_word = autocorrect(low_conf_text)
                    # corrected_word = corrected_word.upper()
                    # text = text.replace(low_conf_text.upper(),corrected_word)

            # Create TXT File
            f = open(EXPORT_FOLDER_PATH+upload_filename + '.txt', 'w')
            f.write(text)
            f.close()

            # Generate searchable/OCR-ed PDF
            pdf = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
            with open(EXPORT_FOLDER_PATH + upload_filename+"-searchable.pdf", 'w+b') as f:
                f.write(pdf)  # pdf type is bytes by default

            f.close()

            # Create CSV File

            tabula.convert_into(EXPORT_FOLDER_PATH + upload_filename+"-searchable.pdf", EXPORT_FOLDER_PATH +
                                upload_filename+"-full.csv", output_format="csv", pages='all', area=(0, 0, 100, 100), relative_area=True)  # top,left,bottom,right
            tabula.convert_into(EXPORT_FOLDER_PATH + upload_filename+"-searchable.pdf", EXPORT_FOLDER_PATH +
                                upload_filename+"-body.csv", output_format="csv", pages='all', area=(35, 0, 70, 100), relative_area=True)  # area=(524.201, 5.316, 1074.984, 1700.197)
            # # Fix bug if no column
            df = pd.read_csv(EXPORT_FOLDER_PATH +
                             upload_filename+"-body.csv", encoding='latin1')
            # df = tabula.io.read_pdf(EXPORT_FOLDER_PATH + upload_filename+"-searchable.pdf", output_format="dataframe",
            #                         encoding='utf-8', java_options=None, pandas_options=None, multiple_tables=True, user_agent=None, pages="all")
            df = df.fillna(" ")
            body_content = df

            exampleFile = open(EXPORT_FOLDER_PATH +
                               upload_filename+"-body.csv")
            exampleReader = csv.reader(exampleFile)
            exampleData = list(exampleReader)
            row_content = ""
            for i in range(len(exampleData)):
                if(i == (len(exampleData)-1)):
                    row_content += "[" + ', '.join(['"{}"'.format(value)
                                                    for value in exampleData[i]])+"]"
                elif(i != 0):
                    row_content += "[" + ', '.join(['"{}"'.format(value)
                                                    for value in exampleData[i]])+"],"

            table_json = """{
                "table_header":[""" + ', '.join(['"{}"'.format(value) for value in exampleData[0]])+"""],
                "row_data": [""" + row_content+"""]
            }
            """

            print(table_json)

            table_json = json.loads(table_json)
            # Footer content

            image = cv2.resize(img, (1000, 1000))
            thresh = 255 - cv2.threshold(image, 0, 255,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # left,top,width,height
            x = 0
            y = 700
            w = 1000
            h = 300

            height = image.shape[0]
            width = image.shape[1]

            ROI = thresh[y:y+h, x:x+w]
            trim_footer = pytesseract.image_to_string(
                ROI, lang='eng', config='--psm 6')
            print(trim_footer)

            # dictionary_json call from MySQL
            try:

                # Open database connection
                db = pymysql.connect(host='localhost',
                                     user='root',
                                     db='tvextract')

                sql = "SELECT * FROM `document_category` where id = 1"
                cursor = db.cursor()

                # Execute the SQL command
                cursor.execute(sql)

                # Fetch all the rows in a list of lists.
                jsonStr = cursor.fetchall()
                for row in jsonStr:
                    jsonStr = row[4]
                    print("dictionary_json = ", row[4])
                    # print(row[5])

                dictionary_json = json.loads(jsonStr)

                dictionary_key = []

                for i in range(len(dictionary_json)):
                    dictionary_key.append(dictionary_json[i]["text"])

                print(dictionary_key)

            except pymysql.Error as e:
                print("MySQL Database Error")

            user_defined_list_string = request.args.get('predefined')
            if(user_defined_list_string == None):
                user_defined_list_string = ""
            user_defined_list_string = user_defined_list_string.lower()
            user_defined_list = user_defined_list_string.split(",")

            # data cleaning
            # start from here replaced text to extracted_text
            #extracted_text = extracted_text.upper()
            extracted_text = clean_text_round1(text)
            extracted_text = clean_text_round2(text)

            print(extracted_text)

            extracted_text = extracted_text.lower()
            words = extracted_text.split()
            words = text_lemmatization(words)
            extracted_text = extracted_text.replace('\n', '<br>')
            print("Extracted text after data cleaning  \n")
            print(words)
            len_user_defined_list = range(len(user_defined_list))
            result = []
            next_word = ""
            found_similar_label = False

            searchLabel(user_defined_list, next_word, result, words,
                        found_similar_label, dictionary_json, dictionary_key)

            print(user_defined_list)
            print(result)
            my_array = [user_defined_list, result]

            pairs = zip(my_array[0], my_array[1])
            json_values = ('"{}": "{}"'.format(label, value)
                           for label, value in pairs)
            my_string = '{' + ', '.join(json_values) + '}'

            print(my_string)

            my_string = json.loads(my_string)

            # "Automated" label retrival
            system_defined_label = ['invoice', 'date', 'deadline', 'balance', 'p.o',
                                    'purchase', 'subtotal', 'discounts', 'tax', 'shipping', 'total', 'paid']
            system_result = []
            searchLabel(system_defined_label, next_word, system_result, words,
                        found_similar_label, dictionary_json, dictionary_key)

            print(system_defined_label)
            print(system_result)
            my_array = [system_defined_label, system_result]

            pairs = zip(my_array[0], my_array[1])
            json_values = ('"{}": "{}"'.format(label, value)
                           for label, value in pairs)
            system_string = '{' + ', '.join(json_values) + '}'

            print(system_string)

            system_string = json.loads(system_string)

            extracted_text = clean_text_round1(extracted_text)
            extracted_text = clean_text_round2(extracted_text)

            img_src = UPLOAD_FOLDER + tempUpFileName[:-4]+"-1.jpg"

            print("--- %s seconds ---" % (time.time() - start_time))

            # return jsonify({"status": "Successfully processed",
            #                 "average_confidence_level": avgConf_str,
            #                 "all_extracted_text": extracted_text,
            #                 "result_list": my_string,
            #                 "user_defined_list": user_defined_list,
            #                 "result": result,
            #                 # "Item Content":[df.to_html(classes='df', header="true")],
            #                 # "Text Coordinates":[img_data.to_html(classes='img_data', header="true")],
            #                 "img_src": img_src,
            #                 "table_json": table_json,
            #                 "footer": trim_footer,
            #                 "sugested_key_value_pair": system_string,

            #                 })

            return render_template('Edit & Review.html',
                                   msg='Successfully processed',
                                   low_conf_text_lst=low_conf_text_lst,
                                   avgConf_str=avgConf_str,
                                   before_correction=before_correction,
                                   extracted_text=extracted_text,
                                   len_user_defined_list=len_user_defined_list,
                                   user_defined_list=user_defined_list,
                                   result=result,
                                   tables2=[img_data.to_html(
                                       classes='img_data', header="true")],
                                   tables=[body_content.to_html(
                                       classes='body_content', header="true")],
                                   footer=trim_footer,
                                   img_src=img_src,
                                   system_string=system_string,
                                   upload_filename=upload_filename,
                                   export_extractedResult_json=my_string,
                                   tables_json=json.dumps(table_json)
                                   )
        else:
            return "File type not support."


def searchLabel(user_defined_list, next_word, result, words, found_similar_label, dictionary_json, dictionary_key):
    for i in range(len(user_defined_list)):
        # Find if user_defined_label is in the extreacted text
        if(user_defined_list[i] in words):
            try:
                next_word = words[words.index(user_defined_list[i])+1]+" "+words[words.index(
                    user_defined_list[i])+2]+" "+words[words.index(user_defined_list[i])+3]+words[words.index(user_defined_list[i])+4]+words[words.index(user_defined_list[i])+5]+words[words.index(user_defined_list[i])+6]
            except:
                next_word = words[words.index(user_defined_list[i])+1]+" "

            print(user_defined_list[i]+"="+next_word)
            search_word = user_defined_list[i]

            getNext(search_word, next_word, result, words)
        # If no then see if there is any similar label that is defined in the list
        elif(user_defined_list[i] in dictionary_key):
            print(user_defined_list[i]+" in dict.keys()")
            for j in range(len(dictionary_json)):  # loop to match key
                if(user_defined_list[i] == dictionary_json[j]['text']):
                    # loop similar label list
                    for k in range(len(dictionary_json[j]['nodes'])):
                        if(dictionary_json[j]['nodes'][k]['text'] in words):
                            print("found similar label for " +
                                  user_defined_list[i]+" in docs as "+dictionary_json[j]['nodes'][k]['text'])
                            next_word = words[words.index(dictionary_json[j]['nodes'][k]['text'])+1] + words[words.index(
                                dictionary_json[j]['nodes'][k]['text'])+2] + words[words.index(dictionary_json[j]['nodes'][k]['text'])+3] + words[words.index(dictionary_json[j]['nodes'][k]['text'])+4] + words[words.index(dictionary_json[j]['nodes'][k]['text'])+5] + words[words.index(dictionary_json[j]['nodes'][k]['text'])+6]
                            search_word = user_defined_list[i]
                            getNext(search_word, next_word, result, words)
                            # result.append(next_word)
                            found_similar_label = True
                            break

            if(found_similar_label == False):
                result.append(" ")
                print(user_defined_list[i] +
                      "not in similar label list, please add.")
        # If no then append empty string first
        else:
            result.append(" ")
            print(user_defined_list[i]+" Not in list")


def getNext(search_word, next_word, result, words):

    if (search_word == "invoice" or search_word == "po" or search_word == "account"):
        r1 = re.findall(r"\d[-/]?", next_word)
        strs = ""
        for r in r1:
            strs += ''.join(r)
        print(strs)
        result.append(strs)
    elif(search_word == "date" or search_word == "deadline"):
        matches = datefinder.find_dates(next_word)
        match_date = ""
        for match in matches:
            if(match.date() == None):
                match_date = ""
            else:
                match_date += match.date().strftime("%d %b %Y")
                print(match_date)
        result.append(match_date)
    elif(search_word == "subtotal" or search_word == "tax" or search_word == "total" or search_word == "paid" or search_word == "discount" or search_word == "balance" or search_word == "shipping"):
        price = Price.fromstring(next_word)
        if(price.currency == None):
            price_currency = ""
        else:
            price_currency = price.currency
        if(price.amount_text == None):
            price_amount = ""
        else:
            price_amount = price.amount_text
        print("Currency: "+price_currency + " Amount: "+price_amount)
        result.append(price_currency + price_amount)
    elif(search_word in words):
        result.append(next_word)


def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    #text = text.upper()
    text = re.sub('\[*?:\]', '', text)
    text = re.sub('}', '', text)
    text = re.sub("\)", '', text)
    text = re.sub("\(", '', text)
    #text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    remove = string.punctuation
    remove = remove.replace(".", "")  # don't remove hyphens
    remove = remove.replace("-", "")
    remove = remove.replace("/", "")
    pattern = r"[{}]".format(remove)  # create the pattern
    re.sub(pattern, "", text)
    #text = re.sub('\w*\d\w*', '', text)
    return text

# Apply a second round of cleaning


def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[:‘’“”…]', '', text)
    text = re.sub("\[", '', text)
    text = re.sub("\(", '', text)
    #text = re.sub('\n', '', text)
    # print(text)
    return text


def text_lemmatization(words):

    wordnet_lemmatizer = WordNetLemmatizer()

    words_lemma = []
    for w in words:
        words_lemma.append(wordnet_lemmatizer.lemmatize(w))

    return words_lemma


# still need refinement


def autocorrect(w):
    words = open("C:\\Users\\Althony Real\\Downloads\\spell.words").readlines()
    words = [word.strip() for word in words]
    # w='invc'

    s = 0
    e = 2
    lis = []
    lop = range(len(w))
    print(lop)
    for x in lop:
        if e <= len(w):
            aw = w[s:e]
            print(s, e, aw)
            lis.append(aw)
            s += 1
            e += 1
        else:
            print("done")

    s = 0
    e = 3
    for x in lop:
        if e <= len(w):
            aw = w[s:e]
            print(s, e, aw)
            lis.append(aw)
            s += 1
            e += 1
        else:
            print("done")

    s = 0
    e = 4
    for x in lop:
        if e <= len(w):
            aw = w[s:e]
            print(s, e, aw)
            lis.append(aw)
            s += 1
            e += 1
        else:
            print("done")

    s = 0
    e = 5
    for x in lop:
        if e <= len(w):
            aw = w[s:e]
            print(s, e, aw)
            lis.append(aw)
            s += 1
            e += 1
        else:
            print("done")

    print(lis)

    wordl = []
    freql = []
    j = 0
    z = len(words)
    # print(words)

    for word in words:
        mat = 0
        for i in range(len(lis)):
            if(lis[i] in word):
                mat += 1
        if(mat > 0):
            wordl.append(word)
            freql.append(mat)
        #print(j," of ",z,word,mat)
        # print(freql)
        j += 1

    data = {'Name': wordl, 'freq': freql}
    df = pd.DataFrame(data)

    df['word_rank'] = df['freq'].rank(method='max', ascending=True)

    df = df.sort_values(by=['freq'], ascending=False)[:50]

    print(df)

    name_list = df['Name']

    found = ""
    for n in name_list:
        if("invoice" == n):
            found += n
            print(found)

    if(found == ""):
        found += name_list[0]
        print(found)
    else:
        found += w
        print(found)

    return found


@app.route('/retrieve_roi')
def retrieveROI():
    # Get text of selected area
    img_src = request.args.get('img_src')
    #img_src = request.args.get('upload_filename')

    image = cv2.imread("."+img_src, 0)

    image = cv2.resize(image, (1000, 1000))
    thresh = 255 - cv2.threshold(image, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    roi_coordinates = request.args.get('coordinates')

    roi_coordinates = roi_coordinates.split(",")
    roi_coordinates = [int(i) for i in roi_coordinates]

    print(roi_coordinates)
    x = roi_coordinates[0]
    y = roi_coordinates[1]
    w = roi_coordinates[2]
    h = roi_coordinates[3]

    height = image.shape[0]
    width = image.shape[1]

    x = math.floor(x / 100 * width)
    y = math.floor(y / 100 * height)
    w = math.ceil(w / 100 * width)
    h = math.ceil(h / 100 * height)

    print(x, y, w, h)

    ROI = thresh[y:y+h, x:x+w]
    data = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')
    print(data)
    return jsonify(result=data)


@app.route('/table_content')
def retrieveTableContent():
    upload_filename = request.args.get('upload_filename')
    print(upload_filename)
    roi_coordinates = request.args.get('coordinates')
    print(roi_coordinates)
    roi_coordinates = roi_coordinates.split(",")
    roi_coordinates = [int(i) for i in roi_coordinates]

    top = roi_coordinates[0]
    left = roi_coordinates[1]
    bottom = roi_coordinates[2]
    right = roi_coordinates[3]

    body_content = tabula.read_pdf(
        EXPORT_FOLDER_PATH + upload_filename+"-searchable.pdf", pages="all",
        area=(top, left, bottom, right), relative_area=True)

    tabula.convert_into(EXPORT_FOLDER_PATH + upload_filename+"-searchable.pdf", EXPORT_FOLDER_PATH +
                        upload_filename+"-body.csv", output_format="csv", pages='all', area=(top, left, bottom, right), relative_area=True)
    df = pd.read_csv(EXPORT_FOLDER_PATH + upload_filename +
                     "-body.csv", encoding='latin1')
    df = df.fillna(" ")
    body_content = df

    print(body_content)

    exampleFile = open(EXPORT_FOLDER_PATH +
                       upload_filename+"-body.csv")
    exampleReader = csv.reader(exampleFile)
    exampleData = list(exampleReader)
    row_content = ""
    for i in range(len(exampleData)):
        if(i == (len(exampleData)-1)):
            row_content += "[" + ', '.join(['"{}"'.format(value)
                                            for value in exampleData[i]])+"]"
        elif(i != 0):
            row_content += "[" + ', '.join(['"{}"'.format(value)
                                            for value in exampleData[i]])+"],"

    table_json = """ {
        "table_header":[""" + ', '.join(['"{}"'.format(value) for value in exampleData[0]])+"""],
        "row_data": [""" + row_content+"""]
    }
    """

    table_json = re.sub("\n", '', table_json)
    table_json = table_json.replace("\\", "")
    print(table_json)

    return jsonify(tables_json=table_json.replace("\\", ""), tables_html=[body_content.to_html(
        classes='body_content', header="true")])


if __name__ == '__main__':
    app.run(port=8011, debug=True, threaded=True)
