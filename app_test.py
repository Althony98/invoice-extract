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
import ftfy
import requests
from bs4 import BeautifulSoup
from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring
from json import dumps
from xmljson import parker, Parker
import json
from pyzbar.pyzbar import decode
from PIL import Image
start_time = time.time()


# First
# 1. Change the last line to your local IP address
# 2. Comment/Uncomment the file path of your OS

# Linux path to tesseract
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
# Hendra's Mac
# IMAGE_SAVE_FOLDER = "//Users//hendrara//Workspace//phyton//static//"
# DOWNLOAD_PATH = '//Users//hendrara//Downloads//'

# Windows path to tesseract
#pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# Althony's Windows
TVEXTRACT_FOLDER = "/var/www/html/tvextract-web/application/libraries/python/"
#IMAGE_SAVE_FOLDER = TVEXTRACT_FOLDER+"/static/"
IMAGE_SAVE_FOLDER = "/var/www/html/tvextract-web/uploads/static/"
PROCESS_FOLDER_PATH = TVEXTRACT_FOLDER+"process/"
EXPORT_FOLDER_PATH = TVEXTRACT_FOLDER+"Exports/"


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

# Process Invoice


@app.route('/', methods=['GET', 'POST'])
def processing():
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
            extracted_text = ""

            if file.lower().endswith('pdf'):

                pages = convert_from_path(file, 250)

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
                                       upload_filename+".jpg")
                    invert_img = PIL.ImageOps.invert(image.convert('RGB'))

                    invert_img.save(IMAGE_SAVE_FOLDER +
                                    upload_filename+"_inverted.jpg", 'JPEG')

                    image_counter = image_counter + 1

                filelimit = image_counter-1

                # Create an completely empty Dataframe without any column names, indices or data
                img_data = pd.DataFrame()

                for i in range(1, filelimit + 1):

                    upload_filename = tempUpFileName[:-4] + \
                        "-"+str(i)+".jpg"

                    print(upload_filename)

                    extracted_text += " \n"+pytesseract.image_to_string(
                        Image.open(IMAGE_SAVE_FOLDER +
                                   upload_filename), lang='eng+ind')

                    print(extracted_text)

                    img_data_pg = pytesseract.image_to_data(
                        IMAGE_SAVE_FOLDER + upload_filename, lang='eng+ind',  output_type='data.frame')

                    img_data = img_data.append(img_data_pg, ignore_index=True)
                    print(img_data)

                # only take first page, need fix
                img_src = IMAGE_SAVE_FOLDER + tempUpFileName[:-4]+"-1.jpg"

            elif(file.lower().endswith('png')):
                im = Image.open(file)
                rgb_im = im.convert('RGB')
                rgb_im.save(IMAGE_SAVE_FOLDER + tempUpFileName[:-4]+".jpg")
                img_src = IMAGE_SAVE_FOLDER + tempUpFileName[:-4]+".jpg"
            else:
                print(file)
                image = Image.open(file)
                image.save(IMAGE_SAVE_FOLDER + tempUpFileName[:-4]+".jpg")
                img_src = IMAGE_SAVE_FOLDER + tempUpFileName[:-4]+".jpg"
                print(img_src)

            #img = cv2.resize(img, (1000, 1000))
            # image pre-processing
            #im = cv2.imread(img)
            #im_gray = gray_b(img)
            # im_blur = blur(img)
            # im_thresh = threshold(im_blur)
            # image = im_thresh

            #img = cv2.imread(img_src, 0)
            # alpha = 4.0
            # beta = -160

            # new = alpha * img + beta
            # new = np.clip(new, 0, 255).astype(np.uint8)

            # cv2.imwrite("."+img_src, new)

            image = Image.open(img_src)

            extracted_text = pytesseract.image_to_string(
                image, lang='eng+ind+msa', config="--psm 6")
            img_data = pytesseract.image_to_data(
                image, lang='eng+ind+msa',  output_type='data.frame')
            #img_src = "."+img_src

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
            pdf = pytesseract.image_to_pdf_or_hocr(
                image, extension='pdf', lang="eng+ind")
            with open(EXPORT_FOLDER_PATH + upload_filename+"-searchable.pdf", 'w+b') as f:
                f.write(pdf)  # pdf type is bytes by default

            f.close()

            # Create CSV File

            tabula.convert_into(EXPORT_FOLDER_PATH + upload_filename+"-searchable.pdf", EXPORT_FOLDER_PATH +
                                upload_filename+"-full.csv", output_format="csv", pages='all', area=(0, 0, 100, 100), relative_area=True)  # top,left,bottom,right
            tabula.convert_into(EXPORT_FOLDER_PATH + upload_filename+"-searchable.pdf", EXPORT_FOLDER_PATH +
                                upload_filename+"-body.csv", output_format="csv", pages='all', area=(35, 0, 50, 100), relative_area=True)  # area=(524.201, 5.316, 1074.984, 1700.197)
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
            img = cv2.imread(img_src, 0)
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
                db = pymysql.connect(host='tvextract.chrlbwkar9om.ap-southeast-1.rds.amazonaws.com',
                                     user='drife',
                                     password='123masuk',
                                     port=3306,
                                     db='tvextract')

                sql = "SELECT * FROM `document_category` where id = 1"
                cursor = db.cursor()

                # Execute the SQL command
                cursor.execute(sql)
                print("MySQL database connected")
                # Fetch all the rows in a list of lists.
                jsonStr = cursor.fetchall()
                for row in jsonStr:
                    jsonStr = row[3]
                    print("dictionary_json = ", row[3])
                    # print(row[5])

                #jsonStr = """[{"text":"invoice","label":"invoice number","nodes":[{"text":"inv"},{"text":"iv."},{"text":"faktur"}]},{"text":"date","label":"invoice date","nodes":[{"text":"day"},{"text":"tarikh"}]},{"text":"subtotal","label":"sub total","nodes":[{"text":"subtotal:"},{"text":"subtotal."},{"text":"sub total"},{"text":"sub"}]},{"text":"total","label":"total","nodes":[{"text":"jumlah"},{"text":"all together"},{"text":"sum"},{"text":"totals"}]},{"text":"tax","label":"tax","nodes":[{"text":"vat"}]},{"text":"deadline","label":"due date","nodes":[{"text":"due"}]}]"""

                dictionary_json = json.loads(jsonStr)

                dictionary_key = []

                for i in range(len(dictionary_json)):
                    dictionary_key.append(dictionary_json[i]["text"])

                print(dictionary_key)

            except pymysql.Error as e:
                print("MySQL Database Error: "+str(e))

            # temporary, removed once cloud db connected
            # jsonStr = """[{"text":"invoice","label":"invoice number","nodes":[{"text":"inv"},{"text":"iv."},{"text":"faktur"}]},{"text":"date","label":"invoice date","nodes":[{"text":"day"},{"text":"tarikh"}]},{"text":"subtotal","label":"sub total","nodes":[{"text":"subtotal:"},{"text":"subtotal."},{"text":"sub total"},{"text":"sub"}]},{"text":"total","label":"total","nodes":[{"text":"jumlah"},{"text":"all together"},{"text":"sum"},{"text":"totals"}]},{"text":"tax","label":"tax","nodes":[{"text":"vat"}]},{"text":"deadline","label":"due date","nodes":[{"text":"due"}]}]"""
            # dictionary_json = ""
            # dictionary_json = json.loads(jsonStr)

            # dictionary_key = []

            # for i in range(len(dictionary_json)):
            #     dictionary_key.append(dictionary_json[i]["text"])

            # print(dictionary_key)

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
            extracted_text = extracted_text.replace(
                '\n', '<br>')  # can try remove
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
                                    'purchase', 'subtotal', 'discount', 'tax', 'shipping', 'total', 'paid']
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

            img_src = IMAGE_SAVE_FOLDER + tempUpFileName[:-4]+"-1.jpg"

            print("--- %s seconds ---" % (time.time() - start_time))

            return jsonify({"status": "Successfully processed",
                            "average_confidence_level": avgConf_str,
                            "all_extracted_text": extracted_text,
                            "result_list": my_string,
                            "user_defined_list": user_defined_list,
                            "result": result,
                            # "Item Content":[df.to_html(classes='df', header="true")],
                            # "Text Coordinates":[img_data.to_html(classes='img_data', header="true")],
                            "img_src": img_src,
                            "table_json": table_json,
                            "footer": trim_footer,
                            "sugested_key_value_pair": system_string,

                            })

            # return render_template('Edit & Review.html',
            #                      msg='Successfully processed',
            #                     low_conf_text_lst=low_conf_text_lst,
            #                    avgConf_str=avgConf_str,
            #                   before_correction=before_correction,
            #                  extracted_text=extracted_text,
            #                 len_user_defined_list=len_user_defined_list,
            #                user_defined_list=user_defined_list,
            #               result=result,
            #              tables2=[img_data.to_html(
            #                 classes='img_data', header="true")],
            #            tables=[body_content.to_html(
            #               classes='body_content', header="true")],
            #          footer=trim_footer,
            #         img_src=img_src,
            #        system_string=system_string,
            #       upload_filename=upload_filename,
            #      export_extractedResult_json=my_string,
            #     tables_json=json.dumps(table_json)
            #    )
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
    text = re.sub(pattern, "", text)

    return text

# Apply a second round of cleaning


def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[:‘’“”…]', '', text)
    text = re.sub("\[", '', text)
    text = re.sub("\(", '', text)
    text = re.sub('<br>', '', text)  # remove for jsonify output
    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)

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
    print(img_src)
    image = cv2.imread(img_src, 0)

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


@app.route("/idcard")
def processID():
    doc_type = "Indonesia_IC"
    upload_filepath = request.args.get('upload_filepath')
    # (1) Read
    img = cv2.imread(upload_filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./KTP-OCR/dataset/im_gray.jpg", gray)

    # (2) Threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    cv2.imwrite("./KTP-OCR/dataset/im_thresh.jpg", (threshed))

    img_data = pytesseract.image_to_data(
        (threshed), lang="ind", config="--psm 6", output_type='data.frame')

    # loop over each of the individual text localizations
    for i in range(0, len(img_data["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = img_data["left"][i]
        y = img_data["top"][i]
        w = img_data["width"][i]
        h = img_data["height"][i]

        # extract the OCR text itself along with the confidence of the
        # text localization
        text = img_data["text"][i]
        conf = int(img_data["conf"][i])

        # can be define by user
        min_conf = 62

        # filter out weak confidence text localizations
        if conf > min_conf:
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw a bounding box around the text along
            # with the text itself
            text = "".join(
                [c if ord(c) < 128 else "" for c in text]).strip()
            # cv2.rectangle(img, (x, y), (x + w, y + h),
            #               (0, 255, 0), 2)
            # cv2.putText(img, text, (x, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    result = id_img_data_text(img_data)
    print(result)

    result = id_clean_text_round1(result)

    result = id_clean_text_round2(result)

    print(result)

    # img_data = pytesseract.image_to_data((threshed), lang="ind",config="--psm 6", output_type='data.frame')
    img_data = img_data.dropna()
    # img_data.to_csv("C:\\Users\\HPC1\\Documents\\tvextract-master\\KTP-OCR\\dataset\\coordinates.csv")
    total = img_data.sum(axis=0)
    count = img_data.count(axis=0)
    avgConf = total.conf / count.conf
    avgConf_str = str(("%.2f" % avgConf) + " %")
    print("\nExtraction Confidence Level: "+avgConf_str)

    text = result.split()
    print(text)

    provinsi = nik = nama = lahir = kelamin = darah = alamat = rt_rw = kel_desa = kecamatan = agama = perkawinan = pekerjaan = kewarganegaraan = hingga = ""

    provinsi = result.splitlines()[0] + " " + result.splitlines()[1]
    print(provinsi)

    try:
        nik = text[text.index("nik")+1] + "" + text[text.index("nik")+2]
        print(nik)
    except:
        nik = result.splitlines()[2]
        r1 = re.findall(r"\d[-/]?", nik)
        strs = ""
        for r in r1:
            strs += ''.join(r)
        nik = strs
        print(nik)

    nama = text[text.index("nama")+1]+" "+text[text.index("nama")+2]
    print(nama)
    try:

        lahir = text[text.index("lahir")+1]+" "+text[text.index("lahir")+2]
        print(lahir)
    except:
        lahir = ""
        print("Lahir Not Found")

    try:
        kelamin = text[text.index("kelamin")+1]
        if(kelamin == "-"):
            kelamin = text[text.index("kelamin")+2]
        print(kelamin)
    except:
        kelamin = result.splitlines()[5]
        print(kelamin)
    try:
        if("darah" in text):
            darah = text[text.index("darah")+1]
            print(darah)
        elif("goldarah" in text):
            darah = text[text.index("goldarah")+1]
            print(darah)
    except:
        darah = ""
        print(darah)

    try:
        if("alamat" in text):
            alamat = text[text.index(
                "alamat")+1]+" "+text[text.index("alamat")+2]+" "+text[text.index("alamat")+3]
            print(alamat)
    except:
        alamat = ""
        print("Not found "+alamat)

    try:
        if("rt/rw" in text):
            rt_rw = text[text.index(
                "rt/rw")+1]+" "+text[text.index("rt/rw")+2]+" "+text[text.index("rt/rw")+3]
            print(rt_rw)
        elif("rtrw" in text):
            rt_rw = text[text.index(
                "rtrw")+1]+" "+text[text.index("rtrw")+2]+" "+text[text.index("rtrw")+3]
            print(rt_rw)
        elif("rurw" in text):
            rt_rw = text[text.index(
                "rurw")+1]+" "+text[text.index("rurw")+2]+" "+text[text.index("rurw")+3]
            print(rt_rw)
    except:
        rt_rw = ""
        print("Not found "+rt_rw)

    try:
        if("kel/desa" in text):
            kel_desa = text[text.index("kel/desa")+1]
            print(kel_desa)
        elif("ketdesa" in text):
            kel_desa = text[text.index("ketdesa")+1] + \
                " "+text[text.index("ketdesa")+2]
            print(kel_desa)
        elif("kelidesa" in text):
            kel_desa = text[text.index("kelidesa")+1] + \
                " "+text[text.index("kelidesa")+2]
            print(kel_desa)
        elif("keldesa" in text):
            kel_desa = text[text.index("keldesa")+1] + \
                " "+text[text.index("keldesa")+2]
            print(kel_desa)
        else:
            kel_desa = "Not found"
    except:
        kel_desa = ""
        print("Not found "+kel_desa)

    try:
        if("kecamatan" in text):
            kecamatan = text[text.index("kecamatan")+1] + \
                " "+text[text.index("kecamatan")+2]
            print(kecamatan)
    except:
        kecamatan = ""
        print("Not found "+kecamatan)

    try:
        if("agama" in text):
            agama = text[text.index("agama")+1]
            print(agama)
    except:
        agama = ""
        print("Not found "+agama)

    try:
        if("perkawinan" in text):
            perkawinan = text[text.index(
                "perkawinan")+1]+" "+text[text.index("perkawinan")+2]
            print(perkawinan)
    except:
        pekerjaan = ""
        print("Not found "+pekerjaan)
    try:
        if("pekerjaan" in text):
            pekerjaan = text[text.index("pekerjaan")+1]
            print(pekerjaan)
        elif("pokerjaan" in text):
            pekerjaan = text[text.index("pokerjaan")+1]
            print(pekerjaan)
        else:
            pekerjaan = ""
    except:
        pekerjaan = ""
        print("Not found "+pekerjaan)

    try:
        if("kewarganegaraan" in text):
            kewarganegaraan = text[text.index("kewarganegaraan")+1]
            print(kewarganegaraan)
        else:
            kewarganegaraan = ""
    except:
        kewarganegaraan = ""
        print("Not found "+kewarganegaraan)

    try:
        if("berlaku" in text):
            hingga = text[text.index(
                "berlaku")+1]+" "+text[text.index("berlaku")+2]+" "+text[text.index("berlaku")+3]
            matches = datefinder.find_dates(hingga)
            match_date = ""
            for match in matches:
                if(match.date()):
                    match_date += match.date().strftime("%d %b %Y")
                    print("matched date: "+match_date)
                    hingga = match_date
                    break

            hingga = hingga.replace("hingga", "")
            print(hingga)
        elif ("hingga" in text):
            hingga = text[text.index("hingga")+1] + \
                " "+text[text.index("hingga")+2]
            matches = datefinder.find_dates(hingga)
            match_date = ""
            for match in matches:
                if(match.date() == None):
                    match_date += hingga
                else:
                    match_date += match.date().strftime("%d %b %Y")
                    break
            hingga = match_date
            print(hingga)
    except:
        hingga = ""
        print("Not found "+hingga)

    user_defined_label = ['provinsi', 'nik', 'nama', 'lahir', 'kelamin', 'darah', 'alamat', 'rt/rw', 'kel/desa',
                          'kecamatan', 'agama', 'status_perkawinan', 'pekerjaan', 'kewarganegaraan', 'berlaku_hingga']
    result = [provinsi, nik, nama, lahir, kelamin, darah, alamat, rt_rw,
              kel_desa, kecamatan, agama, perkawinan, pekerjaan, kewarganegaraan, hingga]

    ori_img = cv2.resize(img, (1000, 700))
    y = 650  # x
    x = 150  # y
    h = 1000
    w = 600
    image_cropped = ori_img[x:w, y:h]

    # rename
    cv2.imwrite(IMAGE_SAVE_FOLDER+result[1]+".jpg", image_cropped)

    face_id_url = IMAGE_SAVE_FOLDER+result[1]+".jpg"

    avgConf_str = calc_conf(img_data)

    my_array = [user_defined_label, result]

    pairs = zip(my_array[0], my_array[1])
    json_values = ('"{}": "{}"'.format(label, value)
                   for label, value in pairs)
    my_string = '{' + ', '.join(json_values) + '}'

    export_data = json.loads(my_string)

    # print(export_data)

    data_file = ""

    try:

        with open('./Exports/export-data.csv', newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)

        if(list(export_data.keys()) == row1):
            print("same header")
            data_file = open('./Exports/export-data.csv', 'a', newline='')

            csv_writer = csv.writer(data_file)

            csv_writer.writerow(export_data.values())

        else:
            print("different header")
            data_file = open('./Exports/export-data.csv', 'w', newline='')
            csv_writer = csv.writer(data_file)
            header = export_data.keys()
            csv_writer.writerow(header)
            csv_writer.writerow(export_data.values())

    except:
        print("exception error")

        # now we will open a file for writing
        data_file = open('./Exports/export-data.csv', 'w', newline='')

        # create the csv writer object
        csv_writer = csv.writer(data_file)

        # Writing headers of CSV file
        header = export_data.keys()
        csv_writer.writerow(header)

        # Writing data of CSV file
        csv_writer.writerow(export_data.values())

    finally:
        data_file.close()

    print("\nExtraction Confidence Level: "+avgConf_str)

    json_formatted_str = json.dumps(export_data, indent=2)

    print(json_formatted_str)

    img = cv2.resize(img, (680, 450))
    #cv2.imshow("ic_image_output",img);cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
    processed_image = IMAGE_SAVE_FOLDER+result[1]+"_card.jpg"
    cv2.imwrite(processed_image, img)

    print("--- %s seconds ---" % (time.time() - start_time))
    return jsonify(Output=export_data, upload_filepath=upload_filepath, face_id_url=face_id_url, conf_lv=avgConf_str, doc_type=doc_type, processed_image=processed_image)


def id_img_data_text(img_data):
    img_data = img_data.dropna()
    n_boxes = len(img_data['text'])
    pd_index = []
    for i in range(n_boxes):
        pd_index.append(i)
    s = pd.Series(pd_index)
    img_data = img_data.set_index([s])
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
    return text


def id_clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('—', '-', text)
    text = re.sub('…', '', text)
    text = re.sub('“', '', text)
    remove = string.punctuation
    # don't remove hyphens
    remove = remove.replace("-", "")
    remove = remove.replace("/", "")
    pattern = r"[{}]".format(remove)  # create the pattern
    text = re.sub(pattern, "", text)
    #text = re.sub('\w*\d\w*', '', text)
    text = os.linesep.join([s for s in text.splitlines() if s])
    return text

# Apply a second round of cleaning


def id_clean_text_round2(text):
    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)
    return text


def img_data_text(img_data):
    n_boxes = len(img_data['text'])
    pd_index = []
    for i in range(n_boxes):
        pd_index.append(i)
    s = pd.Series(pd_index)
    img_data = img_data.set_index([s])

    text = ""
    min_conf = 90

    current_block = int(img_data.loc[0, 'block_num'])
    for i in range(n_boxes):
        if int(img_data["conf"][i]) >= min_conf:
            if (int(img_data.loc[i, 'block_num']) == current_block):
                text += img_data.loc[i, 'text'] + " "

            else:
                current_block += 1
                text += "\n"
                text += img_data.loc[i, 'text'] + " "

        else:
            if (int(img_data.loc[i, 'block_num']) is not current_block):
                current_block += 1
                text += "\n"

    return text


def mykad_clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('—', '', text)
    text = re.sub('…', '', text)
    text = re.sub('“', '', text)
    remove = string.punctuation
    # don't remove hyphens
    #remove = remove.replace("-", "")
    #remove = remove.replace("/", "")
    pattern = r"[{}]".format(remove)  # create the pattern
    text = re.sub(pattern, "", text)
    #text = re.sub('\w*\d\w*', '', text)
    text = os.linesep.join([s for s in text.splitlines() if s])
    return text


def calc_conf(img_data):
    total = img_data.sum(axis=0)
    count = img_data.count(axis=0)
    avgConf = total.conf / count.conf
    avgConf_str = str(("%.2f" % avgConf) + " %")
    return avgConf_str


pat = r"""
\b                      # word boundary
(?P<birthdate>\d{6})    # named group capture of birthdate, six digits
-?                      # optional -
(?P<birthplace>\d{2})   # named group, birthplace, 2 digits
-?                      # optional -
\d{3}                   # next 3 digits
(?P<gender>\d)          # capture last digit representing gender
\b                      # word boundary
"""

vpo = re.compile(pat, re.VERBOSE)

codes = [('01', '21', '22', '23', '24'), ('02', '25', '26', '27'), ('03', '28', '29'),
         ('04', '30'), ('05', '31', '59'), ('06',
                                            '32', '33'), ('07', '34', '35'),
         ('08', '36', '37', '38', '39'), ('09',
                                          '40'), ('10', '41', '42', '43', '44'),
         ('11', '45', '46'), ('12', '47', '48',
                              '49'), ('13', '50', '51', '52', '53'),
         ('14', '54', '55', '56', '57'), ('15', '58'), ('16',), ('82',)]

# place of birth
place = ('Johor', 'Kedah', 'Kelantan', 'Malacca', 'Negri Sembilan',
         'Pahang', 'Penang',  'Perak',  'Perlis', 'Selangor', 'Terengganu', 'Sabah',
         'Sarawak', 'Kuala Lumpur', 'Labuan', 'Putrajaya', 'Unknown')


def get_place(code):
    for i, item in enumerate(codes):
        if code in item:
            return place[i]
    return None


def get_gender(n): return 'Male' if int(n) % 2 else 'Female'


def parse_ic(ic):
    m = vpo.search(ic)
    if m:
        return(m.group('birthdate'),
               get_place(m.group('birthplace')),
               get_gender(m.group('gender')))


@app.route("/processmykad")
def process_mykad():
    doc_type = "Malaysia_IC"
    upload_filepath = request.args.get('upload_filepath')

    img = cv2.imread(upload_filepath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

    threshed = cv2.resize(threshed, (1000, 700))

    y = 0
    x = 0
    h = 650  # w
    w = 1000  # h
    ic_details_cropped = (threshed)[x:w, y:h]

    img_data = pytesseract.image_to_data(
        ic_details_cropped, lang="msa", output_type='data.frame')
    img_data = img_data.dropna()

    result = img_data_text(img_data)

    result = mykad_clean_text_round1(result)

    result = id_clean_text_round2(result)

    # print(result)

    result = result.split("\n")

    # Analyze IC No.

    ic_str = parse_ic(result[0])[0]

    ic_str = ic_str[:2] + '-' + ic_str[2:]
    ic_str = ic_str[:5] + '-' + ic_str[5:]

    matches = datefinder.find_dates(ic_str)
    match_date = ""
    for match in matches:
        if(match.date() == None):
            match_date = ""
        else:
            match_date += match.date().strftime("%d %b %Y")
            # print(match_date)
            break

    result.append(match_date)
    result.append(parse_ic(result[0])[1])
    result.append(parse_ic(result[0])[2])

    avgConf_str = calc_conf(img_data)

    ori_img = cv2.resize(img, (1000, 700))
    y = 650  # x
    x = 150  # y
    h = 1000
    w = 600
    image_cropped = ori_img[x:w, y:h]

    # rename
    cv2.imwrite(IMAGE_SAVE_FOLDER+result[2][:-1]+".jpg", image_cropped)

    face_id_url = IMAGE_SAVE_FOLDER+result[2][:-1]+".jpg"

    # plt.imshow(image_cropped[...,::-1])

    ic_key = ["ic_no", "name", "address",
              "birthdate", "birthplace", "gender"]

    my_array = [ic_key, result]

    pairs = zip(my_array[0], my_array[1])
    json_values = ('"{}": "{}"'.format(label, value)
                   for label, value in pairs)
    my_string = '{' + ', '.join(json_values) + '}'

    export_data = json.loads(my_string)

    json_formatted_str = json.dumps(export_data, indent=2)

    print(json_formatted_str)
    ic_saved_image_path = IMAGE_SAVE_FOLDER+result[0]+".jpg"
    ic_saved_image_path = re.sub(" ", "", ic_saved_image_path)
    cv2.imwrite(ic_saved_image_path, img)
    print("--- %s seconds ---" % (time.time() - start_time))
    return jsonify(process_mykad_output=export_data, upload_filepath=upload_filepath, doc_type=doc_type, conf_lv=avgConf_str, face_id_url=face_id_url, ic_saved_image_path=ic_saved_image_path)


@app.route('/processTaxInvoice')
def processTaxInvoice():

    upload_filepath = request.args.get('upload_filepath')
    user_defined_list = request.args.get('predefined')
    upload_filename = "tax_invoice"

    if upload_filepath.endswith('pdf'):
        pages = convert_from_path(upload_filepath)

        image_counter = 1
        for page in pages:
            page.save(IMAGE_SAVE_FOLDER +
                      upload_filename+"-pg"+str(image_counter)+".jpg", 'JPEG')

            qr = decode(Image.open(IMAGE_SAVE_FOLDER +
                                   upload_filename+"-pg" +
                                   str(image_counter)+".jpg"
                                   ))
            image_counter += 1
    else:
        qr = decode(Image.open(
                    upload_filepath))

    print(qr[0].data)

    url = qr[0].data
    document = requests.get(url)

    xml_content = BeautifulSoup(document.content, "xml")
    print(xml_content)

    #user_defined_list = """nomorFaktur,tanggalFaktur,npwpPenjual,namaPenjual,alamatPenjual,npwpLawanTransaksi,namaLawanTransaksi,alamatLawanTransaksi,jumlahDpp,jumlahPpn,jumlahPpnBm,statusApproval,statusFaktur,referensi"""
    user_defined_list = user_defined_list.split(",")
    result = []

    for predefined in user_defined_list:

        if predefined == "nomorFaktur":
            nomorFaktur = xml_content.find_all('kdJenisTransaksi')[0].get_text(
            )+xml_content.find_all('fgPengganti')[0].get_text()+xml_content.find_all('nomorFaktur')[0].get_text()
            result.append(nomorFaktur)
        if predefined == "tanggalFaktur":
            tanggalFaktur = xml_content.find_all('tanggalFaktur')[0].get_text()
            result.append(tanggalFaktur)
        if predefined == "npwpPenjual":
            npwpPenjual = xml_content.find_all('npwpPenjual')[0].get_text()
            result.append(npwpPenjual)
        if predefined == "namaPenjual":
            namaPenjual = xml_content.find_all('namaPenjual')[0].get_text()
            result.append(namaPenjual)
        if predefined == "alamatPenjual":
            alamatPenjual = xml_content.find_all('alamatPenjual')[0].get_text()
            result.append(alamatPenjual)
        if predefined == "npwpLawanTransaksi":
            npwpLawanTransaksi = xml_content.find_all(
                'npwpLawanTransaksi')[0].get_text()
            result.append(npwpLawanTransaksi)
        if predefined == "namaLawanTransaksi":
            namaLawanTransaksi = xml_content.find_all(
                'namaLawanTransaksi')[0].get_text()
            result.append(namaLawanTransaksi)
        if predefined == "alamatLawanTransaksi":
            alamatLawanTransaksi = xml_content.find_all(
                'alamatLawanTransaksi')[0].get_text()
            result.append(alamatLawanTransaksi)
        if predefined == "jumlahDpp":
            jumlahDpp = xml_content.find_all('jumlahDpp')[0].get_text()
            result.append(jumlahDpp)
        if predefined == "jumlahPpn":
            jumlahPpn = xml_content.find_all('jumlahPpn')[0].get_text()
            result.append(jumlahPpn)
        if predefined == "jumlahPpnBm":
            jumlahPpnBm = xml_content.find_all('jumlahPpnBm')[0].get_text()
            result.append(jumlahPpnBm)
        if predefined == "statusApproval":
            statusApproval = xml_content.find_all(
                'statusApproval')[0].get_text()
            result.append(statusApproval)
        if predefined == "statusFaktur":
            statusFaktur = xml_content.find_all('statusFaktur')[0].get_text()
            result.append(statusFaktur)
        if predefined == "referensi":
            referensi = xml_content.find_all('referensi')[0].get_text()
            result.append(referensi)

    my_array = [user_defined_list, result]

    pairs = zip(my_array[0], my_array[1])
    json_values = ('"{}": "{}"'.format(label, value)
                   for label, value in pairs)
    my_string = '{' + ', '.join(json_values) + '}'

    my_string = my_string.replace("\n", "")

    export_data = json.loads(my_string)

    json_formatted_str = json.dumps(export_data, indent=2)

    print(json_formatted_str)

    conf_lv = "95%"
    img_src = IMAGE_SAVE_FOLDER+nomorFaktur+"-pg1.jpg"
    page.save(img_src, "JPEG")

    bf = dumps(parker.data(fromstring(str(xml_content))), indent=2)
    bf = json.loads(bf)

    table_header = list(bf["detailTransaksi"][0].keys())

    row_item = []
    row_data = []

    for i in range(0, len(bf["detailTransaksi"])):
        for j in range(0, len(bf["detailTransaksi"][i])):
            row_item.append((bf["detailTransaksi"][i][table_header[j]]))
        row_data.append(row_item)
        row_item = []
    table_json = """
    {
    "table_header":"""+str(table_header)+""",
    "row_data":"""+str(row_data)+"""
    }

    """
    table_json = table_json.replace("'", "\"")

    table_json = json.loads(table_json)

    print("--- %s seconds ---" % (time.time() - start_time))
    return jsonify(result_list=export_data, user_defined_list=user_defined_list, conf_lv=conf_lv, img_src=img_src, result=result, table_json=table_json)


@app.route('/processBirthCert')
def processBirthCert():
    upload_filepath = request.args.get("upload_filepath")
    user_defined_list = request.args.get("predefined")

    img = cv2.imread(upload_filepath)

    img = cv2.resize(img, (1300, 1800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 150, 255, cv2.THRESH_TRUNC)

    img_data = pytesseract.image_to_data(
        threshed, lang="ind+eng", config="--psm 6", output_type="data.frame")
    img_data = img_data.dropna()
    img_data.to_csv(
        r"C:\Users\HPC1\Documents\tvextract-master\process\Documents\birth-cert\coor.csv")
    avgConf_str = calc_conf(img_data)
    print(avgConf_str)

    text = id_img_data_text(img_data)
    text = clean_text_round1(text)
    text = clean_text_round2(text)
    print(text)
    text = text.split()
    print(text)

    user_defined_list = user_defined_list.split(",")

    result = []

    for predefined in user_defined_list:
        if predefined == "Kependudukan.":
            try:
                found_word = text[text.index(predefined)+1]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "AL.":
            try:
                found_word = text[text.index(predefined)+1]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "NEGARA":
            try:
                found_word = text[text.index(predefined)+1]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "Kelahiran":
            try:
                found_word = text[text.index(predefined)+2]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "bahwa":
            try:
                found_word = text[text.index(
                    predefined)+2:text.index(predefined)+4]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "born":
            try:
                found_word = text[text.index(predefined)+1]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "anak":
            try:
                found_word = text[text.index(
                    predefined)+2:text.index(predefined)+4]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "AYAH":
            try:
                found_word = text[text.index(predefined)+1]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "IBU":
            try:
                found_word = text[text.index(predefined)+1]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "dikeluarkan":
            try:
                found_word = text[text.index(predefined)+2]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "tanggal":
            try:
                found_word = text[text.index(
                    predefined)+1:text.index(predefined)+4]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "Tahun":
            try:
                found_word = text[text.index(
                    predefined)+1:text.index(predefined)+5]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "Kepala":
            try:
                found_word = text[text.index(
                    predefined)+1:text.index(predefined)+4]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "NIP.":
            try:
                found_word = text[text.index(predefined)+1]
                result.append(found_word)
            except:
                result.append("")

    img_src = IMAGE_SAVE_FOLDER+result[0]+".jpg"
    cv2.imwrite(img_src, img)

    my_array = [user_defined_list, result]

    pairs = zip(my_array[0], my_array[1])
    json_values = ('"{}": "{}"'.format(label, value)
                   for label, value in pairs)
    my_string = '{' + ', '.join(json_values) + '}'

    my_string = my_string.replace("\n", "")

    export_data = json.loads(my_string)

    json_formatted_str = json.dumps(export_data, indent=2)

    print(json_formatted_str)

    return jsonify(result_list=export_data, upload_filepath=upload_filepath, user_defined_list=user_defined_list, result=result, conf_lv=avgConf_str, img_src=img_src)


def do_clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    #text = text.upper()
    text = re.sub('\[*—?:§\]', '', text)
    text = re.sub('§', '', text)
    text = re.sub("\)", '', text)
    text = re.sub("\(", '', text)
    #text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    remove = string.punctuation
    remove = remove.replace(".", "")  # don't remove hyphens
    remove = remove.replace("-", "")
    remove = remove.replace("/", "")
    pattern = r"[{}]".format(remove)  # create the pattern
    text = re.sub(pattern, "", text)

    return text

# Apply a second round of cleaning


def do_clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[:—‘’“”…]', '', text)
    text = re.sub("\[", '', text)
    text = re.sub("\(", '', text)
    text = re.sub('<br>', '', text)  # remove for jsonify output
    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)

    return text


def do_img_data_text(img_data):
    img_data = img_data.dropna()
    n_boxes = len(img_data['text'])
    pd_index = []
    for i in range(n_boxes):
        pd_index.append(i)
    s = pd.Series(pd_index)
    img_data = img_data.set_index([s])
    num = 2
    num2 = 2
    num3 = 2
    text = ""
    min_conf = 0
    for i in range(n_boxes):
        if int(img_data["conf"][i]) >= min_conf:
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
        # else:

    return text


@app.route('/processDeliveryOrder')
def processDeliveryOrder():
    upload_filepath = request.args.get("upload_filepath")

    img = cv2.imread(upload_filepath)
    height = img.shape[0]
    width = img.shape[1]
    img = cv2.resize(img, (width*2, height*2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    # cv2.imshow("thresh",gray)
    # cv2.waitKey(0)

    img_data = pytesseract.image_to_data(
        gray, lang="eng+ind", config="--psm 6", output_type="data.frame")
    img_data = img_data.dropna()
    img_data.to_csv(EXPORT_FOLDER_PATH+"do_coordinates.csv")
    avgConf_str = calc_conf(img_data)
    print(avgConf_str)

    text = do_img_data_text(img_data)
    text = do_clean_text_round1(text)
    text = do_clean_text_round2(text)
    text_par = text
    print(text_par)
    text = text.split()

    user_defined_list = "Do,Date,Phone,Fax,Cust,Address,Code,PO"
    user_defined_list = user_defined_list.split(",")

    result = []

    for predefined in user_defined_list:
        if predefined == "Do":
            try:
                found_word = text[text.index(predefined)+2]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "PO":
            try:
                found_word = text[text.index(predefined)+2]
                result.append(found_word)
            except:
                result.append("")
        if predefined == "Date":
            try:
                found_word = text[text.index(
                    predefined)+1:text.index(predefined)+4]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "Phone":
            try:
                found_word = text[text.index(
                    predefined)+1:text.index(predefined)+4]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "Fax":
            try:
                found_word = text[text.index(
                    predefined)+1:text.index(predefined)+5]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "Cust":
            try:
                found_word = text[text.index(
                    predefined)+2:text.index(predefined)+6]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "Address":
            try:
                found_word = text[text.index(
                    predefined)+1:text.index(predefined)+10]
                found_word = " ".join(found_word)
                result.append(found_word)
            except:
                result.append("")
        if predefined == "Code":
            try:
                found_word = text_par.splitlines()[-1]
                result.append(found_word)
            except:
                result.append("")

    img_src = IMAGE_SAVE_FOLDER+result[0]+".jpg"
    cv2.imwrite(img_src, img)

    my_array = [user_defined_list, result]

    pairs = zip(my_array[0], my_array[1])
    json_values = ('"{}": "{}"'.format(label, value)
                   for label, value in pairs)
    my_string = '{' + ', '.join(json_values) + '}'

    export_data = json.loads(my_string)

    json_formatted_str = json.dumps(export_data, indent=2)

    print(json_formatted_str)
    pdf = pytesseract.image_to_pdf_or_hocr(
        img, extension='pdf', lang="eng+ind", config="--psm 6")
    with open(EXPORT_FOLDER_PATH+"OCR-ed_PDF.pdf", 'w+b') as f:
        f.write(pdf)  # pdf type is bytes by default

    f.close()

    tabula.convert_into(EXPORT_FOLDER_PATH+"OCR-ed_PDF.pdf", EXPORT_FOLDER_PATH+"do-body.csv",
                        output_format="csv", pages='all', area=(372.253, 22.559, 745.573, 1735.439))

    try:
        df = pd.read_csv(EXPORT_FOLDER_PATH+"do-body.csv", encoding='latin1')
    except:
        df = pd.DataFrame()

    df['Item Name'] = df[df.columns[1:2]].apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1)
    df = df.drop(columns=[df.columns[1]])
    df.to_csv(EXPORT_FOLDER_PATH+"do-body.csv", index=False)

    exampleFile = open(EXPORT_FOLDER_PATH+"do-body.csv")
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

    return jsonify(result_list=export_data, upload_filepath=upload_filepath, user_defined_list=user_defined_list, result=result, conf_lv=avgConf_str, img_src=img_src)


if __name__ == '__main__':
    app.run(port=8011, debug=True, threaded=True)
