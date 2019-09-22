from flask import Flask, render_template, request, send_from_directory, url_for
import os
import socket
import sqlite3
import time
import json
import requests
from pandas.io.json import json_normalize
import pickle
import pandas as pd,json
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
import numpy as np
import os.path, time


app = Flask(__name__ , static_url_path='/static', static_folder="static")
lr_loaded = joblib.load("DigiReciB.pkl") # Load "model.pkl"
engine = create_engine('sqlite:///DigiReci.db', echo=False)


@app.route('/apiPredict', methods=['POST'])
def api_Predict():
    json_ = request.json
    t = pd.DataFrame.from_dict(json_)
    #t.to_sql('Train', con=engine, if_exists='append',index=False)
    #t.to_csv("Req.csv",mode='a')
    t = t.loc[:, 'pixel0':]
    result = lr_loaded.predict(t)
    return "Predicted Digit : " + str(result)

@app.route('/apiPredictPercentages', methods=['POST'])
def api_PredictPercentages():
    json_ = request.json
    t = pd.DataFrame.from_dict(json_)
    t = t.loc[:, 'pixel0':]
    label = 0
    a = lr_loaded.predict_proba(t)
    out = ''
    for x in np.nditer(a):
        b = format(x * 100, 'f')
        out = out + '(' + str(label) + ' ' + str("{:00.3f}".format(float(b)).zfill(6)) + ')' + '\n'
        label += 1
    return out


@app.route('/apiTrain', methods=['POST'])
def api_Train():
    json_ = request.json
    t = pd.DataFrame.from_dict(json_)
    t.to_sql('Train', con=engine, if_exists='append',index=False)
    t = t.loc[:, 'pixel0':]
    result = lr_loaded.predict(t)
    #retrainModel()
    return "Predicted Digit : " + str(result)

@app.route('/retrainModel', methods=['POST'])
def retrainModel():
    retrainModel()
    return "Model has been updated."

@app.route('/modelAccuracyHC', methods=['GET'])
def modelAccuracyHC():
    data = request.get_json(force=True)
    name = data['name']
    return "hello {0}".format(name)

@app.route('/trainingRecordHC', methods=['GET'])
def trainingRecordHC():
    conn = sqlite3.connect("DigiReci.db")
    t = pd.read_sql_query("SELECT LABEL,COUNT(1) AS RECORDS FROM TRAIN GROUP BY LABEL", conn)
    t = str(list(t['RECORDS']))
    t = t.replace("[","").replace("]","")
    return t

@app.route('/modelSomething', methods=['GET'])
def modelSomething():
    conn = sqlite3.connect("DigiReci.db")
    t = pd.read_sql_query("SELECT * FROM TEST", conn)
    lr_loaded = joblib.load("DigiReciB.pkl")  # Load "model.pkl"
    d = t
    a = pd.DataFrame(t['label'])
    d = d.loc[:, 'pixel0':]
    b = pd.DataFrame(list(lr_loaded.predict(d)))
    b.columns = ['predicted']
    a.columns = ['actual']
    myret = a.join(b)
    overallAccuracy = np.mean(myret.actual.astype(int) == myret.predicted.astype(int))
    overallAccuracy = round(overallAccuracy * 100, 2)
    return str(overallAccuracy)

@app.route('/pixelsdraw', methods=['GET'])
def pixelsdraw():
    conn = sqlite3.connect("DigiReci.db")
    t = pd.read_sql_query("SELECT * FROM TRAIN", conn)
    total = 0
    for x in list(t.sum(axis=1)):
        total = total + x
    return str(total)

@app.route('/trainingcount', methods=['GET'])
def trainingcount():
    conn = sqlite3.connect("DigiReci.db")
    t = pd.read_sql_query("SELECT count(*) as x FROM TRAIN", conn)
    t = str(list(t['x'])).replace('[', '').replace(']', '')
    return t

@app.route('/labelaccuracy', methods=['GET'])
def labelaccuracy():
    # accuracy by label
    conn = sqlite3.connect("DigiReci.db")
    t = pd.read_sql_query("SELECT * FROM TEST", conn)
    lr_loaded = joblib.load("DigiReciB.pkl")  # Load "model.pkl"
    d = t
    a = pd.DataFrame(t['label'])
    d = d.loc[:, 'pixel0':]
    b = pd.DataFrame(list(lr_loaded.predict(d)))
    b.columns = ['predicted']
    a.columns = ['actual']
    myret = a.join(b)
    myret['CNT'] = np.where(myret['actual'].astype(int) == myret['predicted'].astype(int), 1, 0)
    pd.DataFrame(myret['actual'].value_counts()).sort_index()
    pd.DataFrame(myret.groupby('actual')['CNT'].sum())
    aa = pd.DataFrame(myret['actual'].value_counts()).sort_index()
    bb = pd.DataFrame(myret.groupby('actual')['CNT'].sum())
    cc = aa.join(bb)
    cc['Percentage'] = round((cc['CNT'] / cc['actual']) * 100, 2)
    cc = cc.drop('CNT', axis=1)
    cc['label'] = cc.index.values
    cc = cc.drop('actual', axis=1)
    cc = cc[['label', 'Percentage']]
    cc = cc.drop('label', axis=1)
    cc = str(list(cc['Percentage'])).replace('[', '').replace(']', '')
    return cc

@app.route('/moddate', methods=['GET'])
def moddate():
    moddate = time.ctime(os.path.getmtime('DigiReciB.pkl'))
    return str(moddate)

def loadnewPickle():
    lr_loaded = joblib.load("DigiReciB.pkl")  # Load "model.pkl"

def retrainModel():
    conn = sqlite3.connect("DigiReci.db")
    train = pd.read_sql_query("select * from Train;", conn)
    #train.loc[:, 'pixel0':] = train.loc[:, 'pixel0':]
    lr = LogisticRegression(random_state=1)
    lr.fit(train.loc[:, 'pixel0':], train.loc[:, 'label'])
    model_file_pickle = open('DigiReciB.pkl', 'wb')
    pickle.dump(lr, model_file_pickle)
    model_file_pickle.close()
    #loadnewPickle()

def create():
    db_path = r'DigiReci.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS Train(Id INTEGER PRIMARY KEY,label INTEGER,pixel0 INTEGER,pixel1 INTEGER,pixel2 INTEGER,pixel3 INTEGER,pixel4 INTEGER,pixel5 INTEGER,pixel6 INTEGER,pixel7 INTEGER,pixel8 INTEGER,pixel9 INTEGER,pixel10 INTEGER,pixel11 INTEGER,pixel12 INTEGER,pixel13 INTEGER,pixel14 INTEGER,pixel15 INTEGER,pixel16 INTEGER,pixel17 INTEGER,pixel18 INTEGER,pixel19 INTEGER,pixel20 INTEGER,pixel21 INTEGER,pixel22 INTEGER,pixel23 INTEGER,pixel24 INTEGER,pixel25 INTEGER,pixel26 INTEGER,pixel27 INTEGER,pixel28 INTEGER,pixel29 INTEGER,pixel30 INTEGER,pixel31 INTEGER,pixel32 INTEGER,pixel33 INTEGER,pixel34 INTEGER,pixel35 INTEGER,pixel36 INTEGER,pixel37 INTEGER,pixel38 INTEGER,pixel39 INTEGER,pixel40 INTEGER,pixel41 INTEGER,pixel42 INTEGER,pixel43 INTEGER,pixel44 INTEGER,pixel45 INTEGER,pixel46 INTEGER,pixel47 INTEGER,pixel48 INTEGER,pixel49 INTEGER,pixel50 INTEGER,pixel51 INTEGER,pixel52 INTEGER,pixel53 INTEGER,pixel54 INTEGER,pixel55 INTEGER,pixel56 INTEGER,pixel57 INTEGER,pixel58 INTEGER,pixel59 INTEGER,pixel60 INTEGER,pixel61 INTEGER,pixel62 INTEGER,pixel63 INTEGER,pixel64 INTEGER,pixel65 INTEGER,pixel66 INTEGER,pixel67 INTEGER,pixel68 INTEGER,pixel69 INTEGER,pixel70 INTEGER,pixel71 INTEGER,pixel72 INTEGER,pixel73 INTEGER,pixel74 INTEGER,pixel75 INTEGER,pixel76 INTEGER,pixel77 INTEGER,pixel78 INTEGER,pixel79 INTEGER,pixel80 INTEGER,pixel81 INTEGER,pixel82 INTEGER,pixel83 INTEGER,pixel84 INTEGER,pixel85 INTEGER,pixel86 INTEGER,pixel87 INTEGER,pixel88 INTEGER,pixel89 INTEGER,pixel90 INTEGER,pixel91 INTEGER,pixel92 INTEGER,pixel93 INTEGER,pixel94 INTEGER,pixel95 INTEGER,pixel96 INTEGER,pixel97 INTEGER,pixel98 INTEGER,pixel99 INTEGER,pixel100 INTEGER,pixel101 INTEGER,pixel102 INTEGER,pixel103 INTEGER,pixel104 INTEGER,pixel105 INTEGER,pixel106 INTEGER,pixel107 INTEGER,pixel108 INTEGER,pixel109 INTEGER,pixel110 INTEGER,pixel111 INTEGER,pixel112 INTEGER,pixel113 INTEGER,pixel114 INTEGER,pixel115 INTEGER,pixel116 INTEGER,pixel117 INTEGER,pixel118 INTEGER,pixel119 INTEGER,pixel120 INTEGER,pixel121 INTEGER,pixel122 INTEGER,pixel123 INTEGER,pixel124 INTEGER,pixel125 INTEGER,pixel126 INTEGER,pixel127 INTEGER,pixel128 INTEGER,pixel129 INTEGER,pixel130 INTEGER,pixel131 INTEGER,pixel132 INTEGER,pixel133 INTEGER,pixel134 INTEGER,pixel135 INTEGER,pixel136 INTEGER,pixel137 INTEGER,pixel138 INTEGER,pixel139 INTEGER,pixel140 INTEGER,pixel141 INTEGER,pixel142 INTEGER,pixel143 INTEGER,pixel144 INTEGER,pixel145 INTEGER,pixel146 INTEGER,pixel147 INTEGER,pixel148 INTEGER,pixel149 INTEGER,pixel150 INTEGER,pixel151 INTEGER,pixel152 INTEGER,pixel153 INTEGER,pixel154 INTEGER,pixel155 INTEGER,pixel156 INTEGER,pixel157 INTEGER,pixel158 INTEGER,pixel159 INTEGER,pixel160 INTEGER,pixel161 INTEGER,pixel162 INTEGER,pixel163 INTEGER,pixel164 INTEGER,pixel165 INTEGER,pixel166 INTEGER,pixel167 INTEGER,pixel168 INTEGER,pixel169 INTEGER,pixel170 INTEGER,pixel171 INTEGER,pixel172 INTEGER,pixel173 INTEGER,pixel174 INTEGER,pixel175 INTEGER,pixel176 INTEGER,pixel177 INTEGER,pixel178 INTEGER,pixel179 INTEGER,pixel180 INTEGER,pixel181 INTEGER,pixel182 INTEGER,pixel183 INTEGER,pixel184 INTEGER,pixel185 INTEGER,pixel186 INTEGER,pixel187 INTEGER,pixel188 INTEGER,pixel189 INTEGER,pixel190 INTEGER,pixel191 INTEGER,pixel192 INTEGER,pixel193 INTEGER,pixel194 INTEGER,pixel195 INTEGER,pixel196 INTEGER,pixel197 INTEGER,pixel198 INTEGER,pixel199 INTEGER,pixel200 INTEGER,pixel201 INTEGER,pixel202 INTEGER,pixel203 INTEGER,pixel204 INTEGER,pixel205 INTEGER,pixel206 INTEGER,pixel207 INTEGER,pixel208 INTEGER,pixel209 INTEGER,pixel210 INTEGER,pixel211 INTEGER,pixel212 INTEGER,pixel213 INTEGER,pixel214 INTEGER,pixel215 INTEGER,pixel216 INTEGER,pixel217 INTEGER,pixel218 INTEGER,pixel219 INTEGER,pixel220 INTEGER,pixel221 INTEGER,pixel222 INTEGER,pixel223 INTEGER,pixel224 INTEGER,pixel225 INTEGER,pixel226 INTEGER,pixel227 INTEGER,pixel228 INTEGER,pixel229 INTEGER,pixel230 INTEGER,pixel231 INTEGER,pixel232 INTEGER,pixel233 INTEGER,pixel234 INTEGER,pixel235 INTEGER,pixel236 INTEGER,pixel237 INTEGER,pixel238 INTEGER,pixel239 INTEGER,pixel240 INTEGER,pixel241 INTEGER,pixel242 INTEGER,pixel243 INTEGER,pixel244 INTEGER,pixel245 INTEGER,pixel246 INTEGER,pixel247 INTEGER,pixel248 INTEGER,pixel249 INTEGER,pixel250 INTEGER,pixel251 INTEGER,pixel252 INTEGER,pixel253 INTEGER,pixel254 INTEGER,pixel255 INTEGER,pixel256 INTEGER,pixel257 INTEGER,pixel258 INTEGER,pixel259 INTEGER,pixel260 INTEGER,pixel261 INTEGER,pixel262 INTEGER,pixel263 INTEGER,pixel264 INTEGER,pixel265 INTEGER,pixel266 INTEGER,pixel267 INTEGER,pixel268 INTEGER,pixel269 INTEGER,pixel270 INTEGER,pixel271 INTEGER,pixel272 INTEGER,pixel273 INTEGER,pixel274 INTEGER,pixel275 INTEGER,pixel276 INTEGER,pixel277 INTEGER,pixel278 INTEGER,pixel279 INTEGER,pixel280 INTEGER,pixel281 INTEGER,pixel282 INTEGER,pixel283 INTEGER,pixel284 INTEGER,pixel285 INTEGER,pixel286 INTEGER,pixel287 INTEGER,pixel288 INTEGER,pixel289 INTEGER,pixel290 INTEGER,pixel291 INTEGER,pixel292 INTEGER,pixel293 INTEGER,pixel294 INTEGER,pixel295 INTEGER,pixel296 INTEGER,pixel297 INTEGER,pixel298 INTEGER,pixel299 INTEGER,pixel300 INTEGER,pixel301 INTEGER,pixel302 INTEGER,pixel303 INTEGER,pixel304 INTEGER,pixel305 INTEGER,pixel306 INTEGER,pixel307 INTEGER,pixel308 INTEGER,pixel309 INTEGER,pixel310 INTEGER,pixel311 INTEGER,pixel312 INTEGER,pixel313 INTEGER,pixel314 INTEGER,pixel315 INTEGER,pixel316 INTEGER,pixel317 INTEGER,pixel318 INTEGER,pixel319 INTEGER,pixel320 INTEGER,pixel321 INTEGER,pixel322 INTEGER,pixel323 INTEGER,pixel324 INTEGER,pixel325 INTEGER,pixel326 INTEGER,pixel327 INTEGER,pixel328 INTEGER,pixel329 INTEGER,pixel330 INTEGER,pixel331 INTEGER,pixel332 INTEGER,pixel333 INTEGER,pixel334 INTEGER,pixel335 INTEGER,pixel336 INTEGER,pixel337 INTEGER,pixel338 INTEGER,pixel339 INTEGER,pixel340 INTEGER,pixel341 INTEGER,pixel342 INTEGER,pixel343 INTEGER,pixel344 INTEGER,pixel345 INTEGER,pixel346 INTEGER,pixel347 INTEGER,pixel348 INTEGER,pixel349 INTEGER,pixel350 INTEGER,pixel351 INTEGER,pixel352 INTEGER,pixel353 INTEGER,pixel354 INTEGER,pixel355 INTEGER,pixel356 INTEGER,pixel357 INTEGER,pixel358 INTEGER,pixel359 INTEGER,pixel360 INTEGER,pixel361 INTEGER,pixel362 INTEGER,pixel363 INTEGER,pixel364 INTEGER,pixel365 INTEGER,pixel366 INTEGER,pixel367 INTEGER,pixel368 INTEGER,pixel369 INTEGER,pixel370 INTEGER,pixel371 INTEGER,pixel372 INTEGER,pixel373 INTEGER,pixel374 INTEGER,pixel375 INTEGER,pixel376 INTEGER,pixel377 INTEGER,pixel378 INTEGER,pixel379 INTEGER,pixel380 INTEGER,pixel381 INTEGER,pixel382 INTEGER,pixel383 INTEGER,pixel384 INTEGER,pixel385 INTEGER,pixel386 INTEGER,pixel387 INTEGER,pixel388 INTEGER,pixel389 INTEGER,pixel390 INTEGER,pixel391 INTEGER,pixel392 INTEGER,pixel393 INTEGER,pixel394 INTEGER,pixel395 INTEGER,pixel396 INTEGER,pixel397 INTEGER,pixel398 INTEGER,pixel399 INTEGER,pixel400 INTEGER,pixel401 INTEGER,pixel402 INTEGER,pixel403 INTEGER,pixel404 INTEGER,pixel405 INTEGER,pixel406 INTEGER,pixel407 INTEGER,pixel408 INTEGER,pixel409 INTEGER,pixel410 INTEGER,pixel411 INTEGER,pixel412 INTEGER,pixel413 INTEGER,pixel414 INTEGER,pixel415 INTEGER,pixel416 INTEGER,pixel417 INTEGER,pixel418 INTEGER,pixel419 INTEGER,pixel420 INTEGER,pixel421 INTEGER,pixel422 INTEGER,pixel423 INTEGER,pixel424 INTEGER,pixel425 INTEGER,pixel426 INTEGER,pixel427 INTEGER,pixel428 INTEGER,pixel429 INTEGER,pixel430 INTEGER,pixel431 INTEGER,pixel432 INTEGER,pixel433 INTEGER,pixel434 INTEGER,pixel435 INTEGER,pixel436 INTEGER,pixel437 INTEGER,pixel438 INTEGER,pixel439 INTEGER,pixel440 INTEGER,pixel441 INTEGER,pixel442 INTEGER,pixel443 INTEGER,pixel444 INTEGER,pixel445 INTEGER,pixel446 INTEGER,pixel447 INTEGER,pixel448 INTEGER,pixel449 INTEGER,pixel450 INTEGER,pixel451 INTEGER,pixel452 INTEGER,pixel453 INTEGER,pixel454 INTEGER,pixel455 INTEGER,pixel456 INTEGER,pixel457 INTEGER,pixel458 INTEGER,pixel459 INTEGER,pixel460 INTEGER,pixel461 INTEGER,pixel462 INTEGER,pixel463 INTEGER,pixel464 INTEGER,pixel465 INTEGER,pixel466 INTEGER,pixel467 INTEGER,pixel468 INTEGER,pixel469 INTEGER,pixel470 INTEGER,pixel471 INTEGER,pixel472 INTEGER,pixel473 INTEGER,pixel474 INTEGER,pixel475 INTEGER,pixel476 INTEGER,pixel477 INTEGER,pixel478 INTEGER,pixel479 INTEGER,pixel480 INTEGER,pixel481 INTEGER,pixel482 INTEGER,pixel483 INTEGER,pixel484 INTEGER,pixel485 INTEGER,pixel486 INTEGER,pixel487 INTEGER,pixel488 INTEGER,pixel489 INTEGER,pixel490 INTEGER,pixel491 INTEGER,pixel492 INTEGER,pixel493 INTEGER,pixel494 INTEGER,pixel495 INTEGER,pixel496 INTEGER,pixel497 INTEGER,pixel498 INTEGER,pixel499 INTEGER,pixel500 INTEGER,pixel501 INTEGER,pixel502 INTEGER,pixel503 INTEGER,pixel504 INTEGER,pixel505 INTEGER,pixel506 INTEGER,pixel507 INTEGER,pixel508 INTEGER,pixel509 INTEGER,pixel510 INTEGER,pixel511 INTEGER,pixel512 INTEGER,pixel513 INTEGER,pixel514 INTEGER,pixel515 INTEGER,pixel516 INTEGER,pixel517 INTEGER,pixel518 INTEGER,pixel519 INTEGER,pixel520 INTEGER,pixel521 INTEGER,pixel522 INTEGER,pixel523 INTEGER,pixel524 INTEGER,pixel525 INTEGER,pixel526 INTEGER,pixel527 INTEGER,pixel528 INTEGER,pixel529 INTEGER,pixel530 INTEGER,pixel531 INTEGER,pixel532 INTEGER,pixel533 INTEGER,pixel534 INTEGER,pixel535 INTEGER,pixel536 INTEGER,pixel537 INTEGER,pixel538 INTEGER,pixel539 INTEGER,pixel540 INTEGER,pixel541 INTEGER,pixel542 INTEGER,pixel543 INTEGER,pixel544 INTEGER,pixel545 INTEGER,pixel546 INTEGER,pixel547 INTEGER,pixel548 INTEGER,pixel549 INTEGER,pixel550 INTEGER,pixel551 INTEGER,pixel552 INTEGER,pixel553 INTEGER,pixel554 INTEGER,pixel555 INTEGER,pixel556 INTEGER,pixel557 INTEGER,pixel558 INTEGER,pixel559 INTEGER,pixel560 INTEGER,pixel561 INTEGER,pixel562 INTEGER,pixel563 INTEGER,pixel564 INTEGER,pixel565 INTEGER,pixel566 INTEGER,pixel567 INTEGER,pixel568 INTEGER,pixel569 INTEGER,pixel570 INTEGER,pixel571 INTEGER,pixel572 INTEGER,pixel573 INTEGER,pixel574 INTEGER,pixel575 INTEGER,pixel576 INTEGER,pixel577 INTEGER,pixel578 INTEGER,pixel579 INTEGER,pixel580 INTEGER,pixel581 INTEGER,pixel582 INTEGER,pixel583 INTEGER,pixel584 INTEGER,pixel585 INTEGER,pixel586 INTEGER,pixel587 INTEGER,pixel588 INTEGER,pixel589 INTEGER,pixel590 INTEGER,pixel591 INTEGER,pixel592 INTEGER,pixel593 INTEGER,pixel594 INTEGER,pixel595 INTEGER,pixel596 INTEGER,pixel597 INTEGER,pixel598 INTEGER,pixel599 INTEGER,pixel600 INTEGER,pixel601 INTEGER,pixel602 INTEGER,pixel603 INTEGER,pixel604 INTEGER,pixel605 INTEGER,pixel606 INTEGER,pixel607 INTEGER,pixel608 INTEGER,pixel609 INTEGER,pixel610 INTEGER,pixel611 INTEGER,pixel612 INTEGER,pixel613 INTEGER,pixel614 INTEGER,pixel615 INTEGER,pixel616 INTEGER,pixel617 INTEGER,pixel618 INTEGER,pixel619 INTEGER,pixel620 INTEGER,pixel621 INTEGER,pixel622 INTEGER,pixel623 INTEGER,pixel624 INTEGER,pixel625 INTEGER,pixel626 INTEGER,pixel627 INTEGER,pixel628 INTEGER,pixel629 INTEGER,pixel630 INTEGER,pixel631 INTEGER,pixel632 INTEGER,pixel633 INTEGER,pixel634 INTEGER,pixel635 INTEGER,pixel636 INTEGER,pixel637 INTEGER,pixel638 INTEGER,pixel639 INTEGER,pixel640 INTEGER,pixel641 INTEGER,pixel642 INTEGER,pixel643 INTEGER,pixel644 INTEGER,pixel645 INTEGER,pixel646 INTEGER,pixel647 INTEGER,pixel648 INTEGER,pixel649 INTEGER,pixel650 INTEGER,pixel651 INTEGER,pixel652 INTEGER,pixel653 INTEGER,pixel654 INTEGER,pixel655 INTEGER,pixel656 INTEGER,pixel657 INTEGER,pixel658 INTEGER,pixel659 INTEGER,pixel660 INTEGER,pixel661 INTEGER,pixel662 INTEGER,pixel663 INTEGER,pixel664 INTEGER,pixel665 INTEGER,pixel666 INTEGER,pixel667 INTEGER,pixel668 INTEGER,pixel669 INTEGER,pixel670 INTEGER,pixel671 INTEGER,pixel672 INTEGER,pixel673 INTEGER,pixel674 INTEGER,pixel675 INTEGER,pixel676 INTEGER,pixel677 INTEGER,pixel678 INTEGER,pixel679 INTEGER,pixel680 INTEGER,pixel681 INTEGER,pixel682 INTEGER,pixel683 INTEGER,pixel684 INTEGER,pixel685 INTEGER,pixel686 INTEGER,pixel687 INTEGER,pixel688 INTEGER,pixel689 INTEGER,pixel690 INTEGER,pixel691 INTEGER,pixel692 INTEGER,pixel693 INTEGER,pixel694 INTEGER,pixel695 INTEGER,pixel696 INTEGER,pixel697 INTEGER,pixel698 INTEGER,pixel699 INTEGER,pixel700 INTEGER,pixel701 INTEGER,pixel702 INTEGER,pixel703 INTEGER,pixel704 INTEGER,pixel705 INTEGER,pixel706 INTEGER,pixel707 INTEGER,pixel708 INTEGER,pixel709 INTEGER,pixel710 INTEGER,pixel711 INTEGER,pixel712 INTEGER,pixel713 INTEGER,pixel714 INTEGER,pixel715 INTEGER,pixel716 INTEGER,pixel717 INTEGER,pixel718 INTEGER,pixel719 INTEGER,pixel720 INTEGER,pixel721 INTEGER,pixel722 INTEGER,pixel723 INTEGER,pixel724 INTEGER,pixel725 INTEGER,pixel726 INTEGER,pixel727 INTEGER,pixel728 INTEGER,pixel729 INTEGER,pixel730 INTEGER,pixel731 INTEGER,pixel732 INTEGER,pixel733 INTEGER,pixel734 INTEGER,pixel735 INTEGER,pixel736 INTEGER,pixel737 INTEGER,pixel738 INTEGER,pixel739 INTEGER,pixel740 INTEGER,pixel741 INTEGER,pixel742 INTEGER,pixel743 INTEGER,pixel744 INTEGER,pixel745 INTEGER,pixel746 INTEGER,pixel747 INTEGER,pixel748 INTEGER,pixel749 INTEGER,pixel750 INTEGER,pixel751 INTEGER,pixel752 INTEGER,pixel753 INTEGER,pixel754 INTEGER,pixel755 INTEGER,pixel756 INTEGER,pixel757 INTEGER,pixel758 INTEGER,pixel759 INTEGER,pixel760 INTEGER,pixel761 INTEGER,pixel762 INTEGER,pixel763 INTEGER,pixel764 INTEGER,pixel765 INTEGER,pixel766 INTEGER,pixel767 INTEGER,pixel768 INTEGER,pixel769 INTEGER,pixel770 INTEGER,pixel771 INTEGER,pixel772 INTEGER,pixel773 INTEGER,pixel774 INTEGER,pixel775 INTEGER,pixel776 INTEGER,pixel777 INTEGER,pixel778 INTEGER,pixel779 INTEGER,pixel780 INTEGER,pixel781 INTEGER,pixel782 INTEGER,pixel783 INTEGER )""")
    c.execute("""CREATE TABLE IF NOT EXISTS Test(Id INTEGER PRIMARY KEY,label INTEGER,pixel0 INTEGER,pixel1 INTEGER,pixel2 INTEGER,pixel3 INTEGER,pixel4 INTEGER,pixel5 INTEGER,pixel6 INTEGER,pixel7 INTEGER,pixel8 INTEGER,pixel9 INTEGER,pixel10 INTEGER,pixel11 INTEGER,pixel12 INTEGER,pixel13 INTEGER,pixel14 INTEGER,pixel15 INTEGER,pixel16 INTEGER,pixel17 INTEGER,pixel18 INTEGER,pixel19 INTEGER,pixel20 INTEGER,pixel21 INTEGER,pixel22 INTEGER,pixel23 INTEGER,pixel24 INTEGER,pixel25 INTEGER,pixel26 INTEGER,pixel27 INTEGER,pixel28 INTEGER,pixel29 INTEGER,pixel30 INTEGER,pixel31 INTEGER,pixel32 INTEGER,pixel33 INTEGER,pixel34 INTEGER,pixel35 INTEGER,pixel36 INTEGER,pixel37 INTEGER,pixel38 INTEGER,pixel39 INTEGER,pixel40 INTEGER,pixel41 INTEGER,pixel42 INTEGER,pixel43 INTEGER,pixel44 INTEGER,pixel45 INTEGER,pixel46 INTEGER,pixel47 INTEGER,pixel48 INTEGER,pixel49 INTEGER,pixel50 INTEGER,pixel51 INTEGER,pixel52 INTEGER,pixel53 INTEGER,pixel54 INTEGER,pixel55 INTEGER,pixel56 INTEGER,pixel57 INTEGER,pixel58 INTEGER,pixel59 INTEGER,pixel60 INTEGER,pixel61 INTEGER,pixel62 INTEGER,pixel63 INTEGER,pixel64 INTEGER,pixel65 INTEGER,pixel66 INTEGER,pixel67 INTEGER,pixel68 INTEGER,pixel69 INTEGER,pixel70 INTEGER,pixel71 INTEGER,pixel72 INTEGER,pixel73 INTEGER,pixel74 INTEGER,pixel75 INTEGER,pixel76 INTEGER,pixel77 INTEGER,pixel78 INTEGER,pixel79 INTEGER,pixel80 INTEGER,pixel81 INTEGER,pixel82 INTEGER,pixel83 INTEGER,pixel84 INTEGER,pixel85 INTEGER,pixel86 INTEGER,pixel87 INTEGER,pixel88 INTEGER,pixel89 INTEGER,pixel90 INTEGER,pixel91 INTEGER,pixel92 INTEGER,pixel93 INTEGER,pixel94 INTEGER,pixel95 INTEGER,pixel96 INTEGER,pixel97 INTEGER,pixel98 INTEGER,pixel99 INTEGER,pixel100 INTEGER,pixel101 INTEGER,pixel102 INTEGER,pixel103 INTEGER,pixel104 INTEGER,pixel105 INTEGER,pixel106 INTEGER,pixel107 INTEGER,pixel108 INTEGER,pixel109 INTEGER,pixel110 INTEGER,pixel111 INTEGER,pixel112 INTEGER,pixel113 INTEGER,pixel114 INTEGER,pixel115 INTEGER,pixel116 INTEGER,pixel117 INTEGER,pixel118 INTEGER,pixel119 INTEGER,pixel120 INTEGER,pixel121 INTEGER,pixel122 INTEGER,pixel123 INTEGER,pixel124 INTEGER,pixel125 INTEGER,pixel126 INTEGER,pixel127 INTEGER,pixel128 INTEGER,pixel129 INTEGER,pixel130 INTEGER,pixel131 INTEGER,pixel132 INTEGER,pixel133 INTEGER,pixel134 INTEGER,pixel135 INTEGER,pixel136 INTEGER,pixel137 INTEGER,pixel138 INTEGER,pixel139 INTEGER,pixel140 INTEGER,pixel141 INTEGER,pixel142 INTEGER,pixel143 INTEGER,pixel144 INTEGER,pixel145 INTEGER,pixel146 INTEGER,pixel147 INTEGER,pixel148 INTEGER,pixel149 INTEGER,pixel150 INTEGER,pixel151 INTEGER,pixel152 INTEGER,pixel153 INTEGER,pixel154 INTEGER,pixel155 INTEGER,pixel156 INTEGER,pixel157 INTEGER,pixel158 INTEGER,pixel159 INTEGER,pixel160 INTEGER,pixel161 INTEGER,pixel162 INTEGER,pixel163 INTEGER,pixel164 INTEGER,pixel165 INTEGER,pixel166 INTEGER,pixel167 INTEGER,pixel168 INTEGER,pixel169 INTEGER,pixel170 INTEGER,pixel171 INTEGER,pixel172 INTEGER,pixel173 INTEGER,pixel174 INTEGER,pixel175 INTEGER,pixel176 INTEGER,pixel177 INTEGER,pixel178 INTEGER,pixel179 INTEGER,pixel180 INTEGER,pixel181 INTEGER,pixel182 INTEGER,pixel183 INTEGER,pixel184 INTEGER,pixel185 INTEGER,pixel186 INTEGER,pixel187 INTEGER,pixel188 INTEGER,pixel189 INTEGER,pixel190 INTEGER,pixel191 INTEGER,pixel192 INTEGER,pixel193 INTEGER,pixel194 INTEGER,pixel195 INTEGER,pixel196 INTEGER,pixel197 INTEGER,pixel198 INTEGER,pixel199 INTEGER,pixel200 INTEGER,pixel201 INTEGER,pixel202 INTEGER,pixel203 INTEGER,pixel204 INTEGER,pixel205 INTEGER,pixel206 INTEGER,pixel207 INTEGER,pixel208 INTEGER,pixel209 INTEGER,pixel210 INTEGER,pixel211 INTEGER,pixel212 INTEGER,pixel213 INTEGER,pixel214 INTEGER,pixel215 INTEGER,pixel216 INTEGER,pixel217 INTEGER,pixel218 INTEGER,pixel219 INTEGER,pixel220 INTEGER,pixel221 INTEGER,pixel222 INTEGER,pixel223 INTEGER,pixel224 INTEGER,pixel225 INTEGER,pixel226 INTEGER,pixel227 INTEGER,pixel228 INTEGER,pixel229 INTEGER,pixel230 INTEGER,pixel231 INTEGER,pixel232 INTEGER,pixel233 INTEGER,pixel234 INTEGER,pixel235 INTEGER,pixel236 INTEGER,pixel237 INTEGER,pixel238 INTEGER,pixel239 INTEGER,pixel240 INTEGER,pixel241 INTEGER,pixel242 INTEGER,pixel243 INTEGER,pixel244 INTEGER,pixel245 INTEGER,pixel246 INTEGER,pixel247 INTEGER,pixel248 INTEGER,pixel249 INTEGER,pixel250 INTEGER,pixel251 INTEGER,pixel252 INTEGER,pixel253 INTEGER,pixel254 INTEGER,pixel255 INTEGER,pixel256 INTEGER,pixel257 INTEGER,pixel258 INTEGER,pixel259 INTEGER,pixel260 INTEGER,pixel261 INTEGER,pixel262 INTEGER,pixel263 INTEGER,pixel264 INTEGER,pixel265 INTEGER,pixel266 INTEGER,pixel267 INTEGER,pixel268 INTEGER,pixel269 INTEGER,pixel270 INTEGER,pixel271 INTEGER,pixel272 INTEGER,pixel273 INTEGER,pixel274 INTEGER,pixel275 INTEGER,pixel276 INTEGER,pixel277 INTEGER,pixel278 INTEGER,pixel279 INTEGER,pixel280 INTEGER,pixel281 INTEGER,pixel282 INTEGER,pixel283 INTEGER,pixel284 INTEGER,pixel285 INTEGER,pixel286 INTEGER,pixel287 INTEGER,pixel288 INTEGER,pixel289 INTEGER,pixel290 INTEGER,pixel291 INTEGER,pixel292 INTEGER,pixel293 INTEGER,pixel294 INTEGER,pixel295 INTEGER,pixel296 INTEGER,pixel297 INTEGER,pixel298 INTEGER,pixel299 INTEGER,pixel300 INTEGER,pixel301 INTEGER,pixel302 INTEGER,pixel303 INTEGER,pixel304 INTEGER,pixel305 INTEGER,pixel306 INTEGER,pixel307 INTEGER,pixel308 INTEGER,pixel309 INTEGER,pixel310 INTEGER,pixel311 INTEGER,pixel312 INTEGER,pixel313 INTEGER,pixel314 INTEGER,pixel315 INTEGER,pixel316 INTEGER,pixel317 INTEGER,pixel318 INTEGER,pixel319 INTEGER,pixel320 INTEGER,pixel321 INTEGER,pixel322 INTEGER,pixel323 INTEGER,pixel324 INTEGER,pixel325 INTEGER,pixel326 INTEGER,pixel327 INTEGER,pixel328 INTEGER,pixel329 INTEGER,pixel330 INTEGER,pixel331 INTEGER,pixel332 INTEGER,pixel333 INTEGER,pixel334 INTEGER,pixel335 INTEGER,pixel336 INTEGER,pixel337 INTEGER,pixel338 INTEGER,pixel339 INTEGER,pixel340 INTEGER,pixel341 INTEGER,pixel342 INTEGER,pixel343 INTEGER,pixel344 INTEGER,pixel345 INTEGER,pixel346 INTEGER,pixel347 INTEGER,pixel348 INTEGER,pixel349 INTEGER,pixel350 INTEGER,pixel351 INTEGER,pixel352 INTEGER,pixel353 INTEGER,pixel354 INTEGER,pixel355 INTEGER,pixel356 INTEGER,pixel357 INTEGER,pixel358 INTEGER,pixel359 INTEGER,pixel360 INTEGER,pixel361 INTEGER,pixel362 INTEGER,pixel363 INTEGER,pixel364 INTEGER,pixel365 INTEGER,pixel366 INTEGER,pixel367 INTEGER,pixel368 INTEGER,pixel369 INTEGER,pixel370 INTEGER,pixel371 INTEGER,pixel372 INTEGER,pixel373 INTEGER,pixel374 INTEGER,pixel375 INTEGER,pixel376 INTEGER,pixel377 INTEGER,pixel378 INTEGER,pixel379 INTEGER,pixel380 INTEGER,pixel381 INTEGER,pixel382 INTEGER,pixel383 INTEGER,pixel384 INTEGER,pixel385 INTEGER,pixel386 INTEGER,pixel387 INTEGER,pixel388 INTEGER,pixel389 INTEGER,pixel390 INTEGER,pixel391 INTEGER,pixel392 INTEGER,pixel393 INTEGER,pixel394 INTEGER,pixel395 INTEGER,pixel396 INTEGER,pixel397 INTEGER,pixel398 INTEGER,pixel399 INTEGER,pixel400 INTEGER,pixel401 INTEGER,pixel402 INTEGER,pixel403 INTEGER,pixel404 INTEGER,pixel405 INTEGER,pixel406 INTEGER,pixel407 INTEGER,pixel408 INTEGER,pixel409 INTEGER,pixel410 INTEGER,pixel411 INTEGER,pixel412 INTEGER,pixel413 INTEGER,pixel414 INTEGER,pixel415 INTEGER,pixel416 INTEGER,pixel417 INTEGER,pixel418 INTEGER,pixel419 INTEGER,pixel420 INTEGER,pixel421 INTEGER,pixel422 INTEGER,pixel423 INTEGER,pixel424 INTEGER,pixel425 INTEGER,pixel426 INTEGER,pixel427 INTEGER,pixel428 INTEGER,pixel429 INTEGER,pixel430 INTEGER,pixel431 INTEGER,pixel432 INTEGER,pixel433 INTEGER,pixel434 INTEGER,pixel435 INTEGER,pixel436 INTEGER,pixel437 INTEGER,pixel438 INTEGER,pixel439 INTEGER,pixel440 INTEGER,pixel441 INTEGER,pixel442 INTEGER,pixel443 INTEGER,pixel444 INTEGER,pixel445 INTEGER,pixel446 INTEGER,pixel447 INTEGER,pixel448 INTEGER,pixel449 INTEGER,pixel450 INTEGER,pixel451 INTEGER,pixel452 INTEGER,pixel453 INTEGER,pixel454 INTEGER,pixel455 INTEGER,pixel456 INTEGER,pixel457 INTEGER,pixel458 INTEGER,pixel459 INTEGER,pixel460 INTEGER,pixel461 INTEGER,pixel462 INTEGER,pixel463 INTEGER,pixel464 INTEGER,pixel465 INTEGER,pixel466 INTEGER,pixel467 INTEGER,pixel468 INTEGER,pixel469 INTEGER,pixel470 INTEGER,pixel471 INTEGER,pixel472 INTEGER,pixel473 INTEGER,pixel474 INTEGER,pixel475 INTEGER,pixel476 INTEGER,pixel477 INTEGER,pixel478 INTEGER,pixel479 INTEGER,pixel480 INTEGER,pixel481 INTEGER,pixel482 INTEGER,pixel483 INTEGER,pixel484 INTEGER,pixel485 INTEGER,pixel486 INTEGER,pixel487 INTEGER,pixel488 INTEGER,pixel489 INTEGER,pixel490 INTEGER,pixel491 INTEGER,pixel492 INTEGER,pixel493 INTEGER,pixel494 INTEGER,pixel495 INTEGER,pixel496 INTEGER,pixel497 INTEGER,pixel498 INTEGER,pixel499 INTEGER,pixel500 INTEGER,pixel501 INTEGER,pixel502 INTEGER,pixel503 INTEGER,pixel504 INTEGER,pixel505 INTEGER,pixel506 INTEGER,pixel507 INTEGER,pixel508 INTEGER,pixel509 INTEGER,pixel510 INTEGER,pixel511 INTEGER,pixel512 INTEGER,pixel513 INTEGER,pixel514 INTEGER,pixel515 INTEGER,pixel516 INTEGER,pixel517 INTEGER,pixel518 INTEGER,pixel519 INTEGER,pixel520 INTEGER,pixel521 INTEGER,pixel522 INTEGER,pixel523 INTEGER,pixel524 INTEGER,pixel525 INTEGER,pixel526 INTEGER,pixel527 INTEGER,pixel528 INTEGER,pixel529 INTEGER,pixel530 INTEGER,pixel531 INTEGER,pixel532 INTEGER,pixel533 INTEGER,pixel534 INTEGER,pixel535 INTEGER,pixel536 INTEGER,pixel537 INTEGER,pixel538 INTEGER,pixel539 INTEGER,pixel540 INTEGER,pixel541 INTEGER,pixel542 INTEGER,pixel543 INTEGER,pixel544 INTEGER,pixel545 INTEGER,pixel546 INTEGER,pixel547 INTEGER,pixel548 INTEGER,pixel549 INTEGER,pixel550 INTEGER,pixel551 INTEGER,pixel552 INTEGER,pixel553 INTEGER,pixel554 INTEGER,pixel555 INTEGER,pixel556 INTEGER,pixel557 INTEGER,pixel558 INTEGER,pixel559 INTEGER,pixel560 INTEGER,pixel561 INTEGER,pixel562 INTEGER,pixel563 INTEGER,pixel564 INTEGER,pixel565 INTEGER,pixel566 INTEGER,pixel567 INTEGER,pixel568 INTEGER,pixel569 INTEGER,pixel570 INTEGER,pixel571 INTEGER,pixel572 INTEGER,pixel573 INTEGER,pixel574 INTEGER,pixel575 INTEGER,pixel576 INTEGER,pixel577 INTEGER,pixel578 INTEGER,pixel579 INTEGER,pixel580 INTEGER,pixel581 INTEGER,pixel582 INTEGER,pixel583 INTEGER,pixel584 INTEGER,pixel585 INTEGER,pixel586 INTEGER,pixel587 INTEGER,pixel588 INTEGER,pixel589 INTEGER,pixel590 INTEGER,pixel591 INTEGER,pixel592 INTEGER,pixel593 INTEGER,pixel594 INTEGER,pixel595 INTEGER,pixel596 INTEGER,pixel597 INTEGER,pixel598 INTEGER,pixel599 INTEGER,pixel600 INTEGER,pixel601 INTEGER,pixel602 INTEGER,pixel603 INTEGER,pixel604 INTEGER,pixel605 INTEGER,pixel606 INTEGER,pixel607 INTEGER,pixel608 INTEGER,pixel609 INTEGER,pixel610 INTEGER,pixel611 INTEGER,pixel612 INTEGER,pixel613 INTEGER,pixel614 INTEGER,pixel615 INTEGER,pixel616 INTEGER,pixel617 INTEGER,pixel618 INTEGER,pixel619 INTEGER,pixel620 INTEGER,pixel621 INTEGER,pixel622 INTEGER,pixel623 INTEGER,pixel624 INTEGER,pixel625 INTEGER,pixel626 INTEGER,pixel627 INTEGER,pixel628 INTEGER,pixel629 INTEGER,pixel630 INTEGER,pixel631 INTEGER,pixel632 INTEGER,pixel633 INTEGER,pixel634 INTEGER,pixel635 INTEGER,pixel636 INTEGER,pixel637 INTEGER,pixel638 INTEGER,pixel639 INTEGER,pixel640 INTEGER,pixel641 INTEGER,pixel642 INTEGER,pixel643 INTEGER,pixel644 INTEGER,pixel645 INTEGER,pixel646 INTEGER,pixel647 INTEGER,pixel648 INTEGER,pixel649 INTEGER,pixel650 INTEGER,pixel651 INTEGER,pixel652 INTEGER,pixel653 INTEGER,pixel654 INTEGER,pixel655 INTEGER,pixel656 INTEGER,pixel657 INTEGER,pixel658 INTEGER,pixel659 INTEGER,pixel660 INTEGER,pixel661 INTEGER,pixel662 INTEGER,pixel663 INTEGER,pixel664 INTEGER,pixel665 INTEGER,pixel666 INTEGER,pixel667 INTEGER,pixel668 INTEGER,pixel669 INTEGER,pixel670 INTEGER,pixel671 INTEGER,pixel672 INTEGER,pixel673 INTEGER,pixel674 INTEGER,pixel675 INTEGER,pixel676 INTEGER,pixel677 INTEGER,pixel678 INTEGER,pixel679 INTEGER,pixel680 INTEGER,pixel681 INTEGER,pixel682 INTEGER,pixel683 INTEGER,pixel684 INTEGER,pixel685 INTEGER,pixel686 INTEGER,pixel687 INTEGER,pixel688 INTEGER,pixel689 INTEGER,pixel690 INTEGER,pixel691 INTEGER,pixel692 INTEGER,pixel693 INTEGER,pixel694 INTEGER,pixel695 INTEGER,pixel696 INTEGER,pixel697 INTEGER,pixel698 INTEGER,pixel699 INTEGER,pixel700 INTEGER,pixel701 INTEGER,pixel702 INTEGER,pixel703 INTEGER,pixel704 INTEGER,pixel705 INTEGER,pixel706 INTEGER,pixel707 INTEGER,pixel708 INTEGER,pixel709 INTEGER,pixel710 INTEGER,pixel711 INTEGER,pixel712 INTEGER,pixel713 INTEGER,pixel714 INTEGER,pixel715 INTEGER,pixel716 INTEGER,pixel717 INTEGER,pixel718 INTEGER,pixel719 INTEGER,pixel720 INTEGER,pixel721 INTEGER,pixel722 INTEGER,pixel723 INTEGER,pixel724 INTEGER,pixel725 INTEGER,pixel726 INTEGER,pixel727 INTEGER,pixel728 INTEGER,pixel729 INTEGER,pixel730 INTEGER,pixel731 INTEGER,pixel732 INTEGER,pixel733 INTEGER,pixel734 INTEGER,pixel735 INTEGER,pixel736 INTEGER,pixel737 INTEGER,pixel738 INTEGER,pixel739 INTEGER,pixel740 INTEGER,pixel741 INTEGER,pixel742 INTEGER,pixel743 INTEGER,pixel744 INTEGER,pixel745 INTEGER,pixel746 INTEGER,pixel747 INTEGER,pixel748 INTEGER,pixel749 INTEGER,pixel750 INTEGER,pixel751 INTEGER,pixel752 INTEGER,pixel753 INTEGER,pixel754 INTEGER,pixel755 INTEGER,pixel756 INTEGER,pixel757 INTEGER,pixel758 INTEGER,pixel759 INTEGER,pixel760 INTEGER,pixel761 INTEGER,pixel762 INTEGER,pixel763 INTEGER,pixel764 INTEGER,pixel765 INTEGER,pixel766 INTEGER,pixel767 INTEGER,pixel768 INTEGER,pixel769 INTEGER,pixel770 INTEGER,pixel771 INTEGER,pixel772 INTEGER,pixel773 INTEGER,pixel774 INTEGER,pixel775 INTEGER,pixel776 INTEGER,pixel777 INTEGER,pixel778 INTEGER,pixel779 INTEGER,pixel780 INTEGER,pixel781 INTEGER,pixel782 INTEGER,pixel783 INTEGER )""")
    #c.execute("""CREATE TABLE IF NOT EXISTS Predictions(Id INTEGER PRIMARY KEY,Predicted INTEGER,Actual INTEGER)""")
    #c.execute("""CREATE VIEW  IF NOT EXISTS VW_MODEL_ACCURACY_PER_LABEL AS SELECT *,(CAST(CORRECT AS FLOAT) / TOTAL) * 100 AS ACCURACY FROM (SELECT predicted,COUNT(*) AS TOTAL,CASE WHEN PREDICTED = ACTUAL THEN 1 ELSE 0 END AS CORRECT FROM PREDICTIONS GROUP BY PREDICTED) AS X""")
    #c.execute("""CREATE VIEW  IF NOT EXISTS VW_MODEL_ACCURACY_OVERALL AS SELECT AVG(ACCURACY) AS MODELACC FROM(SELECT (CAST(CORRECT AS FLOAT) / TOTAL) * 100 AS ACCURACY FROM(SELECT predicted,COUNT(*) AS TOTAL,CASE WHEN PREDICTED = ACTUAL THEN 1 ELSE 0 END AS CORRECT FROM PREDICTIONS GROUP BY PREDICTED) AS X ) AS Y""")
    conn.commit()  # commit needed
    c.close()

@app.route('/index')
def index():
    create()
    #mimetypes.add_type('text/css', '.css')
    return render_template('index.html')

@app.route('/train')
def train():
    #mimetypes.add_type('text/css', '.css')
    return render_template('train.html')

@app.route('/test')
def test():
    #mimetypes.add_type('text/css', '.css')
    return render_template('test.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
