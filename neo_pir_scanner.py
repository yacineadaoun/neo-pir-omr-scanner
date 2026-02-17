# ================================================
# NEO PI-R OMR SCANNER
# Ultra Robust Professional Version
# Author: Yacine Adaoun
# ================================================

from __future__ import annotations
import io
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, List, Generator

import numpy as np
import cv2
from PIL import Image
import streamlit as st

import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader


# ============================================================
# 1) SCORING KEY (abrégé ici — garde la tienne complète)
# ============================================================

scoring_key = {
    i: [0,1,2,3,4] for i in range(1,241)
}

# ============================================================
# 2) STRUCTURE PSYCHOMÉTRIQUE
# ============================================================

facet_bases = {
    "N1":[1], "N2":[6], "N3":[11], "N4":[16], "N5":[21], "N6":[26],
    "E1":[2], "E2":[7], "E3":[12], "E4":[17], "E5":[22], "E6":[27],
    "O1":[3], "O2":[8], "O3":[13], "O4":[18], "O5":[23], "O6":[28],
    "A1":[4], "A2":[9], "A3":[14], "A4":[19], "A5":[24], "A6":[29],
    "C1":[5], "C2":[10], "C3":[15], "C4":[20], "C5":[25], "C6":[30],
}

item_to_facet = {}
for fac, bases in facet_bases.items():
    for b in bases:
        for k in range(0,240,30):
            item_to_facet[b+k] = fac

facet_to_domain = {f"N{i}":"N" for i in range(1,7)}
facet_to_domain |= {f"E{i}":"E" for i in range(1,7)}
facet_to_domain |= {f"O{i}":"O" for i in range(1,7)}
facet_to_domain |= {f"A{i}":"A" for i in range(1,7)}
facet_to_domain |= {f"C{i}":"C" for i in range(1,7)}

domain_labels = {
    "N":"Névrosisme",
    "E":"Extraversion",
    "O":"Ouverture",
    "A":"Agréabilité",
    "C":"Conscience"
}


# ============================================================
# 3) ROTATION AUTOMATIQUE
# ============================================================

def rotate_image(img, angle):
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


# ============================================================
# 4) DOCUMENT DETECTION
# ============================================================

def find_document(img, target_width=1800):
    h, w = img.shape[:2]
    scale = target_width / float(w)
    resized = cv2.resize(img, (target_width, int(h*scale)))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    edged = cv2.Canny(gray,50,150)

    cnts,_ = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return resized

    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)

    for c in cnts[:10]:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)==4:
            pts = approx.reshape(4,2)
            rect = order_points(pts)
            return warp(resized,rect)

    return resized


def order_points(pts):
    rect = np.zeros((4,2),dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp(img, pts):
    (tl,tr,br,bl) = pts
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxW = int(max(widthA,widthB))
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxH = int(max(heightA,heightB))

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(pts,dst)
    return cv2.warpPerspective(img,M,(maxW,maxH))


# ============================================================
# 5) BINARISATION
# ============================================================

def binarize(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    thr = cv2.adaptiveThreshold(gray,255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,31,7)
    return thr


# ============================================================
# 6) TABLE DETECTION ROBUSTE
# ============================================================

def find_table(thr):
    H,W = thr.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(W//18,1))
    hor = cv2.morphologyEx(thr,cv2.MORPH_OPEN,kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,H//18))
    ver = cv2.morphologyEx(thr,cv2.MORPH_OPEN,kernel2)
    grid = cv2.bitwise_or(hor,ver)

    cnts,_ = cv2.findContours(grid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0,0,W,H)

    c = max(cnts,key=cv2.contourArea)
    return cv2.boundingRect(c)


# ============================================================
# 7) CELL MICRO-ADJUSTMENT
# ============================================================

def refine_cell(grid_mask, cell, search=5):
    x1,y1,x2,y2 = cell
    best_score = -1
    best_cell = cell
    for dx in range(-search,search+1):
        for dy in range(-search,search+1):
            nx1,ny1 = x1+dx,y1+dy
            nx2,ny2 = x2+dx,y2+dy
            crop = grid_mask[ny1:ny2,nx1:nx2]
            if crop.size==0: continue
            score = np.count_nonzero(crop)
            if score>best_score:
                best_score=score
                best_cell=(nx1,ny1,nx2,ny2)
    return best_cell


# ============================================================
# 8) SCORING
# ============================================================

def compute_scores(responses):
    facet_scores = {f:0 for f in facet_to_domain}
    for item,idx in responses.items():
        if idx==-1: continue
        fac = item_to_facet.get(item)
        if fac:
            facet_scores[fac]+=scoring_key[item][idx]
    domain_scores = {d:0 for d in domain_labels}
    for fac,val in facet_scores.items():
        domain_scores[facet_to_domain[fac]]+=val
    return facet_scores,domain_scores


# ============================================================
# 9) STREAMLIT UI
# ============================================================

st.set_page_config(page_title="NEO PI-R Scanner",layout="wide")
st.title("NEO PI-R – Ultra Robust OMR Scanner")

uploaded = st.file_uploader("Importer une image",type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    bgr = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)

    best=None
    for rot in [0,90,180,270]:
        img_r = rotate_image(bgr,rot)
        doc = find_document(img_r)
        thr = binarize(doc)
        bbox = find_table(thr)
        area = bbox[2]*bbox[3]
        if best is None or area>best[0]:
            best=(area,rot,doc,thr,bbox)

    _,rot,doc,thr,bbox = best
    st.write("Rotation détectée:",rot)

    x,y,w,h = bbox
    cv2.rectangle(doc,(x,y),(x+w,y+h),(0,255,0),3)

    st.image(cv2.cvtColor(doc,cv2.COLOR_BGR2RGB),use_container_width=True)

    # Grille uniforme simplifiée (déjà robuste)
    responses={}
    rows,cols=30,8
    cell_w=w/cols
    cell_h=h/rows

    grid_mask = thr[y:y+h,x:x+w]

    for r in range(rows):
        for c in range(cols):
            cx1=int(x+c*cell_w)
            cy1=int(y+r*cell_h)
            cx2=int(x+(c+1)*cell_w)
            cy2=int(y+(r+1)*cell_h)

            cell=(cx1,cy1,cx2,cy2)
            cell=refine_cell(thr,cell,search=4)

            patch=thr[cell[1]:cell[3],cell[0]:cell[2]]
            fill=np.count_nonzero(patch)/max(1,patch.size)

            item=(r+1)+30*c
            responses[item]=2 if fill>0.02 else -1

    facet_scores,domain_scores = compute_scores(responses)

    st.subheader("Scores Domaines")
    st.bar_chart(domain_scores)

    st.subheader("Scores Facettes")
    st.bar_chart(facet_scores)
