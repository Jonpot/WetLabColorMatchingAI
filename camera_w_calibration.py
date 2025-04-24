#!/usr/bin/env python3
# coding: utf-8
"""
camera_color24.py  –  24-patch ColorChecker calibration
改动：
  • avg_rgb：11×11 trimmed-mean 抽样
  • 二阶 Root-Polynomial CCM（10 维）
  • 24 色卡点：左半 raw / 右半 corr / 右侧 Macbeth 参考
"""

from __future__ import annotations
import cv2, time, json, os, argparse
import numpy as np
from pathlib import Path

WIN = "Calibration"

# -------- Macbeth 24 参考色 ------------------------------------------------
MACBETH_24_BGR = [
    ( 68,  82,115),(130,150,194),(157,122, 98),( 67,108, 87),
    (177,128,133),(170,189,103),( 44,126,214),(166, 91, 80),
    ( 99,  90,193),(108, 60, 94),( 64,188,157),( 46,163,224),
    (150, 61, 56),( 73,148, 70),( 60, 54,175),( 31,199,231),
    (149, 86,187),(161,133,  8),(242,243,242),(200,200,200),
    (160,160,160),(121,122,122),( 85, 85, 85),( 52, 52, 52)]
MACBETH_24_RGB = np.array([[b,g,r] for r,g,b in MACBETH_24_BGR], np.float32)

# ---------- γ helpers -----------------------------------------------------
def srgb2lin(s):
    s = s/255.0
    return np.where(s<=0.04045, s/12.92, ((s+0.055)/1.055)**2.4)
def lin2srgb(l):
    s = np.where(l<=0.0031308, 12.92*l, 1.055*l**(1/2.4)-0.055)
    return np.clip(s*255, 0, 255)

# ══════════════════════  PlateProcessor  ══════════════════════════════════
class PlateProcessor:
    def __init__(self):
        self.pts, self.cpts = [], []
        self.drag_idx = self.drag_cidx = -1
        self.img_copy = None; self.confirmed=False
        self.btnTL=None; self.BW,self.BH=140,30

    # ---------- camera ----------------------------------------------------
    @staticmethod
    def snapshot(cam=0, path="snapshot.jpg", warm=10, burst=5,
                 res=(1600,1200)):
        cap=cv2.VideoCapture(cam,cv2.CAP_DSHOW)
        if res: w,h=res; cap.set(3,w); cap.set(4,h); time.sleep(.2)
        for _ in range(warm): cap.read(); time.sleep(.04)
        acc=None
        for _ in range(burst):
            _,f=cap.read(); acc=f.astype(np.float32) if acc is None else acc+f
            time.sleep(.02)
        cap.release()
        img=(acc/burst).astype(np.uint8)
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(path,img,[cv2.IMWRITE_JPEG_QUALITY,95]); return path

    # ---------- helpers ---------------------------------------------------
    @staticmethod
    def plate_from_tb(v): return {0:"12",1:"24",2:"48",3:"96"}.get(v,"96")
    @staticmethod
    def well_centers(x1,y1,x2,y2, p="96"):
        r,c={"12":(8,12),"24":(4,6),"48":(6,8),"96":(8,12)}[p]
        dx,dy=(x2-x1)/c,(y2-y1)/r
        g=np.empty((r,c,2),float)
        for i in range(r):
            for j in range(c):
                g[i,j]=(x1+(j+.5)*dx, y1+(i+.5)*dy)
        return g

    # -------- avg_rgb : 11×11 trimmed mean --------------------------------
    @staticmethod
    def avg_rgb(img, centers, win=11, trim=0.1):
        r=[]; h,w=img.shape[:2]; half=win//2
        for row in centers:
            rr=[]
            for cx,cy in row:
                x,y=int(cx),int(cy)
                x1,x2=max(0,x-half),min(w,x+half+1)
                y1,y2=max(0,y-half),min(h,y+half+1)
                patch=img[y1:y2,x1:x2].reshape(-1,3).astype(np.float32)
                if trim>0:
                    k=int(len(patch)*trim)
                    patch=np.sort(patch,0)[k:-k] if len(patch)>2*k else patch
                rr.append(patch.mean(0)[::-1])   # RGB
            r.append(rr)
        return r

    # -------- Root-Polynomial CCM (10 cols) -------------------------------
    @staticmethod
    def _poly_terms(lin):          # lin : N×3 in linear space
        r,g,b = lin.T
        return np.stack([r,g,b,
                         np.sqrt(r*g),np.sqrt(r*b),np.sqrt(g*b),
                         r*r,g*g,b*b,
                         np.ones_like(r)],1)      # N×10

    @staticmethod
    def fit_rpcc(meas_rgb8, ref_rgb8):
        m_lin = srgb2lin(meas_rgb8)
        r_lin = srgb2lin(ref_rgb8)
        X = PlateProcessor._poly_terms(m_lin)     # N×10
        return r_lin.T @ np.linalg.pinv(X.T)      # 3×10

    @staticmethod
    def apply_rpcc(rgb8, M10):
        lin = srgb2lin(rgb8.astype(np.float32).reshape(-1,3))
        corr = M10 @ PlateProcessor._poly_terms(lin).T
        return lin2srgb(corr.T).reshape(rgb8.shape).clip(0,255).astype(np.uint8)

    # -------------------- (UI 代码保持不变，略) -----------------------------

    # ----------------------- processing pipeline --------------------------
    def process_image(self, cam_index=0, snap="snapshot.jpg",
                      calib="calibration.json", force_ui=False):
        self.snapshot(cam_index,snap)

        cfg=None
        if os.path.exists(calib):
            cfg=json.load(open(calib))
        if force_ui or cfg is None:
            cfg=self.run_ui(cv2.imread(snap),cfg)
            json.dump(cfg,open(calib,"w"),indent=2)

        img=cv2.imread(snap)
        r=cfg["rectangle"]
        centres=self.well_centers(r["x1"],r["y1"],r["x2"],r["y2"],cfg["plate_type"])
        raw = self.avg_rgb(img, centres)

        dots=np.asarray(cfg["calibration_dots"],int)
        meas_raw = img[dots[:,1],dots[:,0],::-1].astype(np.float32) # N×3 RGB
        M10 = self.fit_rpcc(meas_raw, MACBETH_24_RGB)
        meas_corr = self.apply_rpcc(meas_raw, M10)
        corr = self.apply_rpcc(np.array(raw,np.float32), M10)

        # -------- draw results --------
        alpha=cfg.get("contrast",10)/10.0; beta=cfg.get("brightness",100)-100
        raw_adj=np.clip(np.array(raw)*alpha+beta,0,255).astype(np.uint8)

        marked=img.copy(); R=10
        for ctr_row,r_row,c_row in zip(centres,raw_adj,corr):
            for (cx,cy),rgb_r,rgb_c in zip(ctr_row,r_row,c_row):
                cv2.circle(marked,(int(cx),int(cy)),R,tuple(int(v) for v in rgb_c[::-1]),-1)
                cv2.ellipse(marked,(int(cx),int(cy)),(R,R),0,90,270,
                            tuple(int(v) for v in rgb_r[::-1]),-1)

        SHIFT=R+10
        for i,((x,y),rgb_r,rgb_c) in enumerate(zip(dots,meas_raw,meas_corr)):
            centre=(int(x),int(y))
            bgr_r = tuple(int(v) for v in rgb_r[::-1])
            bgr_c = tuple(int(v) for v in rgb_c[::-1])
            bgr_ref = MACBETH_24_BGR[i]

            cv2.circle(marked, centre, R, bgr_c, -1)
            cv2.ellipse(marked, centre, (R,R), 0, 90, 270, bgr_r, -1)
            cv2.circle(marked, (centre[0]+SHIFT, centre[1]), R, bgr_ref, -1)

        cv2.imwrite("output_with_centers_corr.jpg",marked)
        print("[Saved] output_with_centers_corr.jpg")
        return corr

# ---------------- CLI ----------------
if __name__ == "__main__":
    # ap=argparse.ArgumentParser()
    # ap.add_argument("--cam",type=int,default=1)
    # ap.add_argument("--recalib",action="store_true")
    # args=ap.parse_args()

    corr = PlateProcessor().process_image(cam_index=1)
    # print("First corrected RGB:", corr[0][0])
