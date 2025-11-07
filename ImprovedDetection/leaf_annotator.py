# leaf_annotator.py
from __future__ import annotations
from pathlib import Path
import json, tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import random

HELP = """Left-click: add point
Enter/Return: close polygon
Backspace: undo last point
Z: undo last polygon
C: cancel current polygon
S: save annotations
N / P: next / previous image
Q or Esc: quit"""

class Annotator(tk.Tk):
    def __init__(self, images_dir: str, out_dir: str):
        super().__init__()
        self.title("Leaf Polygon Labeler")
        self.geometry("1100x800")

        self.images = sorted([p for p in Path(images_dir).glob("*.*")
                              if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}])
        if not self.images:
            raise SystemExit(f"No images in {images_dir}")

        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0

        self.canvas = tk.Canvas(self, bg="#202225", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._redraw)
        self.canvas.bind("<Button-1>", self._on_click)
        self.bind("<Return>", self._close_poly)
        self.bind("<KP_Enter>", self._close_poly)
        self.bind("<BackSpace>", self._undo_point)
        self.bind("<z>", self._undo_poly); self.bind("<Z>", self._undo_poly)
        self.bind("<c>", self._cancel_poly); self.bind("<C>", self._cancel_poly)
        self.bind("<s>", self._save); self.bind("<S>", self._save)
        self.bind("<n>", self._next); self.bind("<N>", self._next)
        self.bind("<p>", self._prev); self.bind("<P>", self._prev)
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<q>", lambda e: self.destroy()); self.bind("<Q>", lambda e: self.destroy())

        self.scale = 1.0; self.offset = (0,0)
        self.im = None; self.tk_im = None; self.w0 = self.h0 = 0

        # per image state
        self.polys = []          # list of list[(x,y)] in image coords
        self.poly_colors = []    # parallel colors
        self.curr = []           # current polygon points (image coords)
        self.drawn = []          # canvas item ids
        self._load_image()

    # ---------- I/O ----------
    def _img_path(self): return self.images[self.idx]
    def _ann_path(self): return self.out_dir / f"{self._img_path().stem}.json"

    def _load_image(self):
        path = self._img_path()
        self.im = Image.open(path).convert("RGB")
        self.w0, self.h0 = self.im.size
        # load existing polygons if present
        self.polys, self.poly_colors = [], []
        if self._ann_path().exists():
            with open(self._ann_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("image_size") == [self.w0, self.h0]:
                self.polys = [ [(float(x), float(y)) for (x,y) in poly] for poly in data.get("polygons", []) ]
                rnd = random.Random(self._img_path().stem)
                self.poly_colors = [f"#{rnd.randrange(0,0xFFFFFF):06x}" for _ in self.polys]
        self.curr = []
        self._redraw()

    def _save(self, *_):
        data = {
            "image": str(self._img_path().name),
            "image_size": [self.w0, self.h0],
            "polygons": [[ [float(x), float(y)] for (x,y) in poly ] for poly in self.polys]
        }
        with open(self._ann_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.title(f"Saved → {self._ann_path().name}")

    # ---------- Navigation ----------
    def _next(self, *_):
        self._save()
        if self.idx < len(self.images)-1:
            self.idx += 1; self._load_image()

    def _prev(self, *_):
        self._save()
        if self.idx > 0:
            self.idx -= 1; self._load_image()

    # ---------- Drawing helpers ----------
    def _canvas_to_img(self, x, y):
        ox, oy = self.offset
        return ((x - ox)/self.scale, (y - oy)/self.scale)

    def _img_to_canvas(self, x, y):
        ox, oy = self.offset
        return (ox + x*self.scale, oy + y*self.scale)

    def _redraw(self, *_):
        self.canvas.delete("all"); self.drawn.clear()
        W = self.canvas.winfo_width() or 1
        H = self.canvas.winfo_height() or 1
        # scale to fit
        sx, sy = W / max(1,self.w0), H / max(1,self.h0)
        self.scale = min(sx, sy)
        new_w, new_h = int(self.w0*self.scale), int(self.h0*self.scale)
        self.offset = ((W-new_w)//2, (H-new_h)//2)
        disp = self.im.resize((new_w, new_h), Image.BILINEAR)
        self.tk_im = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.offset[0], self.offset[1], image=self.tk_im, anchor="nw")

        # draw existing polys
        for poly, col in zip(self.polys, self.poly_colors):
            pts = sum([self._img_to_canvas(x,y) for (x,y) in poly], ())
            pid = self.canvas.create_polygon(*pts, outline=col, width=2, fill="", smooth=False)
            self.drawn.append(pid)

        # draw current
        if self.curr:
            pts = sum([self._img_to_canvas(x,y) for (x,y) in self.curr], ())
            self.canvas.create_line(*pts, fill="#00ffff", width=2)
            for (x,y) in self.curr:
                cx, cy = self._img_to_canvas(x,y)
                self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, outline="#00ffff")

        # header/footer
        self.canvas.create_rectangle(0,0,W,28, fill="#000000", stipple="gray25", outline="")
        self.canvas.create_text(8,14, anchor="w", fill="white",
            text=f"{self.idx+1}/{len(self.images)}  {self._img_path().name}", font=("Segoe UI", 10, "bold"))
        self.canvas.create_text(W-8,14, anchor="e", fill="#cbd5e1",
            text="S=save  Enter=close polygon  Z=undo poly  Backspace=undo point  N/P=next/prev  Q=quit",
            font=("Segoe UI", 10))
        self.canvas.create_text(8,H-10, anchor="w", fill="#a3e635", text=HELP, font=("Segoe UI", 9))

    # ---------- Interactions ----------
    def _on_click(self, e):
        x, y = self._canvas_to_img(e.x, e.y)
        x = min(max(x, 0), self.w0-1); y = min(max(y, 0), self.h0-1)
        if not self.curr:  # start polygon
            self.curr = []
        self.curr.append((x, y))
        self._redraw()

    def _close_poly(self, *_):
        if len(self.curr) >= 3:
            self.polys.append(self.curr[:])
            rnd = random.Random(len(self.polys) * 1337)
            self.poly_colors.append(f"#{rnd.randrange(0,0xFFFFFF):06x}")
        self.curr = []
        self._redraw()

    def _undo_point(self, *_):
        if self.curr:
            self.curr.pop()
        self._redraw()

    def _undo_poly(self, *_):
        if self.polys:
            self.polys.pop(); self.poly_colors.pop()
        self._redraw()

    def _cancel_poly(self, *_):
        self.curr = []; self._redraw()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=False, default="images", help="folder with images")
    ap.add_argument("--out", required=False, default="anns", help="where to save per-image JSONs")
    args = ap.parse_args()
    app = Annotator(args.images, args.out)
    app.mainloop()
