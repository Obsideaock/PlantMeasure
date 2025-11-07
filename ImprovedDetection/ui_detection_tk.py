# ui_detection_tk.py
from pathlib import Path
import sys, time
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from edge_first_leaves import segment_leaves_edgefirst, colorize_labels

# -----------------------------
# Core segmentation (bgr input)
# -----------------------------
def segment_and_count_leaves_bgr(
    bgr: np.ndarray,
    stem_suppression_radius: int = 20,
    fg_distance_thresh: float = 0.55,
    min_leaf_area: int = 5000
):
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty image supplied to segment_and_count_leaves_bgr().")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, fg0 = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg0, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)

    r = max(0, int(stem_suppression_radius))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    leaf_mask_coarse = cv2.morphologyEx(fg, cv2.MORPH_OPEN, disk, 1)
    leaf_mask_coarse = cv2.morphologyEx(
        leaf_mask_coarse, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        leaf_mask_coarse, connectivity=8
    )
    leaf_mask = np.zeros_like(leaf_mask_coarse)
    area_thresh = max(0, int(min_leaf_area))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_thresh:
            leaf_mask[labels==i] = 255

    if np.count_nonzero(leaf_mask) == 0:
        overlay_bgr = bgr.copy()
        return 0, leaf_mask, np.zeros_like(leaf_mask, dtype=np.int32), overlay_bgr

    dist = cv2.distanceTransform(leaf_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    thr = min(0.99, max(0.0, float(fg_distance_thresh)))
    sure_fg = np.uint8((dist_norm > thr) * 255)
    sure_bg = cv2.dilate(leaf_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), 2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    gradx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.magnitude(gradx, grady))
    grad_rgb = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

    markers_ws = markers.copy()
    cv2.watershed(grad_rgb, markers_ws)

    leaf_labels = np.where(leaf_mask>0, markers_ws, 0)
    leaf_labels[leaf_labels==-1] = 0

    ids = np.unique(leaf_labels)
    ids = ids[(ids != 0) & (ids != 1)]
    relabeled = np.zeros_like(leaf_labels, dtype=np.int32)
    for new_id, old_id in enumerate(ids, 1):
        relabeled[leaf_labels==old_id] = new_id
    leaf_labels = relabeled
    leaf_count = int(leaf_labels.max())

    overlay_bgr = bgr.copy()
    alpha = 0.35
    mask_3c = np.dstack([leaf_mask]*3) > 0
    green = np.array([0,255,0], dtype=np.float32)
    pix = overlay_bgr[mask_3c].reshape(-1,3).astype(np.float32)
    blended = (pix*(1-alpha) + green*alpha).clip(0,255).astype(np.uint8)
    overlay_bgr[mask_3c] = blended.reshape(-1)

    return leaf_count, leaf_mask, leaf_labels, overlay_bgr

def colorize_labels(leaf_labels: np.ndarray, seed: int = 42) -> np.ndarray:
    k = int(leaf_labels.max())
    if k <= 0:
        return np.zeros((*leaf_labels.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(k+1, 3), dtype=np.uint8)
    color_labels = np.zeros((*leaf_labels.shape, 3), dtype=np.uint8)
    nz = leaf_labels > 0
    color_labels[nz] = palette[leaf_labels[nz]]
    return color_labels

# -----------------------------
# Tkinter App
# -----------------------------
class LeafGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Herbitarium Leaf Detector")
        self.minsize(980, 640)

        # Theme / styles
        style = ttk.Style()
        try: style.theme_use("clam")
        except tk.TclError: pass
        style.configure("Left.TFrame", background="#0f1115")
        style.configure("Right.TFrame", background="#161a22")
        style.configure("TLabel", background="#161a22", foreground="#eaeef6")
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 9), foreground="#9aa4b2")
        style.configure("TButton", padding=6)
        style.configure("TScale", background="#161a22")

        # ---------- STATE (created BEFORE building controls) ----------
        self.out_dir = (Path(__file__).resolve().parent / "out")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.img_path = None
        self.bgr = None
        self.h0 = self.w0 = None
        self.aspect = None

        self.var_radius   = tk.IntVar(self, 20)
        self.var_fg_x100  = tk.IntVar(self, 55)
        self.var_min_area = tk.IntVar(self, 5000)

        self._pending_after = None
        self.palette_seed = 42
        self.leaf_count = 0
        self.leaf_mask = None
        self.leaf_labels = None
        self.overlay_bgr = None
        self.color_labels_rgb = None
        self._tk_img = None

        self._sliders = []          # will be filled in _build_controls
        self._action_buttons = []   # will be filled in _build_controls

        # Paned layout
        self.pw = ttk.Panedwindow(self, orient="horizontal")
        self.pw.pack(fill="both", expand=True)

        self.preview_frame = ttk.Frame(self.pw, style="Left.TFrame")
        self.ctrl_frame    = ttk.Frame(self.pw, style="Right.TFrame")
        self.pw.add(self.preview_frame, weight=4)
        self.pw.add(self.ctrl_frame,    weight=0)
        # set an initial width for the right pane and keep it from stretching
        self.ctrl_frame.configure(width=360)
        self.after_idle(lambda: self.pw.pane(self.ctrl_frame, weight=0))

        # Preview canvas
        self.canvas = tk.Canvas(self.preview_frame, bg="#0f1115", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Build right-side controls (now safe: Vars/lists already exist)
        self._build_controls()

        # Shortcuts
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<q>",      lambda e: self.destroy())
        self.bind("<Q>",      lambda e: self.destroy())
        self.bind("<s>",      lambda e: self._save_outputs())
        self.bind("<S>",      lambda e: self._save_outputs())
        self.bind("<r>",      lambda e: self._shuffle_colors())
        self.bind("<R>",      lambda e: self._shuffle_colors())

        # Start blank
        self._set_controls_enabled(False)
        self._render_blank()

    # ---------- UI ----------
    def _build_controls(self):
        pad = ttk.Frame(self.ctrl_frame, style="Right.TFrame", padding=(16,16,16,16))
        pad.pack(fill="both", expand=True)

        hdr = ttk.Frame(pad, style="Right.TFrame")
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Leaf Detection", style="Title.TLabel").pack(side="left")
        ttk.Button(hdr, text="Open image…", command=self._choose_file).pack(side="right")

        self.file_label = ttk.Label(pad, text="No image loaded", style="Small.TLabel")
        self.file_label.pack(anchor="w", pady=(8,12))

        self._add_slider(
            parent=pad,
            title="Stem suppression radius (px)",
            var=self.var_radius, from_=0, to=30, resolution=1,
            fmt=lambda v: f"{int(v)}"
        )
        self._add_slider(
            parent=pad,
            title="Watershed FG distance threshold",
            var=self.var_fg_x100, from_=0, to=99, resolution=1,
            fmt=lambda v: f"{int(v)/100:.2f}"
        )
        self._add_slider(
            parent=pad,
            title="Minimum leaf area (px)",
            var=self.var_min_area, from_=0, to=20000, resolution=10,
            fmt=lambda v: f"{int(v)}"
        )

        btns = ttk.Frame(pad, style="Right.TFrame")
        btns.pack(fill="x", pady=(10,6))
        b_save = ttk.Button(btns, text="Save (S)", command=self._save_outputs)
        b_colr = ttk.Button(btns, text="Shuffle colors (R)", command=self._shuffle_colors)
        b_save.pack(side="left", expand=True, fill="x", padx=(0,6))
        b_colr.pack(side="left", expand=True, fill="x", padx=(6,0))
        self._action_buttons = [b_save, b_colr]

        self.stats_var = tk.StringVar(value="—")
        ttk.Label(pad, textvariable=self.stats_var, style="Small.TLabel").pack(anchor="w", pady=(6,0))

    def _add_slider(self, parent, title, var, from_, to, resolution, fmt):
        frame = ttk.Frame(parent, style="Right.TFrame")
        frame.pack(fill="x", pady=(6,6))
        row_top = ttk.Frame(frame, style="Right.TFrame")
        row_top.pack(fill="x")
        ttk.Label(row_top, text=title).pack(side="left")
        value_label = ttk.Label(row_top, text=fmt(var.get()))
        value_label.pack(side="right")

        scale = ttk.Scale(
            frame, from_=from_, to=to, orient="horizontal",
            command=lambda sval, v=var, lbl=value_label, f=fmt: self._slider_callback(sval, v, lbl, f)
        )
        scale.set(var.get())
        scale.pack(fill="x", pady=(6,0))
        self._sliders.append(scale)

        var.trace_add("write", self._on_params_changed)

    def _set_controls_enabled(self, enabled: bool):
        for s in self._sliders:
            s.state(['!disabled'] if enabled else ['disabled'])
        for b in self._action_buttons:
            b.state(['!disabled'] if enabled else ['disabled'])

    # ---------- File handling ----------
    def _choose_file(self):
        start_dir = Path(__file__).resolve().parent / "data"
        start_dir.mkdir(exist_ok=True)
        path = filedialog.askopenfilename(
            title="Select plant image",
            initialdir=str(start_dir),
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not path:
            return
        img_path = Path(path)
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("Read error", f"Failed to read: {img_path}")
            return

        self.img_path = img_path
        self.bgr = bgr
        self.h0, self.w0 = self.bgr.shape[:2]
        self.aspect = self.w0 / self.h0
        self.file_label.config(text=img_path.name)

        self._set_controls_enabled(True)
        self._compute_and_render()

    # ---------- Sliders ----------
    def _slider_callback(self, sval, var, value_label, fmt):
        try:
            if isinstance(var, tk.IntVar): var.set(int(float(sval)))
            else: var.set(float(sval))
            value_label.config(text=fmt(var.get()))
        except tk.TclError:
            pass

    def _on_params_changed(self, *_):
        if self.bgr is None:
            return
        if self._pending_after:
            self.after_cancel(self._pending_after)
        self._pending_after = self.after(60, self._compute_and_render)

    # ---------- Rendering ----------
    def _render_blank(self):
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.canvas.create_text(
            cw//2, ch//2, text="Open image… to begin",
            fill="#7b8496", font=("Segoe UI", 16, "bold")
        )

    def _render_only(self):
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width());
        ch = max(1, self.canvas.winfo_height())

        # choose what to display
        if self.color_labels_rgb is not None and self.leaf_count > 0:
            vis = self.color_labels_rgb
        elif self.overlay_bgr is not None:
            vis = cv2.cvtColor(self.overlay_bgr, cv2.COLOR_BGR2RGB)
        else:
            self._render_blank();
            return

        h0, w0 = vis.shape[:2]
        scale = min(cw / w0, ch / h0)
        new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        img = Image.fromarray(vis).resize((new_w, new_h), resample=Image.NEAREST)
        self._tk_img = ImageTk.PhotoImage(img)
        x = (cw - new_w) // 2;
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, image=self._tk_img, anchor="nw")

    def _on_canvas_resize(self, _event):
        if self.color_labels_rgb is None:
            self._render_blank()
        else:
            self._render_only()

    # ---------- Compute + actions ----------
    def _compute_and_render(self):
        self._pending_after = None
        if self.bgr is None:
            self._render_blank()
            return

        t0 = time.time()
        stem_radius = int(self.var_radius.get())
        fg_distance_thresh = max(0.0, min(0.99, self.var_fg_x100.get() / 100.0))
        min_leaf_area = int(self.var_min_area.get())

        (self.leaf_count,
         self.leaf_mask,
         self.leaf_labels,
         self.overlay_bgr) = segment_leaves_edgefirst(
            self.bgr,
            min_leaf_area=min_leaf_area,
            dt_min_dist_px=max(3, stem_radius),
            dt_rel_thresh=fg_distance_thresh
        )
        self.color_labels_rgb = colorize_labels(self.leaf_labels, seed=self.palette_seed)
        elapsed_ms = (time.time() - t0) * 1000.0
        self.stats_var.set(
            f"Leaves: {self.leaf_count}    compute: {elapsed_ms:.1f} ms    "
            f"(stem_radius={stem_radius}, fg_dist={fg_distance_thresh:.2f}, min_area={min_leaf_area})"
        )
        self._render_only()

    def _save_outputs(self):
        if self.bgr is None or self.leaf_labels is None:
            return
        try:
            stem_radius = int(self.var_radius.get())
            fg_distance_thresh = max(0.0, min(0.99, self.var_fg_x100.get() / 100.0))
            min_leaf_area = int(self.var_min_area.get())

            stem = self.img_path.stem
            stem_out = f"{stem}_r{stem_radius}_thr{int(fg_distance_thresh*100)}_a{min_leaf_area}"
            mask_path   = self.out_dir / f"{stem_out}_leafmask.png"
            labels_path = self.out_dir / f"{stem_out}_labels.png"
            overlay_path= self.out_dir / f"{stem_out}_overlay.png"

            cv2.imwrite(str(mask_path), self.leaf_mask)
            color_bgr = cv2.cvtColor(self.color_labels_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(labels_path), color_bgr)
            cv2.imwrite(str(overlay_path), self.overlay_bgr)

            messagebox.showinfo("Saved", f"Saved:\n{mask_path}\n{labels_path}\n{overlay_path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _shuffle_colors(self):
        if self.leaf_labels is None: return
        self.palette_seed = np.random.randint(0, 1_000_000)
        self.color_labels_rgb = colorize_labels(self.leaf_labels, seed=self.palette_seed)
        self._render_only()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        app = LeafGUI()
        app.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)
