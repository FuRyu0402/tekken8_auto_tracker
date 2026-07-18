"""Tekken 8 character HUD template cropper.

The reusable functions in the first half of this module deliberately have no
Tk dependency.  The GUI is only constructed by ``CharacterTemplateCropper``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import cv2
import numpy as np

Side = Literal["left", "right"]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_CONFIG = {
    "version": 1,
    "source_resolution": {"width": 1920, "height": 1080},
    "roi": {
        "left": {"x": 40, "y": 30, "width": 260, "height": 72},
        "right": {"x": 1620, "y": 30, "width": 260, "height": 72},
    },
    "display": {"zoom": 1.0, "mode": "color"},
    "paths": {
        "source_root": "template_sources",
        "output_root": "templates/characters",
    },
}


class CropperError(ValueError):
    """An expected, user-facing cropper error."""


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_mapping(cls, value: object) -> "ROI":
        if not isinstance(value, dict):
            raise CropperError("ROI must be an object")
        try:
            return cls(*(int(value[key]) for key in ("x", "y", "width", "height")))
        except (KeyError, TypeError, ValueError) as exc:
            raise CropperError("ROI must contain integer x, y, width and height") from exc


def validate_roi(roi: ROI, image_width: int, image_height: int) -> None:
    if image_width <= 0 or image_height <= 0:
        raise CropperError("Image dimensions must be positive")
    if roi.width <= 0 or roi.height <= 0:
        raise CropperError("ROI width and height must be greater than zero")
    if roi.x < 0 or roi.y < 0:
        raise CropperError("ROI must start inside the image")
    if roi.x + roi.width > image_width or roi.y + roi.height > image_height:
        raise CropperError("ROI extends outside the image")


def clamp_roi(roi: ROI, image_width: int, image_height: int) -> ROI:
    """Return the largest valid equivalent ROI, keeping at least one pixel."""
    if image_width <= 0 or image_height <= 0:
        raise CropperError("Image dimensions must be positive")
    x = min(max(roi.x, 0), image_width - 1)
    y = min(max(roi.y, 0), image_height - 1)
    width = min(max(roi.width, 1), image_width - x)
    height = min(max(roi.height, 1), image_height - y)
    return ROI(x, y, width, height)


def crop_image(image: np.ndarray, roi: ROI) -> np.ndarray:
    if image is None or image.ndim not in (2, 3):
        raise CropperError("Invalid image")
    height, width = image.shape[:2]
    validate_roi(roi, width, height)
    return image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width].copy()


def adjust_roi(
    roi: ROI,
    dx: int = 0,
    dy: int = 0,
    dw: int = 0,
    dh: int = 0,
    bounds: tuple[int, int] | None = None,
) -> ROI:
    changed = ROI(roi.x + dx, roi.y + dy, roi.width + dw, roi.height + dh)
    return clamp_roi(changed, *bounds) if bounds else changed


def mirror_roi(roi: ROI, image_width: int) -> ROI:
    """Mirror an ROI horizontally around the source image centre."""
    return ROI(image_width - roi.x - roi.width, roi.y, roi.width, roi.height)


def copy_roi(rois: dict[Side, ROI], source: Side, target: Side) -> dict[Side, ROI]:
    copied = dict(rois)
    copied[target] = rois[source]
    return copied


def sanitize_character_name(value: str) -> str:
    """Create a stable, portable template stem from user input."""
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value:
        raise CropperError("Character name is empty or contains no usable characters")
    return value


def infer_character_name(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    stem = re.sub(r"(?:[_\-\s]+(?:left|right))?(?:[_\-\s]*\d+)?$", "", stem)
    stem = re.sub(r"(?:[_\-\s]+(?:left|right))$", "", stem)
    return sanitize_character_name(stem)


def build_output_path(output_root: Path, side: Side, character_name: str) -> Path:
    return output_root / side / f"{sanitize_character_name(character_name)}.png"


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise CropperError(f"Folder does not exist: {folder}")
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )


def read_image(path: Path) -> np.ndarray:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError) as exc:
        raise CropperError(f"Could not read image: {path}") from exc
    if image is None:
        raise CropperError(f"Unsupported or corrupt image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise CropperError(f"Could not encode PNG: {path}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise CropperError(f"Could not save PNG: {path}") from exc


def _valid_config(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise CropperError("Configuration root must be an object")
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    resolution = raw.get("source_resolution", {})
    width = int(resolution.get("width", 1920))
    height = int(resolution.get("height", 1080))
    if width <= 0 or height <= 0:
        raise CropperError("Invalid source resolution")
    config["source_resolution"] = {"width": width, "height": height}
    raw_roi = raw.get("roi", raw)
    for side in ("left", "right"):
        roi = ROI.from_mapping(raw_roi.get(side, config["roi"][side]))
        validate_roi(roi, width, height)
        config["roi"][side] = asdict(roi)
    display = raw.get("display", {})
    zoom = float(display.get("zoom", 1.0))
    mode = display.get("mode", "color")
    if not 0.05 <= zoom <= 8.0 or mode not in ("color", "grayscale"):
        raise CropperError("Invalid display settings")
    config["display"] = {"zoom": zoom, "mode": mode}
    paths = raw.get("paths", {})
    for key in ("source_root", "output_root"):
        value = paths.get(key, config["paths"][key])
        if not isinstance(value, str) or not value.strip():
            raise CropperError(f"Invalid paths.{key}")
        config["paths"][key] = value
    config["version"] = 1
    return config


def load_config(path: Path) -> tuple[dict, str]:
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG)), "Config not found; defaults loaded"
    try:
        with path.open("r", encoding="utf-8") as stream:
            return _valid_config(json.load(stream)), f"Config loaded: {path}"
    except (OSError, json.JSONDecodeError, CropperError, TypeError, ValueError) as exc:
        return json.loads(json.dumps(DEFAULT_CONFIG)), f"Invalid config; defaults loaded ({exc})"


def save_config(path: Path, config: dict) -> None:
    validated = _valid_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(validated, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def resolve_config_path(project_root: Path, configured: str) -> Path:
    candidate = Path(configured)
    return candidate if candidate.is_absolute() else project_root / candidate


def template_status(output_root: Path, side: Side, name: str) -> str:
    return "saved" if build_output_path(output_root, side, name).exists() else "unsaved"


@dataclass(frozen=True)
class BatchItem:
    source: Path
    output: Path
    name: str
    exists: bool


def prepare_batch(
    sources: Iterable[Path], output_root: Path, side: Side
) -> tuple[list[BatchItem], list[tuple[Path, str]]]:
    items: list[BatchItem] = []
    rejected: list[tuple[Path, str]] = []
    outputs: set[Path] = set()
    for source in sources:
        try:
            name = infer_character_name(source)
            output = build_output_path(output_root, side, name)
            if output in outputs:
                raise CropperError(f"Duplicate output name: {output.name}")
            outputs.add(output)
            items.append(BatchItem(source, output, name, output.exists()))
        except CropperError as exc:
            rejected.append((source, str(exc)))
    return items, rejected


def run_batch(
    items: Sequence[BatchItem],
    roi: ROI,
    overwrite: bool = False,
    reader: Callable[[Path], np.ndarray] = read_image,
    writer: Callable[[Path, np.ndarray], None] = write_png,
) -> tuple[int, int, list[str]]:
    saved = skipped = 0
    errors: list[str] = []
    for item in items:
        if item.output.exists() and not overwrite:
            skipped += 1
            continue
        try:
            writer(item.output, crop_image(reader(item.source), roi))
            saved += 1
        except (CropperError, OSError) as exc:
            errors.append(f"{item.source.name}: {exc}")
    return saved, skipped, errors


class CharacterTemplateCropper:
    """Tk GUI. Imports are local so core tests work without a display."""

    def __init__(self, root, project_root: Path, config_path: Path):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root = root
        self.project_root = project_root
        self.config_path = config_path
        self.config, message = load_config(config_path)
        self.rois = {side: ROI.from_mapping(self.config["roi"][side]) for side in ("left", "right")}
        self.side: Side = "left"
        self.paths: list[Path] = []
        self.index = -1
        self.image: np.ndarray | None = None
        self.photo = None
        self.zoom = float(self.config["display"]["zoom"])
        self.mode = self.config["display"]["mode"]
        self.drag_start: tuple[float, float] | None = None
        self.roi_dirty = False
        self.name_dirty = False
        self.saved_this_session: set[tuple[Path, Side, str]] = set()

        root.title("Tekken 8 Character Template Cropper")
        root.geometry("1280x820")
        root.minsize(900, 600)
        self._build_ui()
        self.status_var.set(message)
        root.bind("<KeyPress>", self._on_key)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._refresh_all()

    @property
    def output_root(self) -> Path:
        return resolve_config_path(self.project_root, self.output_var.get())

    def _build_ui(self):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="Open image", command=self.open_image).pack(side="left")
        ttk.Button(bar, text="Open folder", command=self.open_folder).pack(side="left", padx=4)
        for text, command in (("|<", self.first), ("<", self.previous), (">", self.next), (">|", self.last)):
            ttk.Button(bar, text=text, width=4, command=command).pack(side="left", padx=1)
        self.counter_var = tk.StringVar(value="0 / 0")
        ttk.Label(bar, textvariable=self.counter_var, width=12).pack(side="left", padx=5)
        ttk.Button(bar, text="-", width=3, command=lambda: self.set_zoom(self.zoom / 1.25)).pack(side="left")
        ttk.Button(bar, text="100%", command=lambda: self.set_zoom(1.0)).pack(side="left")
        ttk.Button(bar, text="+", width=3, command=lambda: self.set_zoom(self.zoom * 1.25)).pack(side="left")
        ttk.Button(bar, text="Fit", command=self.fit).pack(side="left", padx=3)
        self.mode_var = tk.StringVar(value=self.mode)
        ttk.Combobox(bar, textvariable=self.mode_var, values=("color", "grayscale"), state="readonly", width=10).pack(side="left")
        self.mode_var.trace_add("write", lambda *_: self._change_mode())

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)
        canvas_frame = ttk.Frame(body)
        panel = ttk.Frame(body, padding=8, width=310)
        body.add(canvas_frame, weight=4)
        body.add(panel, weight=1)
        self.canvas = tk.Canvas(canvas_frame, background="#202124", highlightthickness=0)
        xbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        xbar.grid(row=1, column=0, sticky="ew")
        ybar.grid(row=0, column=1, sticky="ns")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self._drag_begin)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)

        ttk.Label(panel, text="Side").pack(anchor="w")
        self.side_var = tk.StringVar(value="left")
        sides = ttk.Frame(panel)
        sides.pack(fill="x")
        for side in ("left", "right"):
            ttk.Radiobutton(sides, text=side.title(), value=side, variable=self.side_var, command=self._change_side).pack(side="left")
        copybar = ttk.Frame(panel)
        copybar.pack(fill="x", pady=4)
        ttk.Button(copybar, text="Left → Right", command=lambda: self._copy("left", "right")).pack(side="left")
        ttk.Button(copybar, text="Right → Left", command=lambda: self._copy("right", "left")).pack(side="left", padx=3)
        ttk.Button(panel, text="Mirror to other side", command=self._mirror).pack(fill="x")

        ttk.Separator(panel).pack(fill="x", pady=8)
        ttk.Label(panel, text="Character name").pack(anchor="w")
        self.name_var = tk.StringVar()
        entry = ttk.Entry(panel, textvariable=self.name_var)
        entry.pack(fill="x")
        entry.bind("<KeyRelease>", lambda _event: self._name_changed())
        ttk.Label(panel, text="ROI (source pixels)").pack(anchor="w", pady=(8, 0))
        self.roi_var = tk.StringVar()
        ttk.Label(panel, textvariable=self.roi_var).pack(anchor="w")
        self.saved_var = tk.StringVar()
        ttk.Label(panel, textvariable=self.saved_var).pack(anchor="w", pady=4)

        ttk.Label(panel, text="Crop preview").pack(anchor="w", pady=(8, 2))
        self.preview = tk.Canvas(panel, width=280, height=120, background="#111", highlightthickness=1)
        self.preview.pack(fill="x")
        self.preview_photo = None
        ttk.Button(panel, text="Save ROI settings", command=self.save_settings).pack(fill="x", pady=(8, 2))
        ttk.Button(panel, text="Save template", command=self.save_template).pack(fill="x")
        ttk.Button(panel, text="Batch crop current folder", command=self.batch_crop).pack(fill="x", pady=2)

        ttk.Label(panel, text="Output root").pack(anchor="w", pady=(8, 0))
        self.output_var = tk.StringVar(value=self.config["paths"]["output_root"])
        ttk.Entry(panel, textvariable=self.output_var).pack(fill="x")
        ttk.Button(panel, text="Choose output folder", command=self.choose_output).pack(fill="x", pady=2)

        self.status_var = tk.StringVar()
        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=4)
        status.pack(fill="x")

    def _ask_discard(self) -> bool:
        from tkinter import messagebox
        if not (self.roi_dirty or self.name_dirty):
            return True
        return messagebox.askyesno("Unsaved changes", "Discard unsaved ROI/name changes and continue?")

    def open_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path and self._ask_discard():
            self.paths, self.index = [Path(path)], 0
            self._load_current()

    def open_folder(self):
        from tkinter import filedialog, messagebox
        folder = filedialog.askdirectory(initialdir=resolve_config_path(self.project_root, self.config["paths"]["source_root"]))
        if not folder or not self._ask_discard():
            return
        try:
            self.paths = list_images(Path(folder))
            if not self.paths:
                raise CropperError("No PNG/JPG/JPEG images found")
            self.index = 0
            self._load_current()
        except CropperError as exc:
            messagebox.showerror("Open folder", str(exc))
            self.status_var.set(str(exc))

    def _load_current(self):
        from tkinter import messagebox
        try:
            self.image = read_image(self.paths[self.index])
            self.name_var.set(infer_character_name(self.paths[self.index]))
            for side in ("left", "right"):
                self.rois[side] = clamp_roi(
                    self.rois[side], self.image.shape[1], self.image.shape[0]
                )
            self.roi_dirty = self.name_dirty = False
            self.status_var.set(f"Loaded: {self.paths[self.index]}")
            self._refresh_all()
        except CropperError as exc:
            messagebox.showerror("Read error", str(exc))
            self.status_var.set(str(exc))

    def _navigate(self, index: int):
        if self.paths and self._ask_discard():
            self.index = max(0, min(index, len(self.paths) - 1))
            self._load_current()

    def first(self): self._navigate(0)
    def previous(self): self._navigate(self.index - 1)
    def next(self): self._navigate(self.index + 1)
    def last(self): self._navigate(len(self.paths) - 1)

    def _change_side(self):
        self.side = self.side_var.get()
        if self.image is not None:
            self.rois[self.side] = clamp_roi(self.rois[self.side], self.image.shape[1], self.image.shape[0])
        self._refresh_all()

    def _change_mode(self):
        self.mode = self.mode_var.get()
        self._refresh_all()

    def _copy(self, source: Side, target: Side):
        self.rois = copy_roi(self.rois, source, target)
        if self.image is not None:
            self.rois[target] = clamp_roi(self.rois[target], self.image.shape[1], self.image.shape[0])
        self.side = target
        self.side_var.set(target)
        self.roi_dirty = True
        self._refresh_all()

    def _mirror(self):
        if self.image is None:
            return
        target: Side = "right" if self.side == "left" else "left"
        self.rois[target] = clamp_roi(mirror_roi(self.rois[self.side], self.image.shape[1]), self.image.shape[1], self.image.shape[0])
        self.side = target
        self.side_var.set(target)
        self.roi_dirty = True
        self._refresh_all()

    def _name_changed(self):
        self.name_dirty = True
        self._refresh_status()

    def set_zoom(self, value: float):
        self.zoom = min(max(value, 0.05), 8.0)
        self._render()

    def fit(self):
        if self.image is None:
            return
        self.root.update_idletasks()
        width = max(self.canvas.winfo_width() - 20, 1)
        height = max(self.canvas.winfo_height() - 20, 1)
        self.set_zoom(min(width / self.image.shape[1], height / self.image.shape[0]))

    def _display_image(self, image: np.ndarray) -> np.ndarray:
        if self.mode == "grayscale":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _to_photo(self, rgb: np.ndarray, scale: float):
        from tkinter import PhotoImage
        width = max(1, round(rgb.shape[1] * scale))
        height = max(1, round(rgb.shape[0] * scale))
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
        ok, data = cv2.imencode(".png", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        if not ok:
            raise CropperError("Preview encoding failed")
        return PhotoImage(data=data.tobytes())

    def _render(self):
        self.canvas.delete("all")
        if self.image is None:
            return
        self.photo = self._to_photo(self._display_image(self.image), self.zoom)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        roi = self.rois[self.side]
        x1, y1 = roi.x * self.zoom, roi.y * self.zoom
        x2, y2 = (roi.x + roi.width) * self.zoom, (roi.y + roi.height) * self.zoom
        color = "#00e5ff" if self.side == "left" else "#ffab40"
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#000", width=5)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
        self.canvas.create_text(x1 + 5, max(10, y1 - 10), text=self.side.upper(), fill=color, anchor="w", font=("TkDefaultFont", 10, "bold"))
        self.canvas.configure(scrollregion=(0, 0, self.photo.width(), self.photo.height()))

    def _refresh_preview(self):
        self.preview.delete("all")
        if self.image is None:
            return
        try:
            crop = self._display_image(crop_image(self.image, self.rois[self.side]))
            scale = min(280 / crop.shape[1], 120 / crop.shape[0], 4.0)
            self.preview_photo = self._to_photo(crop, scale)
            self.preview.create_image(140, 60, image=self.preview_photo, anchor="center")
        except CropperError:
            pass

    def _refresh_status(self):
        roi = self.rois[self.side]
        self.roi_var.set(f"x={roi.x}, y={roi.y}, width={roi.width}, height={roi.height}")
        try:
            name = sanitize_character_name(self.name_var.get())
            output = build_output_path(self.output_root, self.side, name)
            key = (output.resolve(), self.side, name)
            if key in self.saved_this_session:
                state = "saved"
            elif output.exists():
                state = "same-name file exists"
            else:
                state = "unsaved"
            self.saved_var.set(f"Status: {state}\n{output}")
        except CropperError:
            self.saved_var.set("Status: invalid/empty character name")

    def _refresh_all(self):
        self.counter_var.set(f"{self.index + 1 if self.index >= 0 else 0} / {len(self.paths)}")
        self._render()
        self._refresh_preview()
        self._refresh_status()

    def _canvas_point(self, event) -> tuple[int, int]:
        x = round(self.canvas.canvasx(event.x) / self.zoom)
        y = round(self.canvas.canvasy(event.y) / self.zoom)
        if self.image is None:
            return 0, 0
        return max(0, min(x, self.image.shape[1] - 1)), max(0, min(y, self.image.shape[0] - 1))

    def _drag_begin(self, event):
        if self.image is not None:
            self.drag_start = self._canvas_point(event)

    def _drag_move(self, event):
        if self.drag_start is None or self.image is None:
            return
        x, y = self._canvas_point(event)
        sx, sy = self.drag_start
        self.rois[self.side] = ROI(min(sx, x), min(sy, y), max(1, abs(x - sx)), max(1, abs(y - sy)))
        self.roi_dirty = True
        self._refresh_all()

    def _drag_end(self, _event):
        self.drag_start = None

    def _on_key(self, event):
        if self.image is None or event.keysym not in ("Left", "Right", "Up", "Down"):
            return
        step = 10 if (event.state & 0x0001) else 1
        dx = (-step if event.keysym == "Left" else step if event.keysym == "Right" else 0)
        dy = (-step if event.keysym == "Up" else step if event.keysym == "Down" else 0)
        if event.state & 0x0004:
            self.rois[self.side] = adjust_roi(
                self.rois[self.side], dw=dx, dh=dy,
                bounds=(self.image.shape[1], self.image.shape[0]),
            )
        else:
            self.rois[self.side] = adjust_roi(
                self.rois[self.side], dx=dx, dy=dy,
                bounds=(self.image.shape[1], self.image.shape[0]),
            )
        self.roi_dirty = True
        self._refresh_all()
        return "break"

    def save_settings(self):
        from tkinter import messagebox
        if self.image is not None:
            width, height = self.image.shape[1], self.image.shape[0]
        else:
            width = self.config["source_resolution"]["width"]
            height = self.config["source_resolution"]["height"]
        config = {
            "version": 1,
            "source_resolution": {"width": width, "height": height},
            "roi": {side: asdict(self.rois[side]) for side in ("left", "right")},
            "display": {"zoom": self.zoom, "mode": self.mode},
            "paths": {
                "source_root": self.config["paths"]["source_root"],
                "output_root": self.output_var.get(),
            },
        }
        try:
            save_config(self.config_path, config)
            self.config = config
            self.roi_dirty = False
            self.status_var.set(f"Settings saved: {self.config_path}")
        except (OSError, CropperError) as exc:
            messagebox.showerror("Save settings", str(exc))
            self.status_var.set(f"Settings save failed: {exc}")

    def save_template(self):
        from tkinter import messagebox
        if self.image is None:
            messagebox.showerror("Save template", "Open an image first")
            return
        try:
            output = build_output_path(self.output_root, self.side, self.name_var.get())
            validate_roi(self.rois[self.side], self.image.shape[1], self.image.shape[0])
            if output.exists() and not messagebox.askyesno("Overwrite?", f"{output.name} already exists.\nOverwrite it?"):
                self.status_var.set("Overwrite cancelled")
                return
            write_png(output, crop_image(self.image, self.rois[self.side]))
            name = sanitize_character_name(self.name_var.get())
            self.name_var.set(name)
            self.saved_this_session.add((output.resolve(), self.side, name))
            self.name_dirty = False
            self.status_var.set(f"Template saved: {output}")
            self._refresh_status()
        except CropperError as exc:
            messagebox.showerror("Save template", str(exc))
            self.status_var.set(f"Template save failed: {exc}")

    def choose_output(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(initialdir=self.output_root)
        if path:
            try:
                relative = Path(path).resolve().relative_to(self.project_root.resolve())
                self.output_var.set(relative.as_posix())
            except ValueError:
                self.output_var.set(path)
            self._refresh_status()

    def batch_crop(self):
        from tkinter import messagebox
        if not self.paths:
            messagebox.showerror("Batch crop", "Open a folder first")
            return
        items, rejected = prepare_batch(self.paths, self.output_root, self.side)
        existing = sum(item.exists for item in items)
        preview = "\n".join(f"{item.source.name} → {item.output.name}" for item in items[:12])
        if len(items) > 12:
            preview += f"\n… and {len(items) - 12} more"
        if rejected:
            preview += f"\n\nRejected: {len(rejected)}"
        overwrite = False
        if existing:
            choice = messagebox.askyesnocancel(
                "Existing templates",
                f"{existing} output file(s) already exist.\nYes: overwrite\nNo: skip\nCancel: stop\n\n{preview}",
            )
            if choice is None:
                self.status_var.set("Batch crop cancelled")
                return
            overwrite = choice
        elif not messagebox.askokcancel("Confirm batch crop", f"Create {len(items)} template(s)?\n\n{preview}"):
            self.status_var.set("Batch crop cancelled")
            return
        saved, skipped, errors = run_batch(items, self.rois[self.side], overwrite)
        summary = f"Batch complete: saved {saved}, skipped {skipped}, failed {len(errors)}"
        self.status_var.set(summary)
        if errors:
            messagebox.showwarning("Batch crop", summary + "\n\n" + "\n".join(errors[:10]))
        else:
            messagebox.showinfo("Batch crop", summary)
        self._refresh_status()

    def _close(self):
        if self._ask_discard():
            self.root.destroy()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crop Tekken 8 HUD character templates")
    parser.add_argument("--config", type=Path, help="ROI config path")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config or project_root / "config" / "character_roi.json"
    import tkinter as tk
    root = tk.Tk()
    CharacterTemplateCropper(root, project_root, config_path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
