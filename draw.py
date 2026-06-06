import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk
from typing import Callable

from img import bresenham_filled_circle, bresenham_line


class PixelCanvas:
    _palette = [
        '#ffffff',
        '#000000',
        '#ff0000'
    ]

    def __init__(self, frame: tk.Frame, pw: int, ph: int, p: int):
        self.canvas = tk.Canvas(frame, width=pw * p, height=ph * p, bg='white')
        self.pack = self.canvas.pack
        self.pw = pw
        self.ph = ph
        self.p = p
        self.brush_size = 6
        self.palette_idx = 1
        self.last_x = 0
        self.last_y = 0

        self.clear()

        self.canvas.bind('<ButtonPress-1>', self._onpress)
        self.canvas.bind('<B1-Motion>', self._ondrag)

    def _get_xy(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return int(x // self.p), int(y // self.p)

    def _get_rect(self, x: int, y: int):
        rx0 = x * self.p
        ry0 = y * self.p
        rx1 = rx0 + self.p
        ry1 = ry0 + self.p
        return rx0, ry0, rx1, ry1

    def _paint(self, x: int, y: int):
        rx0, ry0, rx1, ry1 = self._get_rect(x, y)
        color = self._palette[self.palette_idx]
        self.canvas.create_rectangle(rx0, ry0, rx1, ry1, fill=color, outline="")

    def _in_canvas(self, x: int, y: int) -> bool:
        h, w = self.control_image.shape
        return 0 <= x < w and 0 <= y < h

    def _press(self, x: int, y: int):
        for lx, ly in bresenham_line(self.last_x, self.last_y, x, y):
            for px, py in bresenham_filled_circle(lx, ly, self.brush_size):
                if self._in_canvas(px, py):
                    self.control_image[py, px] = self.palette_idx
                    self._paint(px, py)

    def _onpress(self, event):
        x, y = self._get_xy(event)
        self.last_x = x
        self.last_y = y
        self._press(x, y)

    def _ondrag(self, event):
        x, y = self._get_xy(event)
        self._press(x, y)
        self.last_x = x
        self.last_y = y

    def clear(self):
        self.control_image = np.zeros((self.ph, self.pw), dtype=np.uint8)
        self.canvas.delete('all')

    def set_brush_size(self, value: int):
        self.brush_size = value

    def set_eraser(self):
        self.palette_idx = 0

    def set_black_brush(self):
        self.palette_idx = 1

    def set_red_brush(self):
        self.palette_idx = 2


class IntSlideEntry:
    def __init__(self, frame: tk.Frame, min_value: int, max_value: int, default_value: int, change_handler: Callable):
        self.value = tk.IntVar(value=default_value)
        self.min_value = min_value
        self.max_value = max_value

        self.frame = tk.Frame(frame)
        self.pack = self.frame.pack
        self.entry = ttk.Entry(self.frame, textvariable=self.value, width=8)
        self.slide = ttk.Scale(self.frame, from_=min_value, to=max_value, orient="horizontal", variable=self.value, length=256)
        self.slide.pack(side='left', padx=2, fill='x')
        self.entry.pack(side='left', padx=2)

        self.change_handler = change_handler
        self.value.trace_add('write', self._on_change)
        self.entry.bind("<FocusOut>", self._on_focus_out)
        self.slide.bind("<ButtonRelease-1>", self._on_scale_change)
        self.slide.bind("<B1-Motion>", self._on_scale_change)  

    def _validate(self, text: str) -> bool:
        if text == "":
            return True
        is_negative = text[0] == '-'
        if is_negative:
            text = text[1:]
        if not text.isnumeric():
            return False
        value = int(text)
        if is_negative:
            value = -value
        return self.min_value <= value <= self.max_value

    def _on_scale_change(self, *args):
        value = int(round(self.slide.get()))
        self.slide.set(value)

    def _on_focus_out(self):
        if self.entry.get() == "":
            self.entry.insert(0, str(self.value.get()))

    def _on_change(self, *args):
        self.change_handler(self.value.get())


def run_draw(predict_img: Callable):
    PIXEL = 2

    root = tk.Tk()
    root.title('ByzKit drawing util')

    top_frame = ttk.Frame(root)
    top_frame.pack(side='top', fill='x', padx=384)
    mid_frame = ttk.Frame(root)
    mid_frame.pack(side='top', fill='both')
    bottom_frame = ttk.Frame(root)
    bottom_frame.pack(side='top', fill='both')

    canvas = PixelCanvas(mid_frame, 894, 196, PIXEL)
    canvas.pack(padx=8, pady=8)

    def get_image():
        ctrl_img = canvas.control_image
        img = np.zeros((ctrl_img.shape[0], ctrl_img.shape[1], 3), dtype=np.uint8)
        for y in range(ctrl_img.shape[0]):
            for x in range(ctrl_img.shape[1]):
                c = ctrl_img[y, x]
                if c == 0:
                    img[y, x] = np.array([255, 255, 255])
                elif c == 1:
                    img[y, x] = np.array([0, 0, 0])
                elif c == 2:
                    img[y, x] = np.array([255, 0, 0])
                else:
                    raise ValueError('invalid color')
        return img

    erase_button = ttk.Button(top_frame, text='erase', command=canvas.clear)
    erase_button.pack(side='left', padx=8, pady=8)

    brush_size_entry = IntSlideEntry(top_frame, 1, 64, canvas.brush_size, canvas.set_brush_size)
    brush_size_entry.pack(side='left', padx=8, pady=8)

    black_brush_btn = ttk.Button(top_frame, command=canvas.set_black_brush, text='black')
    black_brush_btn.pack(side='left', padx=8, pady=8)

    red_brush_btn = ttk.Button(top_frame, command=canvas.set_red_brush, text='red')
    red_brush_btn.pack(side='left', padx=8, pady=8)

    eraser_btn = ttk.Button(top_frame, command=canvas.set_eraser, text='eraser')
    eraser_btn.pack(side='left', padx=8, pady=8)

    image_label = ttk.Label(bottom_frame)
    image_label.pack(padx=8, pady=8)

    def predict_cmd():
        nonlocal image_label
        img = predict_img(get_image())
        pil = Image.fromarray(img, mode='RGB')
        tk_img = ImageTk.PhotoImage(pil)
        image_label.configure(image=tk_img)
        image_label.image = tk_img

    predict_btn = ttk.Button(top_frame, text='predict', command=predict_cmd)
    predict_btn.pack(side='left', padx=8, pady=8)

    predict_cmd()

    root.mainloop()
