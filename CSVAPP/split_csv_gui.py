from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pathlib
import chardet

# 检测文件编码
def detect_encoding(fp: str | pathlib.Path, default="utf-8") -> str:
    with open(fp, "rb") as f:
        raw = f.read(4096)
    res = chardet.detect(raw) or {}
    return res.get("encoding") or default

# 按指定大小拆分CSV文件，每个分片带表头
def split_csv(input_path: str | pathlib.Path, chunk_mb: int = 195, progress_cb=None) -> None:
    chunk_bytes = int(chunk_mb * 1_000_000)
    enc = detect_encoding(input_path)
    base = str(pathlib.Path(input_path).with_suffix(""))
    idx = 1
    buf, size = [], 0

    with open(input_path, "r", encoding=enc, errors="replace") as src:
        header = src.readline()
        head_b = len(header.encode(enc))

        for ln in src:
            ln_b = len(ln.encode(enc))
            # 判断当前缓存内容是否达到分片大小
            if size + ln_b + head_b > chunk_bytes and buf:
                out = f"{base}_part{idx:03d}.csv"
                with open(out, "w", encoding=enc) as f:
                    f.write(header)
                    f.writelines(buf)
                idx += 1
                buf, size = [], 0
                if progress_cb:
                    progress_cb(idx - 1)
            # 处理超长行
            if ln_b + head_b > chunk_bytes:
                out = f"{base}_part{idx:03d}.csv"
                with open(out, "w", encoding=enc) as f:
                    f.write(header)
                    f.write(ln)
                idx += 1
                if progress_cb:
                    progress_cb(idx - 1)
                continue
            buf.append(ln)
            size += ln_b

        # 写入最后一片
        if buf:
            out = f"{base}_part{idx:03d}.csv"
            with open(out, "w", encoding=enc) as f:
                f.write(header)
                f.writelines(buf)
            if progress_cb:
                progress_cb(idx)

class SplitterGUI(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=12)
        self.pack(fill="both", expand=True)
        master.title("CSV Split Tool")
        master.minsize(460, 210)
        self.csv_path = tk.StringVar()
        self.max_mb = tk.IntVar(value=195)
        self._thread = None

        ttk.Label(self, text="CSV file:").grid(row=0, column=0, sticky="w")
        self.entry = ttk.Entry(self, textvariable=self.csv_path, width=48)
        self.entry.grid(row=0, column=1, padx=(4, 0), sticky="we")
        ttk.Button(self, text="Browse...", command=self.browse_file).grid(row=0, column=2, padx=4)

        ttk.Label(self, text="Chunk size (MB):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.spin = ttk.Spinbox(self, from_=10, to=500, increment=5, textvariable=self.max_mb, width=10)
        self.spin.grid(row=1, column=1, sticky="w", pady=(8, 0))

        self.btn_split = ttk.Button(self, text="Start Splitting", command=self.start_split, width=20)
        self.btn_split.grid(row=2, column=0, columnspan=3, pady=14)

        self.prog = ttk.Progressbar(self, mode="determinate")
        self.prog.grid(row=3, column=0, columnspan=3, sticky="we")
        self.status = ttk.Label(self, text="", anchor="w")
        self.status.grid(row=4, column=0, columnspan=3, sticky="we", pady=(4, 0))
        self.columnconfigure(1, weight=1)

    # 文件选择
    def browse_file(self) -> None:
        fp = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if fp:
            self.csv_path.set(fp)
            self.status.config(text=f"Selected: {pathlib.Path(fp).name}")

    # 启动拆分
    def start_split(self) -> None:
        path = pathlib.Path(self.csv_path.get())
        if not path.is_file():
            messagebox.showwarning("Warning", "Please select a valid CSV file.")
            return
        try:
            mb = int(self.max_mb.get())
            if mb <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Warning", "Chunk size must be a positive integer.")
            return

        self.btn_split.config(state=tk.DISABLED)
        self.prog["value"] = 0
        self.status.config(text="Counting lines…")
        self.update_idletasks()

        total_lines = sum(1 for _ in open(path, "rb"))
        self.prog["maximum"] = max(1, total_lines // 1000)

        self._thread = threading.Thread(target=self._worker, args=(path, mb), daemon=True)
        self._thread.start()
        self.after(200, self._poll_thread)

    # 后台拆分
    def _worker(self, path: pathlib.Path, mb: int) -> None:
        cnt = 0
        def step_update(n):
            nonlocal cnt
            cnt = n
            self.prog["value"] = n
            self.status.config(text=f"Completed {n} chunks")
        split_csv(path, mb, progress_cb=step_update)

    # 检查线程状态
    def _poll_thread(self):
        if self._thread and self._thread.is_alive():
            self.after(300, self._poll_thread)
        else:
            self.btn_split.config(state=tk.NORMAL)
            if self._thread:
                self.status.config(text="Splitting completed!")
                messagebox.showinfo("Done", "Splitting completed!")
            self._thread = None

def main():
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    app = SplitterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
