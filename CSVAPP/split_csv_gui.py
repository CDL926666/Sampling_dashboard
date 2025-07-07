import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import pathlib
import chardet

def detect_encoding(file_path, default='utf-8'):
    with open(file_path, 'rb') as f:
        raw = f.read(4096)
    result = chardet.detect(raw)
    return result['encoding'] or default

def split_csv(input_path, chunk_mb=199):
    chunk_bytes = chunk_mb * 1024 * 1024
    encoding = detect_encoding(input_path)
    with open(input_path, 'r', encoding=encoding, errors='replace') as src:
        header = src.readline()
        header_bytes = len(header.encode(encoding))
        buf, size, idx = [], 0, 1
        base = str(pathlib.Path(input_path).with_suffix(''))
        for line in src:
            line_bytes = len(line.encode(encoding))
            if size + line_bytes + header_bytes > chunk_bytes and buf:
                partname = f"{base}_part{idx:03d}.csv"
                with open(partname, 'w', encoding=encoding) as f:
                    f.write(header)
                    f.writelines(buf)
                buf, size, idx = [], 0, idx + 1
            if line_bytes + header_bytes > chunk_bytes:
                partname = f"{base}_part{idx:03d}.csv"
                with open(partname, 'w', encoding=encoding) as f:
                    f.write(header)
                    f.write(line)
                idx += 1
                buf, size = [], 0
            else:
                buf.append(line)
                size += line_bytes
        if buf:
            partname = f"{base}_part{idx:03d}.csv"
            with open(partname, 'w', encoding=encoding) as f:
                f.write(header)
                f.writelines(buf)

def gui_main():
    root = tk.Tk()
    root.title("CSV拆分工具")
    root.geometry("460x180")
    file_path = tk.StringVar()
    size_var = tk.StringVar(value="199")

    def browse():
        fp = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv")])
        if fp:
            file_path.set(fp)
            lbl_status.config(text="已选择文件: " + pathlib.Path(fp).name)

    lbl_file = tk.Label(root, text="请选择CSV文件：")
    lbl_file.pack(pady=3)
    entry = tk.Entry(root, textvariable=file_path, width=45)
    entry.pack()
    btn_browse = tk.Button(root, text="浏览", command=browse)
    btn_browse.pack()

    frame = tk.Frame(root)
    frame.pack(pady=3)
    tk.Label(frame, text="分片最大MB:").pack(side=tk.LEFT)
    size_entry = tk.Entry(frame, textvariable=size_var, width=8)
    size_entry.pack(side=tk.LEFT)

    lbl_status = tk.Label(root, text="")
    lbl_status.pack(pady=3)

    def do_split_gui():
        path = file_path.get()
        try:
            mb = int(size_entry.get())
        except Exception:
            mb = 150
        threading.Thread(target=split_and_alert, args=(path, mb)).start()

    def split_and_alert(path, mb):
        btn_split.config(state=tk.DISABLED)
        lbl_status.config(text="正在拆分，请稍等...")
        split_csv(path, mb)
        lbl_status.config(text="拆分完成！")
        messagebox.showinfo("完成", "拆分完成！")
        btn_split.config(state=tk.NORMAL)

    btn_split = tk.Button(root, text="一键拆分", width=30, height=2, command=do_split_gui)
    btn_split.pack(pady=8)
    root.mainloop()

if __name__ == "__main__":
    gui_main()
