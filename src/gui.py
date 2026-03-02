#!/usr/bin/python3

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import subprocess
import shlex
import os
from PIL import Image, ImageTk
import sys
from pathlib import Path
import traceback
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


ROOT_DIR = Path(__file__).resolve().parent
SRC = ROOT_DIR / "src"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC))
from src import sensitive
from options import process_arguments

home_folder = "./"
python_runtime = "python3"




class SensitivityTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.print_help_flag = 0

        # vars
        self.file_var = tk.StringVar()
        self.features = tk.StringVar(value="0")
        self.lb_gap = tk.StringVar(value="0.3")
        self.ub_gap = tk.StringVar(value="0.7")
        self.details_file_var = tk.StringVar()
        self.local_sensitivity_file_var = tk.StringVar()
        self.params_var = tk.StringVar()

        # IMPORTANT: base command for this tab
        self.cmd_var = tk.StringVar(value="./src/sensitive.py")

        self.build_ui()

    def build_ui(self):
        rownum = 0

        tk.Label(self, text="Model file:").grid(row=rownum, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.file_var, width=50).grid(row=rownum, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse...", command=self.browse_file).grid(row=rownum, column=2, padx=5, pady=5)
        rownum += 1

        tk.Label(self, text="Sensitive feature list:").grid(row=rownum, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.features, width=50).grid(row=rownum, column=1, padx=5, pady=5)
        tk.Label(self, text="e.g. 2 5").grid(row=rownum, column=2, sticky="w", padx=5, pady=5)
        rownum += 1

        tk.Label(self, text="Gap lower bound:").grid(row=rownum, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(self, textvariable=self.lb_gap, values=["0.1","0.2","0.3","0.4"], width=47, state="readonly")\
            .grid(row=rownum, column=1, padx=5, pady=5)
        rownum += 1

        tk.Label(self, text="Gap upper bound:").grid(row=rownum, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(self, textvariable=self.ub_gap, values=["0.6","0.7","0.8","0.9"], width=47, state="readonly")\
            .grid(row=rownum, column=1, padx=5, pady=5)
        rownum += 1
        
        # -------- Details File selection --------
        tk.Label(self, text="Details file (optional):").grid(row=rownum, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.details_file_var, width=50).grid(row=rownum, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse...", command=self.browse_details_file).grid(row=rownum, column=2, padx=5, pady=5)
        rownum += 1
        
        # -------- Local sensitivity search --------
        tk.Label(self, text="Sample file for local sensitivity (optional):").grid(row=rownum, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.local_sensitivity_file_var, width=50).grid(row=rownum, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse...", command=self.browse_local_sensitivity_file).grid(row=rownum, column=2, padx=5, pady=5)
        rownum += 1

        # -------- Extra parameters --------
        tk.Label(self, text="Extra parameters (optional):").grid(row=rownum, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.params_var, width=50).grid(row=rownum, column=1, padx=5, pady=5)
        rownum += 1

        # ... keep moving your "details file", "local sensitivity file", "extra parameters", etc here ...

        tk.Label(self, text="Base command:").grid(row=rownum, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.cmd_var, width=50).grid(row=rownum, column=1, padx=5, pady=5)
        rownum += 1

        tk.Button(self, text="Run", command=self.run_command, bg="#4CAF50", fg="white")\
            .grid(row=rownum, column=1, pady=10, sticky="w")
        tk.Button(self, text="Print help", command=self.print_help, bg="#4CAF50", fg="white")\
            .grid(row=rownum, column=2, pady=10, sticky="w")
        rownum += 1

        tk.Label(self, text="Output:").grid(row=rownum, column=0, sticky="ne", padx=5, pady=5)
        output_frame = tk.Frame(self)
        output_frame.grid(row=rownum, column=1, columnspan=2, padx=5, pady=5, sticky="nsew")
        rownum += 1
        
        self.output = tk.Text(output_frame, width=70, height=20, wrap="none")
        self.output.grid(row=0, column=0, sticky="nsew")
        v_scroll = tk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = tk.Scrollbar(output_frame, orient=tk.HORIZONTAL, command=self.output.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.output.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # allow resize
        output_frame.grid_rowconfigure(0, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

    def write(self, text):
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.update_idletasks()

    def flush(self):  # for stdout redirect compatibility
        pass

    def browse_file(self):
        fn = filedialog.askopenfilename()
        if fn:
            self.file_var.set(fn)

    def print_help(self):
        self.print_help_flag = 1
        self.run_command()
        
    def browse_details_file(self):
        fn = filedialog.askopenfilename()
        if fn:
            self.details_file_var.set(fn)

    def browse_local_sensitivity_file(self):
        fn = filedialog.askopenfilename()
        if fn:
            self.local_sensitivity_file_var.set(fn)

    def run_command(self):
        # use the fixed argv approach I showed earlier (help mode works)
        script_path = self.cmd_var.get().strip()
        file_path = self.file_var.get().strip()
        params = self.params_var.get().strip()

        argv = [script_path]
        if self.print_help_flag:
            argv += ["--help"]
            self.print_help_flag = 0
        else:
            if not file_path or not os.path.isfile(file_path):
                messagebox.showerror("Error", "Please select a valid input file.")
                return
            features = self.features.get().strip()
            if not features:
                messagebox.showerror("Error", "Please provide at least one sensitive feature index.")
                return

            argv += [file_path, "--all_opt", "--features", *features.split(),
                     "--output_gap", self.lb_gap.get(), self.ub_gap.get()]
            
            details_file = self.details_file_var.get().strip()
            if details_file:
                argv += ["--details", details_file]
                
            local_file = self.local_sensitivity_file_var.get().strip()
            if local_file:
                argv += ["--local_check_file", local_file]

            # add optional files here like before...
            if params:
                argv += shlex.split(params)

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "> " + "python3 " + " ".join(shlex.quote(a) for a in argv) + "\n\n")

        old_argv = sys.argv
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.argv = argv
            self.encoding = sys.stdout.encoding
            sys.stdout = self
            sys.stderr = self
            args, options = process_arguments()
            sensitive.main(args, options)
        except SystemExit:
            pass
        except Exception:
            self.write(traceback.format_exc())
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class MonitorTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.print_help_flag = 0

        self.file_var = tk.StringVar()
        self.featurefile = tk.StringVar()
        self.epsilon_var = tk.StringVar(value="0.2")
        self.params_var = tk.StringVar()
        self.predcolname = tk.StringVar(value="pred")
        self.cmd_var = tk.StringVar(value="./src/sensitive.py")  # adjust name/path

        self.build_ui()

    def build_ui(self):
        row = 0
        tk.Label(self, text="Input csv:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.file_var, width=50).grid(row=row, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse...", command=self.browse_file).grid(row=row, column=2, padx=5, pady=5)
        row += 1
        
        tk.Label(self, text="featurelist file:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.featurefile, width=50).grid(row=row, column=1, padx=5, pady=5)
        tk.Button(self, text="Browse...", command=self.browse_featurefile).grid(row=row, column=2, padx=5, pady=5)
        row += 1
        
        tk.Label(self, text="epsilon:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(
            self,
            textvariable=self.epsilon_var,
            values=["0.1","0.2","0.3","0.4"],
            width=47,
            state="normal"   # <-- allow typing
        ).grid(row=row, column=1, padx=5, pady=5)

        row += 1
        
        tk.Label(self, text="prediction colname:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.predcolname, width=50).grid(row=row, column=1, padx=5, pady=5)
        row += 1
        
        # tk.Label(self, text="Extra parameters (optional):").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        # tk.Entry(self, textvariable=self.params_var, width=50).grid(row=row, column=1, padx=5, pady=5)
        # row += 1

        tk.Label(self, text="Base command:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self, textvariable=self.cmd_var, width=50).grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Button(self, text="Run", command=self.run_command, bg="#4CAF50", fg="white")\
            .grid(row=row, column=1, pady=10, sticky="w")
        tk.Button(self, text="Print help", command=self.print_help, bg="#4CAF50", fg="white")\
            .grid(row=row, column=2, pady=10, sticky="w")
        row += 1

        # tk.Label(self, text="Output:").grid(row=row, column=0, sticky="ne", padx=5, pady=5)
        # self.output = tk.Text(self, width=70, height=20, wrap="none")
        # self.output.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="nsew")
        tk.Label(self, text="Output:").grid(row=row, column=0, sticky="ne", padx=5, pady=5)
        output_frame = tk.Frame(self)
        output_frame.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="nsew")
        row += 1
        
        self.output = tk.Text(output_frame, width=70, height=20, wrap="none")
        self.output.grid(row=0, column=0, sticky="nsew")
        v_scroll = tk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = tk.Scrollbar(output_frame, orient=tk.HORIZONTAL, command=self.output.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.output.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.grid_rowconfigure(row, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def write(self, text):
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.update_idletasks()

    def flush(self):
        pass

    def browse_file(self):
        fn = filedialog.askopenfilename()
        if fn:
            self.file_var.set(fn)
    def browse_featurefile(self):
        fn = filedialog.askopenfilename()
        if fn:
            self.featurefile.set(fn)
            
    def print_help(self):
        self.print_help_flag = 1
        self.run_command()

    def run_command(self):
        script_path = self.cmd_var.get().strip()
        file_path = self.file_var.get().strip()
        featurefile_path = self.featurefile.get().strip()
        params = self.params_var.get().strip()

        # argv = [script_path]
        argv = [script_path]
        if self.print_help_flag:
            argv += ["--help"]
            self.print_help_flag = 0
        else:
            if not file_path or not os.path.isfile(file_path):
                messagebox.showerror("Error", "Please select a valid input file.")
                return
            argv += [file_path]
            argv += ["--solver", "monitor"]
            argv += ["--cfeaturefile", self.featurefile.get(),
                     '--epsilon', self.epsilon_var.get(),
                     '--predcolname', self.predcolname.get()
                     ]
            if params:
                argv += shlex.split(params)

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "> " + "python3 " + " ".join(shlex.quote(a) for a in argv) + "\n\n")

        # If monitor is implemented like sensitive (importable main), do that here.
        # Otherwise, easiest is subprocess execution. Placeholder:
        try:
            import monitor  
            old_argv = sys.argv
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.argv = argv
            sys.stdout = self
            sys.stderr = self
            args, options = process_arguments()
            sensitive.main(args, options)
        except ImportError:
            self.write("monitor module not found. Update import path (e.g., from src import monitor).\n")
        except SystemExit:
            pass
        except Exception:
            self.write(traceback.format_exc())
        finally:
            try:
                sys.argv = old_argv
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            except Exception:
                pass


def main():
    root = tk.Tk()
    root.title("TreeVerifier - Sensitivity Analysis and Monitor")
    
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    container = ttk.Frame(root)
    container.grid(row=0, column=0, sticky="nsew")
    container.grid_rowconfigure(1, weight=1)   # notebook expands
    container.grid_columnconfigure(0, weight=1)

    # ================= HEADER (top) =================
    header = ttk.Frame(container)
    header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
    header.grid_columnconfigure(1, weight=1)

    img_path = ROOT_DIR / "../images/verifylogo.png"
    img = Image.open(img_path)
    img = img.resize((80, 80), Image.LANCZOS)   # adjust size as you like
    root.logo_img = ImageTk.PhotoImage(img)     # keep reference on root (important!)

    ttk.Label(header, image=root.logo_img).grid(row=0, column=0, sticky="w",padx=(0, 22))

    ttk.Label(
        header,
        text="TreeVerifier built by Indian institute of Technology Bombay",
        font=("TkDefaultFont", 14, "bold"),
    ).grid(row=0, column=1, sticky="w", padx=(12, 0))
    
    right_path = ROOT_DIR / "../images/iitb.png"  
    right_img = Image.open(right_path).resize((90, 90), Image.LANCZOS)
    root.right_logo_img = ImageTk.PhotoImage(right_img)  # KEEP REFERENCE

    ttk.Label(header, image=root.right_logo_img).grid(row=0, column=2, sticky="e",padx=(0, 22))
        
    # ================= NOTEBOOK (below header) =================
    notebook = ttk.Notebook(container)
    notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    # notebook = ttk.Notebook(root)
    # notebook.pack(fill="both", expand=True)

    tab1 = SensitivityTab(notebook)
    tab2 = MonitorTab(notebook)

    notebook.add(tab1, text="Sensitivity")
    notebook.add(tab2, text="Monitoring")
    
    # ================= FOOTER (bottom) =================
    footer = ttk.Frame(container)
    footer.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

    footer.grid_columnconfigure(0, weight=1)

    sbihub_path = ROOT_DIR / "../images/sbihub.png"
    sbimg = Image.open(sbihub_path).resize((300, 50), Image.LANCZOS)
    root.sbihub_img = ImageTk.PhotoImage(sbimg)

    ttk.Label(footer, text="Supported by").grid(row=0, column=1, sticky="ew", pady=(5, 2))
    ttk.Label(footer, image=root.sbihub_img).grid(row=1, column=1, sticky="e", pady=(0, 5))

    root.mainloop()

if __name__ == "__main__":
    main()
