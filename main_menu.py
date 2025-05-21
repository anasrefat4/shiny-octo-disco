import tkinter as tk
import subprocess

class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Medical AI Launcher")
        self.state('zoomed')
        self.configure(bg="#f2f2f2")

        # ESC to exit fullscreen
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))

        # Container Frame
        container = tk.Frame(self, bg="#f2f2f2")
        container.pack(expand=True)

        # Title
        tk.Label(
            container, text="🧬 Medical AI Projects", font=("Helvetica", 32, "bold"),
            bg="#f2f2f2", fg="#222"
        ).pack(pady=30)

        # Buttons
        self.create_menu_button(
            container, "🧠 Brain Tumor Classification", "runbrain.bat.bat", "#0096c7", "Launching Brain Tumor App..."
        )
        self.create_menu_button(
            container, "🎙️ Parkinson's Detection", "runp2.bat.bat", "#38b000", "Launching Parkinson's Predictor..."
        )
        self.create_menu_button(
            container, "🧠 Alzheimer's Risk Prediction", "runalz.bat.bat", "#ff8500", "Launching Alzheimer's Predictor..."
        )
        self.create_menu_button(
            container, "❤️ Heart Disease Prediction", "runheart.bat.bat", "#dc3545", "Launching Heart Disease Predictor..."
        )
        self.create_menu_button(
            container, "🔬 Skin Cancer Classification", "runskin.bat.bat", "#6f42c1", "Launching Skin Cancer Classifier..."
        )
        self.create_menu_button(
            container, "🚪 Exit", None, "#ef233c", is_exit=True
        )

    def create_menu_button(self, parent, text, bat_file, color, message=None, is_exit=False):
        def on_enter(e): btn.config(bg=hover_color)
        def on_leave(e): btn.config(bg=color)

        hover_color = self.darken(color, 0.85)

        def on_click():
            if is_exit:
                self.quit()
            else:
                self.launch_and_delay(bat_file, message)

        btn = tk.Button(
            parent, text=text, font=("Arial", 14), bg=color, fg="white",
            activebackground=hover_color, activeforeground="white",
            width=28, height=1, relief="flat", bd=0, command=on_click
        )

        btn.pack(pady=15)
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    def launch_and_delay(self, bat_file, message):
        # Start the batch file
        subprocess.Popen([bat_file], shell=True)

        # Show popup
        popup = tk.Toplevel(self)
        popup.title("Loading")
        popup.configure(bg="white")

        width, height = 400, 150
        popup.update_idletasks()
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

        tk.Label(popup, text=message, font=("Arial", 14), bg="white", fg="#333").pack(pady=40)
        popup.update()

        # Delay then close
        self.after(6000, lambda: (popup.destroy(), self.destroy()))

    def darken(self, hex_color, factor=0.8):
        """Darkens a hex color by a factor (for hover effect)."""
        hex_color = hex_color.lstrip("#")
        rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        darker_rgb = [int(c * factor) for c in rgb]
        return "#{:02x}{:02x}{:02x}".format(*darker_rgb)

if __name__ == "__main__":
    app = MainMenu()
    app.mainloop()
